import torch
import numpy as np
import json
import os
import logging
# Lazy import: from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.model_selection import train_test_split as sk_train_test_split

# ========== TinyNN ==========
class TinyMLP(torch.nn.Module):
    def __init__(self, in_dim=5, hidden1=64, hidden2=32, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_tiny_mlp(X, y, *, max_epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4, patience=10, device=None, seed=42):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    col_means = np.nanmean(X, axis=0).astype(np.float32)
    X = np.where(np.isnan(X), col_means, X).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)

    from sklearn.model_selection import train_test_split as sk_train_test_split
    X_tr, X_va, y_tr, y_va = sk_train_test_split(X, y, test_size=0.2, random_state=seed)

    xmean = X_tr.mean(0).astype(np.float32)
    xstd  = X_tr.std(0).astype(np.float32); xstd[xstd == 0] = 1.0
    def norm(A): 
        A = A.astype(np.float32, copy=False)
        return (A - xmean) / xstd

    X_tr_t = torch.from_numpy(norm(X_tr)).to(device=device, dtype=torch.float32)
    y_tr_t = torch.from_numpy(y_tr).to(device=device, dtype=torch.float32)
    X_va_t = torch.from_numpy(norm(X_va)).to(device=device, dtype=torch.float32)
    y_va_t = torch.from_numpy(y_va).to(device=device, dtype=torch.float32)

    model = TinyMLP(in_dim=X.shape[1]).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_state, best_val, wait = None, float("inf"), 0
    n = len(X_tr_t); idx = np.arange(n)

    for epoch in range(max_epochs):
        model.train()
        rng.shuffle(idx)
        for s in range(0, n, batch_size):
            mb = idx[s:s+batch_size]
            xb, yb = X_tr_t[mb], y_tr_t[mb]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=np.float32)
        X_new = np.where(np.isnan(X_new), col_means, X_new).astype(np.float32, copy=False)
        Xn = (X_new - xmean) / xstd
        Xn = Xn.astype(np.float32, copy=False)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xn).to(device=device, dtype=torch.float32)
            pred = model(xt).cpu().numpy()
        return pred

    return model, predict

# ===== LLM =====
def setup_model(model_name="Qwen/Qwen3-1.7B", device_map="cuda:1"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Some HF models (Gemma 3, etc.) compile parts of the forward pass via torch
    # dynamo. With our variable-length prompts the recompile counter trips the
    # default cache limit and aborts generation. Raise the limit and fall back
    # to eager on overflow instead of crashing.
    try:
        import torch._dynamo as _td
        _td.config.cache_size_limit = 16384
        _td.config.accumulated_cache_size_limit = 16384
        _td.config.suppress_errors = True
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs = dict(torch_dtype=dtype, device_map=device_map)
    try:
        lm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except (ValueError, ImportError):
        # Some checkpoints (e.g. Phi-4-mini older snapshots) ship custom code with
        # newer transformers APIs; fall back to executing that code.
        lm = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **load_kwargs,
        )
    lm.eval()
    return tok, lm

def _apply_chat_template(tok, msgs):
    """Apply chat template robustly across model families.

    - Qwen3 supports `enable_thinking=False`; other families error on the kwarg.
    - Gemma chat template rejects the `system` role; fold it into the first user turn.
    """
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        pass
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_msgs = [m["content"] for m in msgs if m["role"] == "system"]
        user_msgs = [m for m in msgs if m["role"] != "system"]
        if sys_msgs and user_msgs:
            prefix = "\n\n".join(sys_msgs)
            user_msgs[0] = {**user_msgs[0], "content": f"{prefix}\n\n{user_msgs[0]['content']}"}
        return tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)

def _gen_json_score(text, tok, lm, max_new_tokens=64, do_sample=False, temperature=0.0, top_p=1.0):
    inputs = tok([text], return_tensors="pt").to(lm.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        out = lm.generate(**inputs, **gen_kwargs)
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    s, e = gen.find("{"), gen.rfind("}")
    val = 3.0
    if s != -1 and e != -1 and e > s:
        try:
            val = float(json.loads(gen[s:e+1]).get("score", 3))
        except Exception:
            val = 3.0
    return float(np.clip(val, 1, 5))

def gen_score(prompt, tok, lm, max_new_tokens=64):
    msgs = [
        {"role": "system", "content": "Return JSON ONLY like {\"score\": 3}. Score must be an integer 1–5."},
        {"role": "user", "content": prompt}
    ]
    text = _apply_chat_template(tok, msgs)
    return _gen_json_score(text, tok, lm, max_new_tokens=max_new_tokens, do_sample=False)
