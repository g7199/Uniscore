# Judge prompts

Every judge call scores one text on one criterion. `criteria.json` lists, per dataset, the criterion questions with their
1-5 scale descriptions and the holistic question (`holistic`, used for the holistic baseline).

System message:

```
You are a careful evaluator. Return JSON ONLY like {"score": 3}. The score must be an integer from 1 to 5.
```

User message (the source or prompt is shown only for criteria with `shows_context: true`; texts are clipped to 12,000
characters, sources to 9,000):

```
<text_label>:
<text>

<context_label>:
<source or prompt>

---
Criterion: <question>
Return JSON only.
{"score": N}
```

Score types (see `data/README.md`): open models run with llama.cpp, the answer is prefilled with `{"score": ` and the score
is the expected value of the next digit token (1-5); GPT-6 Luna returns the five most probable tokens, renormalised over the
digits; Claude Haiku 4.5, Claude Sonnet 5.5 and GPT-6.1 Sol give the integer answer.
