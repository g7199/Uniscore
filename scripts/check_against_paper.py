#!/usr/bin/env python3
"""Compare the regenerated tables (tables/*.tex) with the values printed in the paper (expected/*.tex), line by line."""
from __future__ import annotations

import os
import sys

from common import ROOT

ok = True
for name in ("factorial", "lead", "ablation", "panel"):
    got_f, exp_f = os.path.join(ROOT, "tables", f"{name}.tex"), os.path.join(ROOT, "expected", f"{name}.tex")
    if not os.path.exists(got_f):
        print(f"{name}: not generated")
        ok = False
        continue
    got, exp = open(got_f).read().splitlines(), open(exp_f).read().splitlines()
    diff = [(i + 1, e, g) for i, (e, g) in enumerate(zip(exp, got)) if e != g]
    if len(got) != len(exp):
        diff.append(("lines", len(exp), len(got)))
    print(f"{name}: {'identical to the paper' if not diff else f'{len(diff)} differing lines'}")
    for d in diff:
        print("   paper:", d[1], "\n   here: ", d[2])
    ok &= not diff
sys.exit(0 if ok else 1)
