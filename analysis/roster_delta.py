# -*- coding: utf-8 -*-
"""What did the strong4 subset selection actually buy, in gate coordinates? Backs
Section 4.3's "buying rho (+0.0098) and conceding phi (+0.0107)".

The paper (Section 4.3) says the eight-candidate subset search kept strong4 and that the diluted
seven-model vote is what it replaced. Compute (rho, phi) for both votes on validation and report
the deltas -- the reviewer's remaining W3 question: did the intervention mainly buy recall, or
both directions?
"""
import glob
import io
import json
import os
import sys

from binary_swap_experiment import vote, gate_params

VAL = "../data_set/vpesg4k_val_1000.json"
SRC = "io/binary_sources"

rows = json.load(io.open(VAL, encoding="utf-8"))

sources = {}
for d in sorted(os.listdir(SRC)):
    name = d.replace("valeval_", "")
    hits = glob.glob(os.path.join(SRC, d, "*", "val_predictions.json"))
    if not hits:
        continue
    preds = json.load(io.open(sorted(hits)[-1], encoding="utf-8"))
    preds = preds if isinstance(preds, list) else preds.get("predictions", [])
    sources[name] = {str(p["id"]): p for p in preds}

S4 = ["ckip_tapt_ep3", "macbert_tapt", "bgem3", "bgem3_tapt"]
D7 = S4 + ["b0_ckip", "macbert_base", "roberta_wwm"]
missing = [k for k in D7 if k not in sources]
if missing:
    print("missing sources:", missing)
    print("available:", sorted(sources)[:20])
    sys.exit(1)

for label, members in (("strong4(vote)", S4), ("diluted7(vote)", D7)):
    v = vote(sources, members)
    g = gate_params(rows, v)
    print(f"{label:16s} rho={g['rho']:.4f}  phi={g['phi']:.4f}")

gs = gate_params(rows, vote(sources, S4))
gd = gate_params(rows, vote(sources, D7))
print(f"\ndelta (strong4 - diluted7):  drho={gs['rho']-gd['rho']:+.4f}   "
      f"dphi={gs['phi']-gd['phi']:+.4f}")
print("(positive drho = recall bought; negative dphi = specificity bought)")
