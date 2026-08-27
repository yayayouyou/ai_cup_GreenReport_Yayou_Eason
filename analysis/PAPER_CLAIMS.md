# Why numbers kept coming out wrong, and what stops it

Across five review passes of the paper, roughly forty defects were found and fixed. **Almost none
of them was a wrong number.** Every scalar in `out/*.json` was correct the whole time. What kept
failing was the sentence wrapped around the scalar.

That distinction is the whole point of this file. `README.md` maps each paper number to a script
and an output field, which catches a mistyped value. It cannot catch "the ordering survives" or
"Lemma 2 makes the gate worth 0.5216", because those are not values — they are *relations between*
values, or *attributions* of a value to a cause. Nothing was checking those.

`check_claims.py` now does. Run it before quoting any of these numbers in prose:

```bash
python check_claims.py
```

---

## The seven failure modes

Each is a real defect that survived at least one review pass. The example is the actual text that
shipped in a draft.

### A. A number is quoted with a label describing a different computation

The value is right; the words next to it describe an operation the script does not perform.

| Shipped text | What the code actually does |
|---|---|
| Table caption: "Prices are **derivatives** at the deployed operating point" | `repair()` flips **all** *n* rows to gold and divides by *n* — a finite mean repair, not a derivative |
| "138 **promise false-negative** repairs" | the promise rows splice gold **evidence** onto the repaired row too, so the price is for a promise+evidence repair (8.46e-4 with the splice, 6.30e-4 without) |
| "**36.77%** of 10⁴ replicates duly contain no *Misleading* item" | `bootstrap_apparatus.py` never resampled; the field is literally named `p_absent_analytic`. The measured value is 3,641/10,000 = 36.4% |
| "42 of 2,110 items break one" mapped to `metric_design.json` | that file reported **45**; the 42 uses only the rules that transfer, and was not an output field until it was added |

**Rule.** Before writing the noun phrase next to a number, open the function that produced it and
read what it varies and what it holds fixed. If the label cannot be justified from those lines,
the label is wrong, not the number.

### B. A relation is asserted to hold when the outputs say it does not

The hardest class, because both endpoints are individually correct.

| Shipped text | The arithmetic |
|---|---|
| "The direction and **the ordering of channels survive**" | derivative: evidence +0.2575 > promise +0.2330. Finite move: promise +0.0370 > evidence +0.0295. **The top two exchange rank.** |
| "though its predicted quality **and timeline** macros do not [carry within 0.05]" | timeline is off by **0.0319**, which is *inside* 0.05 |
| "**neither variant** beats the mean" | mean 0.0020, class-independent variant 0.0014, lower being better — it beat it |
| "−0.0265 … the largest single modelling gain in this paper" | §4.3 prints +0.0286 for one preprocessing step |

**Rule.** Any sentence containing *survives, holds, tracks, larger than, neither, all, none* is a
claim that must be recomputed, not looked up. `check_claims.py` §1, §4 do the two that recur.

### C. A number is attributed to the wrong cause

| Shipped text | Why it is wrong |
|---|---|
| "**Lemma 2** makes the gate worth 0.5216" | Lemma 2 is a gold-label biconditional; it prices nothing. Eq. (4) on the counted model yields 0.5216. This sentence is what invited a reviewer to relabel Lemma 2 "asymmetry of error propagation" |
| "a gate closure costs recall alone **wherever N/A is unscored**" | the largest channel is evidence **+0.2575**, and evidence is exactly where N/A **is** scored |
| "a single system, scoring **0.6095** on validation" (§1) | 0.6095 is the reconstruction; the shipped build is 0.6087 |

**Rule.** Ask what object the number is a property of. A lemma constrains; an equation evaluates;
a build scores. They are not interchangeable.

### D. Two numbers printed as a pair, unlabelled, in an order that contradicts an earlier pair

| Shipped text | What goes wrong |
|---|---|
| val "γ_Q = 0.832 and 0.959" (Not Clear first), then test "γ_Q is 0.943 against 0.788" (Clear first) | read in the order the paper established, the "within 0.05" check gives **0.111 and 0.171** and fails; read correctly it gives 0.016 and 0.044 and passes |
| "correlating each against φ and against ρ … the composite rewards recall (+0.37 against −0.27)" | the second pair is printed ρ-first, violating the order the same sentence declared; read as written, the composite rewards false-opening |

**Rule.** Never print a bare pair. Attach the class or variable to at least one member. Two
unlabelled numbers cost one word to disambiguate and can invert a conclusion.

### E. The same 4-decimal string denotes two different quantities

| String | Meaning 1 | Meaning 2 |
|---|---|---|
| `0.0104` | val 0.6087 − test 0.5983 | promise's share of the cascade term |
| `0.0319` | cascade headroom | timeline transfer residual |

**Rule.** A repeated 4-decimal string reads as a cross-reference whether or not one is intended.
If two unrelated quantities collide, express one differently ("a third of the cascade term").

### F. The last printed digit does not match the source

| Shipped | Correct | How it was caught |
|---|---|---|
| promise Yes-F1 `0.942` | `0.943` | (0.943+0.686)/2 = 0.814801953839387 = Table 1's M_f exactly; 0.942 does not |
| recall_No `0.562` | `0.561` | 105/187 = 0.5615 |
| "**three** times" | 3.65 | 0.1167 / 0.0319 |
| `0.0065` | `0.0064` | F₁ = 2/37, not 2/36 |
| `172 (72.3%)` | `171 (71.8%)` | recount |

**Rule.** For any per-class figure, re-average the rounded values and check they still reproduce
the field macro (`check_claims.py` §3). This catches a wrong digit without recounting anything.

### G. A claim's scope is wider than the evidence

| Shipped text | The counterexample |
|---|---|
| "**every method here** called `Clear`" | the table's own last row says it flagged the item |
| "more than our ensemble and decision-layer work returned **together**" | the decision layer is never priced anywhere; only the ensemble search is (+0.0042) |
| "lands entirely in one partition with probability 2^(1−s)" | requires equal partitions, which §3.2 says were never published; `≥ 2^(1−s)` holds for any split |
| "a class of support s … carries 0.1167" | 0.1167 is quality's coefficient; timeline classes carry 0.0375 |

**Rule.** For every universal (*every, all, none, together*), name the counterexample that would
falsify it and check it exists in the paper's own tables.

---

## The two root causes

**1. Prose was written from console output, once, and the analysis moved afterwards.**
`0.8144/0.7016` appears in no output file in this repository — the reconstructed ensemble gives
`0.8148/0.7083`, matching Table 1. Nothing linked the sentence to the field it came from, so when
the pipeline changed the sentence did not.

**2. Only scalars were being audited.** The scalar audit passed 24/24 at every stage while classes
B, C and G went undetected, because those are statements *about* scalars. `check_claims.py` exists
to close exactly that gap.

---

## Before quoting a number in prose

1. `python check_claims.py` — read the section covering it.
2. Name the output field it comes from (`README.md`), and read the function that writes it.
3. If the sentence asserts a **relation**, recompute the relation, not the endpoints.
4. If it prints a **pair**, label at least one member.
5. If it contains a **universal**, find the row that would falsify it.
6. If it attributes the number to a **cause**, check that object actually produces it.
