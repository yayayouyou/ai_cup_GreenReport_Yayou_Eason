"""The toy example of Section 3.1 ("Three quantities, one toy"), computed by hand-checkable code.

Six items, two fields: a gate G (weight 0.2; Yes, No) and a downstream field D (weight 0.8;
classes a, b scored, N/A unscored), with the rule G = No => D = N/A. Gold: three (Yes, a),
one (Yes, b), two (No, N/A).

It separates the three things a "weight" can mean:
  nominal weight   w_G = 0.2, as published;
  per-class share  w_D / |C_D| = 0.4 for class b (one item) and class a (three items) alike;
  effective weight the gain in S per point of M_G from repairing one gate false negative --
                   0.667 against the nominal 0.2, 70% of it earned in D.

Run bare:  python toy_example.py
"""


def f1(tp, fn, fp):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fn + fp)


def macro(gold, pred, classes):
    scores = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        scores.append(f1(tp, fn, fp))
    return sum(scores) / len(scores)


W_G, W_D = 0.2, 0.8
GOLD = [("Yes", "a"), ("Yes", "a"), ("Yes", "a"), ("Yes", "b"), ("No", "N/A"), ("No", "N/A")]
# the only error: one a-item gated shut, so its downstream value becomes N/A
BEFORE = [("Yes", "a"), ("Yes", "a"), ("No", "N/A"), ("Yes", "b"), ("No", "N/A"), ("No", "N/A")]


def fields(pred):
    m_g = macro([g for g, _ in GOLD], [p for p, _ in pred], ["Yes", "No"])
    m_d = macro([d for _, d in GOLD], [p for _, p in pred], ["a", "b"])
    return m_g, m_d


def main():
    m_g0, m_d0 = fields(BEFORE)
    m_g1, m_d1 = fields(GOLD)
    d_s = W_G * (m_g1 - m_g0) + W_D * (m_d1 - m_d0)
    print(f"per-class share of D: {W_D / 2:.1f} for a (support 3) and for b (support 1)")
    print(f"before repair: M_G = {m_g0:.3f}  M_D = {m_d0:.3f}")
    print(f"repair: S rises by {d_s:.3f} = {d_s / (m_g1 - m_g0):.3f} per point of M_G "
          f"(nominal {W_G}); share earned in D {100 * W_D * (m_d1 - m_d0) / d_s:.0f}%")


if __name__ == "__main__":
    main()
