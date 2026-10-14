"""The toy example of Section 3.1 ("Three quantities, one toy"), computed by hand-checkable code.

Six items, two fields: a gate G (weight 0.2; Yes, No) and a downstream field D (weight 0.8;
classes a, b scored, N/A unscored), with the rule G = No => D = N/A. Gold: three (Yes, a),
one (Yes, b), two (No, N/A). The system's only error sends one a-item to (No, N/A).

It separates the three things a "weight" can mean:
  nominal weight   w_G = 0.2, as published;
  per-class share  w_D / |C_D| = 0.4 for class b (one item) and class a (three items) alike;
  effective weight the gain in S per point of M_G. Repairing the one error lifts S by 0.114 and
                   M_G by 0.171, an AVERAGE of 0.667 per point; the LOCAL rate at the operating
                   point is 0.794, 74.8% of it earned in D.

The local rate uses the same continuous relaxation as the cascade model: x, the number of
correctly gated a-items, is treated as real-valued, and the rate is dS/dx over dM_G/dx at x = 2
(central difference). Average and local rate differ because F1 is curved in its counts.

Run bare:  python toy_example.py
"""

W_G, W_D = 0.2, 0.8


def f1(tp, fn, fp):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fn + fp)


def fields(x):
    """Field macros when x of the three a-items are gated open (the rest sent to No, N/A)."""
    yes = f1(tp=x + 1, fn=3 - x, fp=0)          # gold Yes: three a-items and the b-item
    no = f1(tp=2, fn=0, fp=3 - x)               # gold No: two items; shut a-items are FP
    a = f1(tp=x, fn=3 - x, fp=0)                # shut a-items become N/A, i.e. missed
    b = f1(tp=1, fn=0, fp=0)
    return (yes + no) / 2, (a + b) / 2


def score(x):
    m_g, m_d = fields(x)
    return W_G * m_g + W_D * m_d


def main():
    m_g0, m_d0 = fields(2)
    m_g1, _ = fields(3)
    d_s = score(3) - score(2)
    print(f"per-class share of D: {W_D / 2:.1f} for a (support 3) and for b (support 1)")
    print(f"operating point: M_G = {m_g0:.3f}  M_D = {m_d0:.3f}")
    print(f"one-item repair: S +{d_s:.3f}, M_G +{m_g1 - m_g0:.3f}, "
          f"average {d_s / (m_g1 - m_g0):.3f} per point (nominal {W_G})")
    h = 1e-4
    ds = (score(2 + h) - score(2 - h)) / (2 * h)
    dg = (fields(2 + h)[0] - fields(2 - h)[0]) / (2 * h)
    dd = W_D * (fields(2 + h)[1] - fields(2 - h)[1]) / (2 * h)
    print(f"local rate at the operating point: {ds / dg:.3f} per point, "
          f"{100 * dd / ds:.1f}% of it earned in D")


if __name__ == "__main__":
    main()
