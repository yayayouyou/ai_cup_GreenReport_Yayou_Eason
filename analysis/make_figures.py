"""Figure 1: predicted vs observed per-field deltas across the 44 binary sources.

Reads paper/out/binary_swap.json (written by binary_swap_experiment.py) and emits a
publication-sized PDF into paper/tex/fig/.

Usage:
  .venv_mac/bin/python paper/make_figures.py
"""
import argparse, json, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swap', default='out/binary_swap.json')
    ap.add_argument('--outdir', default='out/fig')
    ap.add_argument('--sigma', type=float, default=0.0054,
                    help='pooled within-family seed SD (bootstrap_apparatus.py)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = json.load(open(args.swap, encoding='utf-8'))['results']
    ref = next(r for r in rows if r['source'] == 'PIPELINE(self)' and r['mode'] == 'class')

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=False)
    panels = [('verification_timeline', 'timeline', axes[0]),
              ('evidence_quality', 'quality', axes[1])]
    colors = {'class': '#1b6ca8', 'const': '#c85200'}
    labels = {'class': r'v2: class-conditional $\gamma_c$', 'const': r'v1: constant $\gamma$'}

    for field, name, ax in panels:
        lo = hi = None
        for mode in ['const', 'class']:
            xs, ys = [], []
            for r in rows:
                if r['mode'] != mode or r['source'] == 'PIPELINE(self)':
                    continue
                xs.append(r['obs'][field] - ref['obs'][field])
                ys.append(r['pred'][field] - ref['obs'][field])
            ax.scatter(xs, ys, s=16, alpha=.75, edgecolor='none',
                       c=colors[mode], label=labels[mode])
            vals = xs + ys
            lo = min(vals) if lo is None else min(lo, min(vals))
            hi = max(vals) if hi is None else max(hi, max(vals))
        pad = (hi - lo) * 0.12
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color='0.35', lw=.9, zorder=0)
        ax.fill_between([lo, hi], [lo - args.sigma, hi - args.sigma],
                        [lo + args.sigma, hi + args.sigma],
                        color='0.85', alpha=.55, zorder=-1, lw=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'observed $\\Delta$ {name}')
        ax.set_ylabel(f'predicted $\\Delta$ {name}')
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper left')
    fig.tight_layout()
    out = os.path.join(args.outdir, 'predicted_vs_observed.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
