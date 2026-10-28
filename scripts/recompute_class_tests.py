"""Recompute published class-specific Kruskal–Wallis and Mann–Whitney–Holm results.

Run from any directory: python scripts/recompute_class_tests.py
Requires: numpy, pandas, scipy. Uses no random sampling.
Input: data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz
Outputs: data/summary_tables/main/10_kruskal_wallis_clases.csv
         data/summary_tables/main/11_mann_whitney_clases_holm.csv
The original historical notebook mislabels (H-k+1)/(n-k) as epsilon_squared;
this regenerated release correctly names the unchanged estimator eta_squared_h.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('cosine_similarity', 'relative_euclidean_distance')
CLASSES = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
KEYS = ['transformation', 'level', 'block']


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def main():
    df = pd.read_csv(ROOT / 'data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz')
    if len(df) != 151620 or df[KEYS].drop_duplicates().shape[0] != 60:
        raise ValueError('Main metric data do not match expected 151620 rows / 60 conditions')
    if df.duplicated(['image_id'] + KEYS).any() or df[['class'] + KEYS + list(METRICS)].isna().any().any():
        raise ValueError('Missing values or duplicated image-condition keys')
    omnibus, pairwise = [], []
    for metric in METRICS:
        for (transformation, level, block), g in df.groupby(KEYS, observed=True):
            arrays = {c: g.loc[g['class'] == c, metric].to_numpy() for c in CLASSES}
            n, k = len(g), len(CLASSES)
            if n != 2527 or any(len(a) == 0 for a in arrays.values()):
                raise ValueError(f'Unexpected class composition: {metric}, {transformation}, {level}, {block}')
            h, p = kruskal(*[arrays[c] for c in CLASSES])
            omnibus.append({'metric': metric, 'transformation': transformation, 'level': level,
                            'block': block, 'n': n, 'k': k, 'kruskal_h': h, 'df': k - 1,
                            'p_value': p, 'eta_squared_h': max(0., (h - k + 1)/(n - k))})
            family = []
            for a, b in itertools.combinations(CLASSES, 2):
                x, y = arrays[a], arrays[b]
                res = mannwhitneyu(x, y, alternative='two-sided', method='auto')
                family.append({'metric': metric, 'transformation': transformation, 'level': level,
                               'block': block, 'class_a': a, 'class_b': b,
                               'n_a': len(x), 'n_b': len(y), 'mann_whitney_u': res.statistic,
                               'p_value': res.pvalue, 'median_a': np.median(x), 'median_b': np.median(y),
                               'rank_biserial_a_vs_b': 2*res.statistic/(len(x)*len(y)) - 1})
            for row, corrected in zip(family, holm_adjust([r['p_value'] for r in family])):
                row['p_holm_within_family'] = corrected
            pairwise.extend(family)
    omni = pd.DataFrame(omnibus)
    for metric, idx in omni.groupby('metric').groups.items():
        omni.loc[idx, 'p_holm_across_omnibus'] = holm_adjust(omni.loc[idx, 'p_value'])
    omni['significant_holm'] = omni['p_holm_across_omnibus'] < 0.05
    pair = pd.DataFrame(pairwise)
    if (len(omni), len(pair)) != (120, 1800):
        raise AssertionError(f'Unexpected output shapes: {omni.shape}, {pair.shape}')
    output = ROOT / 'data/summary_tables/main'
    output.mkdir(parents=True, exist_ok=True)
    omni.to_csv(output / '10_kruskal_wallis_clases.csv', index=False, float_format='%.17g')
    pair.to_csv(output / '11_mann_whitney_clases_holm.csv', index=False, float_format='%.17g')
    print('PASS: 120 class omnibus rows + 1800 post-hoc rows, 2 metrics, 60 conditions')

if __name__ == '__main__':
    main()
