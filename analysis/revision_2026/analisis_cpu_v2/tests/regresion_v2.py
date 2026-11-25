"""Regresión con datos reales: el núcleo del análisis nuevo reproduce el paquete v2 aprobado.

Aplica las funciones de analyze_adenda_v2.py a las 12 condiciones ya verificadas, tratándolas como un
control nuevo con las reglas de la afirmación correspondiente:
  - brillo → reglas de gamma (C1: referencia B2, región superficial);
  - rotación R0 → reglas de R1 (C3: referencia B3, región intermedia, dirección B1 y B5);
  - zoom Z0 → reglas de Z1 (C5: referencia B5, región profunda, dirección B1).
Compara con las tablas 04, 06, 07, 08, 09 y 11 del paquete v2: regiones pico y estabilidad de F1–F4,
semillas de F4, contrastes (r_rb, mediana, IC), CKA y C4, conservación en las 200 extracciones y veredictos.

Uso:
  python tests/regresion_v2.py --emb-root RUTA/embeddings --v2-package RUTA/run_v2 --repo RUTA/repo
(--emb-root contiene embeddings_originales_VGG16_R1.npz y transformadas/embeddings_{t}_{lv}.npz)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_adenda_v2 as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--emb-root', required=True, type=Path)
ap.add_argument('--v2-package', required=True, type=Path)
ap.add_argument('--repo', required=True, type=Path)
args = ap.parse_args()
log = print

index = pd.read_csv(args.repo / 'data/manifests/original_image_index.csv')
labels = index['class'].to_numpy()
with np.load(args.emb_root / 'embeddings_originales_VGG16_R1.npz') as z:
    orig = [z[b].astype(np.float64) for b in A.BLOCKS]
trans = {}
for t in ['brightness', 'rotation', 'zoom']:
    for lv in A.LEVELS:
        with np.load(args.emb_root / 'transformadas' / f'embeddings_{t}_{lv}.npz') as z:
            trans[f'{t}_{lv}'] = [z[b].astype(np.float64) for b in A.BLOCKS]
IDX, W, draws, k, _ = A.load_frozen(args.v2_package, index, log, test_mode=False)
pi, pj = A.reference_pairs(labels)
pub = np.load(args.repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz')
B_pub = np.array([float(np.median(pub[f'{b}_random'].astype(np.float64))) for b in A.BLOCKS])
num = index.original_filename.str.extract(r'(\d+)')[0].astype(int).to_numpy()
first_k = np.flatnonzero(num <= k)
G = [A.normalize(o) @ A.normalize(o).T for o in orig]

v2 = args.v2_package
reg2 = pd.read_csv(v2 / '04_regiones_estabilidad.csv')
ct2 = pd.read_csv(v2 / '06_contrastes_referencia_F1.csv')
comp2 = pd.read_csv(v2 / '09_composicion_clases.csv')
ver2 = pd.read_csv(v2 / '11_veredictos_por_condicion.csv')
results, worst = {}, {}
for t, ctl in [('brightness', 'gamma'), ('rotation', 'R1'), ('zoom', 'Z1')]:
    conds = [f'{t}_{lv}' for lv in A.LEVELS]
    res = A.region_control(ctl, conds, orig, {c: trans[c] for c in conds}, B_pub, pi, pj, IDX, W, G, draws,
                           first_k, log)
    for f in ['F1', 'F2', 'F3', 'F4']:
        new = res['regions'][res['regions'].family == f].set_index('level')
        old = reg2[(reg2.family == f) & (reg2.transformation == t)].set_index('level')
        for lv in A.LEVELS:
            results[f'{t} {lv} {f}: región pico'] = new.loc[lv, 'peak_region_mean'] == old.loc[lv, 'peak_region_mean']
            d = abs(new.loc[lv, 'stability_expected_region_mean'] - old.loc[lv, 'stability_published_region_mean'])
            worst['stability'] = max(worst.get('stability', 0), d)
            results[f'{t} {lv} {f}: estabilidad idéntica'] = d < 1e-12
            if f == 'F4':
                results[f'{t} {lv} F4: semillas en la región'] = (new.loc[lv, 'F4_seeds_in_expected_region']
                                                                  == old.loc[lv, 'F4_seeds_in_published_region'])
            if f == 'F3':
                results[f'{t} {lv} F3: no informativa igual'] = (bool(new.loc[lv, 'F3_noninformative'])
                                                                 == bool(old.loc[lv, 'F3_noninformative']))
    new = res['contrasts'].assign(level=lambda d: d.level)
    for r in new.itertuples():
        o = ct2[(ct2.transformation == t) & (ct2.level == r.level) & (ct2.other_block == r.other_block)].iloc[0]
        for col_new, col_old in [('r_rb', 'r_rb'), ('F1_median_diff', 'F1_median_diff'),
                                 ('median_diff_ci_low', 'median_diff_ci_low'),
                                 ('median_diff_ci_high', 'median_diff_ci_high'), ('wilcoxon_p', 'wilcoxon_p')]:
            d = abs(getattr(r, col_new) - o[col_old]) / max(1.0, abs(o[col_old]))
            worst[col_new] = max(worst.get(col_new, 0), d)
        results[f'{t} {r.level} {r.other_block}: relevancia igual'] = bool(r.relevant) == bool(o.relevant)
    cn = res['composition'].set_index('level')
    co = comp2[comp2.transformation == t].set_index('level')
    for lv in A.LEVELS:
        results[f'{t} {lv}: conservación con B fijo y recalculado'] = (
            abs(cn.loc[lv, 'conservation_B_fixed'] - co.loc[lv, 'conservation_B_fixed']) < 1e-12
            and abs(cn.loc[lv, 'conservation_B_recomputed'] - co.loc[lv, 'conservation_B_recomputed']) < 1e-12)
    vn = res['conditions'].set_index('level')
    vo = ver2[ver2.transformation == t].set_index('level')
    for lv in A.LEVELS:
        results[f'{t} {lv}: veredicto v1.3 igual'] = vn.loc[lv, 'verdict_v13'] == vo.loc[lv, 'verdict']
# Las tablas de la v2 se guardaron con 6 cifras significativas (float_format='%.6g'): tolerancia relativa 1e-5.
for k_, v in worst.items():
    results[f'Diferencia relativa máxima en {k_} < 1e-5, redondeo de la v2 (valor {v:.1e})'] = v < 1e-5

# CKA y C4 bajo rotación
cka, boots = A.cka_bootstrap(orig, {f'rotation_{lv}': trans[f'rotation_{lv}'] for lv in A.LEVELS}, W)
cka2 = pd.read_csv(v2 / '07_cka_por_bloque.csv')
cka2 = cka2[cka2.transformation == 'rotation'].assign(condition=lambda d: 'rotation_' + d.level)
m = cka.merge(cka2, on=['condition', 'block'], suffixes=('', '_v2'))
results['CKA de rotación (15 celdas): punto e IC iguales a la v2 (rtol 1e-5)'] = (
    len(m) == 15 and np.allclose(m.cka, m.cka_v2, rtol=1e-5, atol=0)
    and np.allclose(m.ci_low, m.ci_low_v2, rtol=1e-5, atol=0) and np.allclose(m.ci_high, m.ci_high_v2, rtol=1e-5, atol=0))
c42 = pd.read_csv(v2 / '08_C4_cka_B1_menos_B5.csv').set_index('level')
for lv in A.LEVELS:
    d = boots[(f'rotation_{lv}', 0)] - boots[(f'rotation_{lv}', 4)]
    results[f'C4 {lv}: IC de CKA(B1) − CKA(B5) igual (rtol 1e-5)'] = np.allclose(
        np.percentile(d, [2.5, 97.5]), c42.loc[lv, ['ci_low', 'ci_high']].to_numpy(float), rtol=1e-5, atol=0)
for k_, v in results.items():
    print(('OK    ' if v else 'FALLO ') + k_)
n_ok = sum(bool(v) for v in results.values())
print(f'\n{n_ok}/{len(results)} comprobaciones de regresión superadas')
sys.exit(0 if n_ok == len(results) else 1)
