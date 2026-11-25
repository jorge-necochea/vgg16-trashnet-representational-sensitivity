"""Pruebas de cka_zoom_controles.py.

1. Con datos sintéticos (el mismo fixture que analisis_cpu_v3/tests/test_analisis.py): ejecución completa,
   tablas, checksums de salida, códigos de parada 6 (salida no vacía), 7 (analyze_adenda_v3.py no auditado)
   y 8 (referencia Z0 incompleta o discrepante), e IC de diferencias con remuestreo pareado.
2. Opcional, con datos reales (--emb-root, --v2-package, --repo): el CKA del zoom original (Z0) calculado con
   la misma función reproduce la tabla CKA publicada y la tabla 07 del paquete v2.

Uso:
  python tests/test_cka_zoom.py --analysis-dir RUTA/analisis_cpu_v3 \
      [--emb-root RUTA/embeddings --v2-package RUTA/run_v2 --repo RUTA/repo_6672c2c]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / 'cka_zoom_controles.py'
ap = argparse.ArgumentParser()
ap.add_argument('--analysis-dir', required=True, type=Path)
ap.add_argument('--emb-root', type=Path)
ap.add_argument('--v2-package', type=Path)
ap.add_argument('--repo', type=Path)
cli = ap.parse_args()
results = {}


def check(name, cond):
    results[name] = bool(cond)


def linear_cka(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, ord='fro') ** 2
    return cross / (np.linalg.norm(x.T @ x, ord='fro') * np.linalg.norm(y.T @ y, ord='fro'))


# Fixture sintético: se toma de las pruebas auditadas de la v3 (hasta la definición de run()).
src = (cli.analysis_dir / 'tests' / 'test_analisis.py').read_text(encoding='utf-8')
ns = {'__file__': str(cli.analysis_dir / 'tests' / 'test_analisis.py'), '__name__': 'fixture'}
code = src[:src.index('def run(')].replace('cli = ap.parse_args()', 'cli = ap.parse_args([])')
exec(compile(code, 'fixture', 'exec'), ns)
A = ns['A']

tmp = Path(tempfile.mkdtemp(prefix='test_cka_zoom_'))
try:
    ext, v2, repo = ns['make_fixture'](tmp)
    pub = repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv'

    def gaps(condition):
        with np.load(ext / 'conjuntos' / f'{condition}.npz', allow_pickle=False) as z:
            return [z[f'gap_{block}'].astype(np.float64) for block in A.BLOCKS]

    original = gaps('original_uint8')
    zooms = {lv: gaps(f'zoom_{lv}') for lv in A.LEVELS}
    ref = pd.DataFrame([
        {'transformation': 'zoom', 'level': lv, 'block': block, 'linear_CKA': linear_cka(x, y)}
        for lv, ys in zooms.items()
        for block, x, y in zip(A.BLOCKS, original, ys)
    ])
    reference = pd.concat([pd.read_csv(pub), ref], ignore_index=True)
    reference.to_csv(pub, index=False)

    def run(out, extra=()):
        return subprocess.run([sys.executable, str(SCRIPT), '--analysis-dir', str(cli.analysis_dir), '--extraction',
                               str(ext), '--v2-package', str(v2), '--repo', str(repo), '--out', str(out),
                               '--test-fixture', *extra], capture_output=True, text=True)

    out = tmp / 'out'
    r = run(out)
    check('Ejecución completa con datos sintéticos', r.returncode == 0)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
    else:
        t20 = pd.read_csv(out / '20_cka_zoom_por_bloque.csv')
        t21 = pd.read_csv(out / '21_cka_zoom_descenso_B1_B5.csv')
        check('Tabla 20: 4 variantes × 3 factores × 5 bloques, con IC finitos y ordenados', len(t20) == 60
              and set(t20.variant) == {'Z0', 'Z1', 'Z1r', 'Z2'} and np.isfinite(t20[['cka', 'ci_low', 'ci_high']]).all().all()
              and (t20.ci_low <= t20.ci_high).all())
        check('Tabla 21: D con IC para las 12 condiciones',
              len(t21) == 12 and np.isfinite(t21[['D_B1_minus_B5', 'D_ci_low', 'D_ci_high']]).all().all())
        p = t20.set_index(['variant', 'level', 'block']).cka
        check('D puntual = CKA(B1) − CKA(B5)', all(np.isclose(r_.D_B1_minus_B5, p[(r_.variant, r_.level, 'Block1')]
                                                             - p[(r_.variant, r_.level, 'Block5')], rtol=0, atol=2e-6)
                                                  for r_ in t21.itertuples()))
        t22 = pd.read_csv(out / '22_cka_zoom_diferencia_Z0.csv')
        check('Tabla 22: 3 controles × 3 factores × 5 bloques; diferencia puntual = CKA(control) − CKA(Z0), IC ordenado',
              len(t22) == 45 and (t22.ci_low <= t22.ci_high).all()
              and all(np.isclose(r_.cka_minus_Z0, p[(r_.variant, r_.level, r_.block)] - p[('Z0', r_.level, r_.block)],
                                 rtol=0, atol=2e-6) for r_ in t22.itertuples()))
        x, y0, y1 = original[0], zooms['low'][0], gaps('zoomZ1_low')[0]
        sampled_idx = np.load(v2 / 'bootstrap_indices.npy', allow_pickle=False)
        paired = [linear_cka(x[i], y1[i]) - linear_cka(x[i], y0[i]) for i in sampled_idx]
        row = t22.loc[(t22.variant == 'Z1') & (t22.level == 'low') & (t22.block == 'Block1')].iloc[0]
        check('IC pareado de Z1 − Z0 coincide con CKA de las filas remuestreadas explícitamente',
              np.allclose([row.ci_low, row.ci_high], np.percentile(paired, [2.5, 97.5]), rtol=0, atol=2e-6))
        listed = dict(ln.split('  ')[::-1] for ln in (out / 'checksums_outputs.sha256').read_text().splitlines())
        check('Checksums de salida completos y correctos (registro cerrado antes)',
              set(listed) == {f.name for f in out.iterdir()} - {'checksums_outputs.sha256'}
              and all(hashlib.sha256((out / k).read_bytes()).hexdigest() == v for k, v in listed.items())
              and (out / 'run_log.txt').read_text().strip().endswith('checksums de salida.'))
        man = json.loads((out / 'manifest.json').read_text())
        check('Manifiesto: descriptivo, no preespecificado, marcado como prueba',
              'no preespecificado' in man['analysis'] and man['test_fixture'] and not man['valid_for_manuscript'])
    check('Carpeta de salida no vacía: código 6', run(out).returncode == 6)
    zoom_idx = reference.index[reference.transformation.eq('zoom')][0]
    drift = reference.copy()
    drift.loc[zoom_idx, 'linear_CKA'] += 0.01
    for suffix, altered, reason in [('mismatch', drift, 'discrepante'),
                                    ('missing', reference.drop(index=zoom_idx), 'incompleta')]:
        test_out = tmp / suffix
        altered.to_csv(pub, index=False)
        check(f'Referencia Z0 {reason}: código 8 y sin tablas',
              run(test_out).returncode == 8 and not list(test_out.glob('*.csv')))
    reference.to_csv(pub, index=False)
    fake = tmp / 'fake_analysis'
    shutil.copytree(cli.analysis_dir, fake)
    with open(fake / 'analyze_adenda_v3.py', 'a') as f:
        f.write('\n# modificado\n')
    r7 = subprocess.run([sys.executable, str(SCRIPT), '--analysis-dir', str(fake), '--extraction', str(ext),
                         '--v2-package', str(v2), '--repo', str(repo), '--out', str(tmp / 'out7')],
                        capture_output=True, text=True)
    check('analyze_adenda_v3.py distinto del auditado (sin --test-fixture): código 7', r7.returncode == 7)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if cli.emb_root and cli.v2_package and cli.repo:
    idx = pd.read_csv(cli.repo / 'data/manifests/original_image_index.csv')
    with np.load(cli.emb_root / 'embeddings_originales_VGG16_R1.npz') as z:
        orig = [z[b].astype(np.float64) for b in A.BLOCKS]
    trans = {}
    for lv in A.LEVELS:
        with np.load(cli.emb_root / 'transformadas' / f'embeddings_zoom_{lv}.npz') as z:
            trans[f'zoom_{lv}'] = [z[b].astype(np.float64) for b in A.BLOCKS]
    IDX, W, *_ = A.load_frozen(cli.v2_package, idx, lambda m: None, test_mode=False)
    cka, _ = A.cka_bootstrap(orig, trans, W)
    cka['level'] = cka.condition.str.split('_').str[1]
    pubt = pd.read_csv(cli.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv')
    m = cka.merge(pubt[pubt.transformation == 'zoom'][['level', 'block', 'linear_CKA']], on=['level', 'block'])
    check(f'Datos reales: CKA de Z0 = tabla publicada (15 celdas, error {np.max(np.abs(m.cka - m.linear_CKA)):.1e})',
          len(m) == 15 and np.max(np.abs(m.cka - m.linear_CKA)) < 1e-6)
    v2t = pd.read_csv(cli.v2_package / '07_cka_por_bloque.csv')
    m2 = cka.merge(v2t[v2t.transformation == 'zoom'], on=['level', 'block'], suffixes=('', '_v2'))
    check('Datos reales: IC del CKA de Z0 = tabla 07 del paquete v2 (rtol 1e-5)',
          len(m2) == 15 and np.allclose(m2.ci_low, m2.ci_low_v2, rtol=1e-5, atol=0)
          and np.allclose(m2.ci_high, m2.ci_high_v2, rtol=1e-5, atol=0))

for k, v in results.items():
    print(('OK    ' if v else 'FALLO ') + k)
n_ok = sum(results.values())
print(f'\n{n_ok}/{len(results)} pruebas superadas')
sys.exit(0 if n_ok == len(results) else 1)
