"""CKA bajo los controles de zoom Z1, Z1r y Z2 — análisis descriptivo adicional E1 (no preespecificado).

El comentario 4 del Revisor 3 pide recalcular la sensibilidad angular ajustada y el CKA para los controles de
zoom. La Adenda 1 v1.2 solo fijó el CKA para la rotación (C4); para el zoom fijó A_adj. Este script añade el CKA
de los controles como resultado DESCRIPTIVO, sin veredicto ni regla de decisión, con la misma extracción
auditada, el mismo bootstrap congelado y la misma implementación del CKA que analyze_adenda_v3.py.
La versión 1.1 exige la concordancia de las 15 celdas de Z0 antes de escribir las tablas.

Salidas (carpeta nueva):
  20_cka_zoom_por_bloque.csv     CKA lineal centrado por bloque, con IC percentil 2,5–97,5 (2000 réplicas),
                                 para el zoom original (Z0 = zoom_*) y Z1, Z1r, Z2, en los tres factores.
  21_cka_zoom_descenso_B1_B5.csv D = CKA(B1) − CKA(B5) con IC, bloque de CKA mínimo y tramos en que el CKA
                                 puntual sube entre bloques consecutivos.
  22_cka_zoom_diferencia_Z0.csv  CKA(control) − CKA(Z0) por bloque, con IC calculado con las mismas réplicas.
                                 (No se usa una razón de descensos como en C4: bajo el zoom original el CKA no
                                 desciende de forma sostenida y D_Z0 es cercano a cero o negativo.)
  manifest.json, run_log.txt, checksums_outputs.sha256 (calculados después de cerrar el registro).

Validación previa idéntica a analyze_adenda_v3.py (checksums 47/47 y H_checksums fijado, huella, archivos del
repositorio, metadatos de los 35 NPZ, bootstrap congelado). Además, el CKA de Z0 se compara con la tabla CKA
publicada. La ejecución se detiene con código 8 si falta alguna celda, hay duplicados o el error
máximo absoluto supera 1e-6.

Uso:
  python cka_zoom_controles.py --analysis-dir RUTA/analisis_cpu_v3 --extraction RUTA/Revision_Adenda1_extraccion_v2 \
      --v2-package RUTA/run_v2 --repo RUTA/repo_6672c2c --out RUTA/analisis_cka_zoom_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_SCRIPT_SHA256 = '6e797590c63f57e17d27619039c6ea6008fcfe2630a36fc4c7821b5d397b2a7e'  # analyze_adenda_v3.py
VARIANTS = ['Z0', 'Z1', 'Z1r', 'Z2']
LEVELS = ['low', 'medium', 'high']
FACTOR = {'low': 1.10, 'medium': 1.25, 'high': 1.50}
Z0_CKA_ATOL = 1e-6


def set_name(variant: str, lv: str) -> str:
    return f'zoom_{lv}' if variant == 'Z0' else f'zoom{variant}_{lv}'


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-dir', required=True, type=Path, help='Carpeta con analyze_adenda_v3.py (v3 auditada)')
    ap.add_argument('--extraction', required=True, type=Path)
    ap.add_argument('--v2-package', required=True, type=Path)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--test-fixture', action='store_true', help='Solo para las pruebas con datos sintéticos')
    args = ap.parse_args()

    mod = args.analysis_dir / 'analyze_adenda_v3.py'
    h_mod = sha256(mod)
    if h_mod != ANALYSIS_SCRIPT_SHA256 and not args.test_fixture:
        print(f'ERROR: analyze_adenda_v3.py ({h_mod}) no es el auditado.', flush=True)
        sys.exit(7)
    sys.path.insert(0, str(args.analysis_dir))
    import analyze_adenda_v3 as A  # noqa: E402

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        print('ERROR: la carpeta de salida no está vacía.', flush=True)
        sys.exit(6)
    log = A.Logger(out / 'run_log.txt')
    script = Path(__file__).resolve()
    log(f'Script {script.name}, SHA-256 {sha256(script)}; usa analyze_adenda_v3.py, SHA-256 {h_mod}')
    log('Análisis descriptivo adicional E1 (no preespecificado en la Adenda 1 v1.2): CKA bajo Z0, Z1, Z1r y Z2.')
    if args.test_fixture:
        log('*** MODO DE PRUEBA CON DATOS SINTÉTICOS: las salidas NO son válidas para el manuscrito. ***')

    checks = A.validate_checksums(args.extraction, log, args.test_fixture)
    fpinfo = A.validate_fingerprint(args.extraction, log, args.test_fixture)
    repo_hashes = A.validate_repo_files(args.repo, log, args.test_fixture)
    index = pd.read_csv(args.repo / 'data/manifests/original_image_index.csv')
    data, _, _ = A.load_sets(args.extraction, fpinfo['fingerprint_id'], len(index), log)
    IDX, W, _, _, frozen = A.load_frozen(args.v2_package, index, log, args.test_fixture)

    conds = [set_name(v, lv) for v in VARIANTS for lv in LEVELS]
    orig = [g.astype(np.float64) for g in data['original_uint8']['gap']]
    trans = {c: [g.astype(np.float64) for g in data[c]['gap']] for c in conds}
    del data
    cka, boots = A.cka_bootstrap(orig, trans, W)
    rev = {set_name(v, lv): (v, lv) for v in VARIANTS for lv in LEVELS}
    cka.insert(0, 'variant', cka.condition.map(lambda c: rev[c][0]))
    cka.insert(1, 'level', cka.condition.map(lambda c: rev[c][1]))
    cka.insert(2, 'factor', cka.level.map(FACTOR))

    pub = pd.read_csv(args.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv')
    z0 = cka.loc[cka.variant == 'Z0', ['level', 'block', 'cka']]
    ref = pub.loc[pub.transformation == 'zoom', ['level', 'block', 'linear_CKA']]
    expected_keys = {(level, block) for level in LEVELS for block in A.BLOCKS}
    for name, table in [('Z0 recalculado', z0), ('tabla publicada', ref)]:
        keys = set(zip(table.level, table.block))
        if len(table) != len(expected_keys) or keys != expected_keys:
            A.fail(log, 8, f'{name}: deben estar las 15 celdas de zoom, una por nivel y bloque')
    mm = z0.merge(ref, on=['level', 'block'], validate='one_to_one')
    err = float(np.max(np.abs(mm.cka.to_numpy() - mm.linear_CKA.to_numpy())))
    if not np.isfinite(err) or err > Z0_CKA_ATOL:
        A.fail(log, 8, f'CKA de Z0 frente a la tabla publicada: error máximo {err:.2e}, tolerancia {Z0_CKA_ATOL:.0e}')
    checks['CKA_Z0_new_vs_published_max_abs_err'] = err
    checks['CKA_Z0_cells_verified'] = len(mm)
    checks['CKA_Z0_absolute_tolerance'] = Z0_CKA_ATOL
    log(f'E1: CKA del zoom original (Z0) frente a la tabla publicada: error máximo {err:.2e} ({len(mm)} celdas).')

    pt = cka.set_index(['condition', 'block']).cka
    rows, diffs = [], []
    for lv in LEVELS:
        z0 = set_name('Z0', lv)
        for v in VARIANTS[1:]:
            c = set_name(v, lv)
            for bi, b in enumerate(A.BLOCKS):
                db = boots[(c, bi)] - boots[(z0, bi)]
                lo, hi = np.percentile(db, [2.5, 97.5])
                diffs.append({'variant': v, 'level': lv, 'factor': FACTOR[lv], 'block': b,
                              'cka_minus_Z0': pt[(c, b)] - pt[(z0, b)], 'ci_low': lo, 'ci_high': hi})
        for v in VARIANTS:
            c = set_name(v, lv)
            vals = [pt[(c, b)] for b in A.BLOCKS]
            Db = boots[(c, 0)] - boots[(c, 4)]
            D = vals[0] - vals[4]
            lo, hi = np.percentile(Db, [2.5, 97.5])
            row = {'variant': v, 'level': lv, 'factor': FACTOR[lv], 'D_B1_minus_B5': D, 'D_ci_low': lo, 'D_ci_high': hi,
                   'argmin_block': A.BLOCKS[int(np.argmin(vals))],
                   'increasing_steps': ', '.join(f'{A.BLOCKS[i]}→{A.BLOCKS[i + 1]} ({vals[i]:.3f} → {vals[i + 1]:.3f})'
                                                 for i in range(4) if vals[i + 1] > vals[i])}
            rows.append(row)
    desc = pd.DataFrame(rows)
    for r in desc.itertuples():
        log(f'E1: {r.variant} {r.factor:.2f}×: D = {r.D_B1_minus_B5:.3f} [{r.D_ci_low:.3f}, {r.D_ci_high:.3f}]'
            + (f'; sube en {r.increasing_steps}' if r.increasing_steps else ''))

    cka.to_csv(out / '20_cka_zoom_por_bloque.csv', index=False, float_format='%.6g')
    desc.to_csv(out / '21_cka_zoom_descenso_B1_B5.csv', index=False, float_format='%.6g')
    pd.DataFrame(diffs).to_csv(out / '22_cka_zoom_diferencia_Z0.csv', index=False, float_format='%.6g')
    manifest = {
        'analysis': 'E1: CKA bajo los controles de zoom (descriptivo, no preespecificado; sin veredicto)',
        'script': script.name, 'script_sha256': sha256(script),
        'analyze_adenda_v3_sha256': h_mod, 'test_fixture': args.test_fixture,
        'valid_for_manuscript': not args.test_fixture,
        'extraction_fingerprint_id': fpinfo['fingerprint_id'], 'checks': checks, 'repo_files_sha256': repo_hashes,
        'frozen_inputs_sha256': frozen, 'n_boot': int(IDX.shape[0]),
        'conditions': conds, 'reference': 'original_uint8 (ruta uint8), como el CKA de C4',
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
                        'pandas': pd.__version__},
        'note': ('El zoom original se denomina Z0 (conjuntos zoom_low/medium/high de la extracción). La tabla 22 '
                 'compara cada control con Z0 bloque a bloque, con las mismas réplicas bootstrap.'),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    log('E1: registro cerrado. A continuación se calculan los checksums de salida.')
    with open(out / 'checksums_outputs.sha256', 'w') as f:
        for p in sorted(out.iterdir()):
            if p.name != 'checksums_outputs.sha256':
                f.write(f'{sha256(p)}  {p.name}\n')


if __name__ == '__main__':
    main()
