"""Verificacion previa de solo lectura para la revision mayor (jimaging-4632369).

Parte 1 (siempre): integridad de los 13 NPZ de embeddings GAP guardados por el
notebook 02 en Drive.
  - existencia, tamano y SHA-256 de cada archivo;
  - claves Block1..Block5, dimensiones (2527 x 64/128/256/512/512), float32, finitos;
  - indice de imagenes del NPZ (Drive) frente al manifiesto publicado en el repositorio;
  - alineacion: coseno y distancia relativa recalculados frente al CSV R1 publicado
    (151,620 filas), para las 12 condiciones y 5 bloques;
  - referencias: se repite el muestreo de pares (semilla 2026) y se compara con las
    300,000 similitudes por bloque publicadas en baseline_pair_similarity_samples.npz.

Parte 2 (opcional, --png-sample K): comprueba en K imagenes por clase que regenerar
las 12 transformaciones en memoria reproduce pixel a pixel los PNG almacenados.
Requiere Pillow y torchvision; registra sus versiones.

No escribe nada fuera de --out. No modifica Drive ni el repositorio.

Uso en Colab:
    from google.colab import drive; drive.mount('/content/gdrive')
    !git clone --depth 1 https://github.com/jorge-necochea/vgg16-trashnet-representational-sensitivity /content/repo
    !python verify_inputs.py --base /content/gdrive/MyDrive/2026-0 --repo /content/repo \
        --out /content/verificacion --png-sample 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_IMAGES = 2527
N_BASELINE_PAIRS = 100_000
EPS = 1e-8
BLOCKS = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5']
DIMS = {'Block1': 64, 'Block2': 128, 'Block3': 256, 'Block4': 512, 'Block5': 512}
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CONDITIONS = {
    'brightness': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'contrast': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'rotation': {'low': 10, 'medium': 30, 'high': 60},
    'zoom': {'low': 1.10, 'medium': 1.25, 'high': 1.50},
}
# Tolerancias del propio notebook 02 (auditoria de reproduccion).
ATOL_COSINE = 1e-6
ATOL_RELATIVE = 3e-6
ATOL_BASELINE = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_npz_file(path: Path) -> tuple[dict, dict[str, np.ndarray] | None]:
    rec = {'file': str(path), 'exists': path.exists()}
    if not rec['exists']:
        rec['passed'] = False
        return rec, None
    rec['size_bytes'] = path.stat().st_size
    rec['sha256'] = sha256(path)
    try:
        arrays = load_npz(path)
    except Exception as exc:  # archivo corrupto o con pickle
        rec['load_error'] = repr(exc)
        rec['passed'] = False
        return rec, None
    rec['keys'] = sorted(arrays)
    rec['keys_ok'] = set(arrays) == set(BLOCKS)
    ok = rec['keys_ok']
    for b in BLOCKS:
        a = arrays.get(b)
        if a is None:
            continue
        rec[f'{b}_shape'] = list(a.shape)
        rec[f'{b}_dtype'] = str(a.dtype)
        rec[f'{b}_finite'] = bool(np.isfinite(a).all())
        # Informativo (no es criterio de aprobacion): fraccion de canales GAP exactamente nulos.
        rec[f'{b}_exact_zero_fraction'] = float(np.mean(a == 0))
        ok &= a.shape == (N_IMAGES, DIMS[b]) and a.dtype == np.float32 and rec[f'{b}_finite']
    rec['passed'] = bool(ok)
    return rec, arrays


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Misma formula que vector_metrics_numpy del notebook 02.
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    return np.sum(x * y, axis=1) / (np.maximum(nx, EPS) * np.maximum(ny, EPS))


def part1(base: Path, repo: Path, out: Path) -> dict:
    emb_root = base / 'Revision_R1_Baseline_CKA_VGG16' / 'embeddings'
    files = {('original', None): emb_root / 'embeddings_originales_VGG16_R1.npz'}
    for t, levels in CONDITIONS.items():
        for lv in levels:
            files[(t, lv)] = emb_root / 'transformadas' / f'embeddings_{t}_{lv}.npz'

    report: dict = {'embedding_root': str(emb_root)}
    file_records, arrays = [], {}
    for key, path in files.items():
        rec, arr = check_npz_file(path)
        rec['condition'] = 'original' if key[1] is None else f'{key[0]}_{key[1]}'
        file_records.append(rec)
        if arr is not None and rec['passed']:
            arrays[key] = arr
        print(f"[NPZ] {rec['condition']:18s} existe={rec['exists']} ok={rec['passed']}")
    pd.DataFrame(file_records).to_csv(out / '01_npz_files.csv', index=False)
    report['npz_files_found'] = int(sum(r['exists'] for r in file_records))
    report['npz_files_passed'] = int(sum(r['passed'] for r in file_records))

    # Indice de imagenes: Drive frente al manifiesto publicado.
    idx_drive_path = emb_root / 'indice_imagenes_originales_R1.csv'
    manifest = pd.read_csv(repo / 'data' / 'manifests' / 'original_image_index.csv')
    cols = ['row_index', 'image_id', 'original_filename', 'class']
    if idx_drive_path.exists():
        idx_drive = pd.read_csv(idx_drive_path)
        same = len(idx_drive) == len(manifest) and idx_drive[cols].reset_index(drop=True).equals(
            manifest[cols].reset_index(drop=True))
        report['index_drive_matches_manifest'] = bool(same)
    else:
        report['index_drive_matches_manifest'] = None  # el archivo no existe; se usa el manifiesto
    index = manifest[cols].copy()
    print(f"[IDX] indice Drive = manifiesto: {report['index_drive_matches_manifest']}")

    if ('original', None) not in arrays:
        report['alignment'] = 'omitido: falta o falla el NPZ de originales'
        report['baseline'] = 'omitido: falta o falla el NPZ de originales'
        return report
    orig = arrays[('original', None)]

    # Alineacion con el CSV R1 publicado.
    published = pd.read_csv(repo / 'data' / 'derived_metrics' / 'r1' /
                            'image_level_adjusted_sensitivity.csv.gz')
    align_rows = []
    for (t, lv), arr in arrays.items():
        if lv is None:
            continue
        sub = published[(published.transformation == t) & (published.level == lv)]
        for b in BLOCKS:
            x, y = orig[b].astype(np.float64), arr[b].astype(np.float64)
            calc = index[['image_id', 'class']].copy()
            calc['cos_calc'] = cosine_rows(x, y)
            calc['rel_calc'] = np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + EPS)
            m = calc.merge(sub[sub.block == b][['image_id', 'class', 'cosine_similarity',
                                                'relative_euclidean_distance']],
                           on=['image_id', 'class'], how='inner', validate='one_to_one')
            e_cos = float(np.max(np.abs(m.cos_calc - m.cosine_similarity)))
            e_rel = float(np.max(np.abs(m.rel_calc - m.relative_euclidean_distance)))
            align_rows.append({'transformation': t, 'level': lv, 'block': b, 'n_matched': len(m),
                               'max_abs_err_cosine': e_cos, 'max_abs_err_relative': e_rel,
                               'passed': len(m) == N_IMAGES and e_cos <= ATOL_COSINE
                               and e_rel <= ATOL_RELATIVE})
    align = pd.DataFrame(align_rows)
    align.to_csv(out / '02_alignment_vs_published_R1.csv', index=False)
    report['alignment_cells_checked'] = len(align)
    report['alignment_cells_passed'] = int(align.passed.sum()) if len(align) else 0
    if len(align):
        report['alignment_max_abs_err_cosine'] = float(align.max_abs_err_cosine.max())
        report['alignment_max_abs_err_relative'] = float(align.max_abs_err_relative.max())
    print(f"[ALN] celdas aprobadas {report['alignment_cells_passed']}/{report['alignment_cells_checked']} (esperadas 60)")

    # Repeticion exacta del muestreo de pares del notebook 02.
    pair_i, pair_j = np.triu_indices(N_IMAGES, k=1)
    labels = index['class'].to_numpy()
    same_mask = labels[pair_i] == labels[pair_j]
    eligible = {'random': np.arange(len(pair_i)), 'same_class': np.flatnonzero(same_mask),
                'different_class': np.flatnonzero(~same_mask)}
    rng = np.random.default_rng(SEED)
    selected = {k: rng.choice(v, size=min(N_BASELINE_PAIRS, len(v)), replace=False)
                for k, v in eligible.items()}
    published_pairs = load_npz(repo / 'data' / 'derived_metrics' / 'r1' /
                               'baseline_pair_similarity_samples.npz')
    base_rows = []
    for b in BLOCKS:
        e = orig[b].astype(np.float64)
        nrm = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), EPS)
        for pt, sel in selected.items():
            sims = np.sum(nrm[pair_i[sel]] * nrm[pair_j[sel]], axis=1).astype(np.float32)
            ref = published_pairs[f'{b}_{pt}']
            err = float(np.max(np.abs(sims.astype(np.float64) - ref))) if sims.shape == ref.shape else None
            base_rows.append({'block': b, 'pair_type': pt, 'median_recomputed': float(np.median(sims)),
                              'median_published': float(np.median(ref)), 'max_abs_err': err,
                              'passed': err is not None and err <= ATOL_BASELINE})
    base = pd.DataFrame(base_rows)
    base.to_csv(out / '03_baseline_pairs_vs_published.csv', index=False)
    report['baseline_cells_passed'] = int(base.passed.sum())
    report['baseline_cells_checked'] = len(base)
    print(f"[REF] referencias aprobadas {report['baseline_cells_passed']}/{len(base)} (esperadas 15)")
    return report


def part2(base: Path, repo: Path, out: Path, k_per_class: int) -> dict:
    import PIL
    import torchvision
    from PIL import Image

    spec = importlib.util.spec_from_file_location('gen', repo / 'scripts' / 'generate_transformations_png.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    original_root = base / 'DatasetOriginalCompleto'
    transformed_root = base / 'TrashNet_transformaciones_completo_png'
    manifest = pd.read_csv(repo / 'data' / 'manifests' / 'original_image_index.csv')
    sample = (manifest.groupby('class', group_keys=False)
              .sample(n=k_per_class, random_state=SEED).sort_values('row_index'))
    rows = []
    for r in sample.to_dict('records'):  # 'class' es palabra reservada: no usar itertuples
        cls, fname = r['class'], r['original_filename']
        with Image.open(original_root / cls / fname) as src:
            rgb = src.convert('RGB')
            for t, levels in CONDITIONS.items():
                for lv, val in levels.items():
                    stored_path = transformed_root / f'{t}_{lv}' / cls / f'{Path(fname).stem}.png'
                    rec = {'image_id': r['image_id'], 'class': cls, 'transformation': t,
                           'level': lv, 'stored_exists': stored_path.exists()}
                    if rec['stored_exists']:
                        regen = gen.transform(rgb, t, val)
                        with Image.open(stored_path) as st:
                            rec['stored_mode'] = st.mode
                            rec['size_match'] = st.size == regen.size
                            a = np.asarray(st.convert('RGB'), dtype=np.int16)
                        b = np.asarray(regen, dtype=np.int16)
                        if a.shape == b.shape:
                            d = np.abs(a - b)
                            rec['max_abs_diff'] = int(d.max())
                            rec['mae'] = float(d.mean())
                            rec['pct_pixels_different'] = float(100 * np.mean(d.max(axis=2) > 0))
                        rec['identical'] = rec.get('max_abs_diff') == 0 and rec['size_match']
                    else:
                        rec['identical'] = False
                    rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / '04_png_regeneration_check.csv', index=False)
    by_cond = df.groupby(['transformation', 'level']).agg(
        n=('identical', 'size'), identical=('identical', 'sum'),
        max_abs_diff=('max_abs_diff', 'max'), max_pct_pixels_different=('pct_pixels_different', 'max'))
    by_cond.to_csv(out / '05_png_regeneration_by_condition.csv')
    print(by_cond.to_string())
    return {'png_sample_per_class': k_per_class, 'png_checks': len(df),
            'png_identical': int(df.identical.sum()), 'pillow': PIL.__version__,
            'torchvision': torchvision.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, type=Path, help='Raiz del proyecto en Drive (2026-0)')
    ap.add_argument('--repo', required=True, type=Path, help='Clon local del repositorio publico')
    ap.add_argument('--out', required=True, type=Path, help='Carpeta de salida para los informes')
    ap.add_argument('--png-sample', type=int, default=0, help='Imagenes por clase para la parte 2 (0 = omitir)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'pandas': pd.__version__, 'seed': SEED}
    report['part1'] = part1(args.base, args.repo, args.out)
    if args.png_sample > 0:
        report['part2'] = part2(args.base, args.repo, args.out, args.png_sample)
    (args.out / 'verification_report.json').write_text(json.dumps(report, indent=2, default=str))
    print('\n=== RESUMEN (copiar y pegar) ===')
    print(json.dumps({k: v for k, v in report.items() if k.startswith('part')}, indent=2, default=str))


if __name__ == '__main__':
    main()
