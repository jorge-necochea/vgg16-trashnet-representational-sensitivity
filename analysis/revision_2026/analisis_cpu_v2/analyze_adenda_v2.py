"""Análisis en CPU de la extracción de la Adenda 1 v1.2 — revisión jimaging-4632369. Versión 2.

Aplica, sin cambiarlas, la Adenda 1 v1.2 (SHA-256 8f0c7b19…a156) y los criterios v1.3 (SHA-256 492a0625…ddf57).
Una sola implementación; ChatGPT audita.

Cambios de la versión 2 (auditoría de ChatGPT de analisis_cpu_v1.zip; la adenda no cambia):
  1. Vínculo criptográfico con la extracción auditada: SHA-256 fijado de H_checksums.sha256 y de los tres
     archivos del repositorio que se usan (índice, pares de referencia, tabla CKA). El error de los pares de
     referencia recalculados con los originales nuevos detiene el análisis si supera TOL_REFERENCE_PAIRS.
  2. Diagnósticos de gamma completos: mediana e IQR de ΔSD(Y) para el brillo multiplicativo y para gamma, y
     proporción media y máxima de saturación nueva por píxel y por componente. La redacción «sin saturación
     nueva» depende de esos valores.
  3. Región pico de cada una de las 20 semillas de F4 (tabla 06b), además del recuento.

Orden:
  0  Validación previa (si falla algo, se detiene sin analizar):
     - checksums de la extracción (H_checksums.sha256): todos los archivos listados, ninguno de más;
     - huella de ejecución: run_fingerprint.json reproduce su propio fingerprint_id y coincide con el manifiesto;
       extracción completa (no --smoke), 2527 imágenes, script de extracción v2, control de píxeles completo
       y aprobado, control de embeddings aprobado;
     - metadatos de los 35 NPZ (nombre, huella, filas, umbrales de casi cero, gamma);
     - bootstrap y extracciones balanceadas congelados del paquete v2, verificados por SHA-256.
  1  Hallazgo post-extracción: pares de imágenes duplicadas entre clases (se registran; no se excluyen).
  2  Diagnósticos: calibración de gamma, B_float(b), equivalencia de rutas, máscaras de rotación, R2–R0.
  3  Gamma (C1), rotación R1 (C3) y zoom Z1/Z1r/Z2: F1 y F2–F4, regiones, estabilidad, contrastes,
     composición de clases. Gamma también con B(b) del manuscrito como sensibilidad.
  4  CKA bajo R0 y R1 (C4) y razón de descensos q = D_R1 / D_R0.
  5  Sparsity antes del GAP.
  6  Cambio en la salida de la cabeza ImageNet (30 condiciones).
  7  Redacción resultante de cada interpretación (reglas de la adenda).
  8  Manifiesto y checksums (después de cerrar el registro).

Códigos de salida: 2 checksums, 3 huella o controles de la extracción, 4 metadatos de un NPZ,
5 bootstrap o extracciones congeladas.

Uso:
  python analyze_adenda_v2.py --extraction RUTA/Revision_Adenda1_extraccion_v2 --v2-package RUTA/run_v2 \
      --repo RUTA/repo --out RUTA/analisis_adenda_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

# ----------------------------------------------------------------------------- constantes congeladas
ADDENDUM_SHA256 = '8f0c7b19079c196f2ad6220340b47adb64b7d775c32d974e8d2a6145decda156'
CRITERIA_SHA256 = '492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57'
EXTRACTION_SCRIPT_SHA256 = 'c103a44e530c7e873b6b3fa4b3f7d54c4a8ba6e46a18e90901f4857de324ec42'
REPO_COMMIT = '6672c2c7e4b84a5b0381de0d70476a97438223e5'
V2_BOOTSTRAP_SHA256 = 'c2f882d2113cb86bcdcfae60d2dc3d9701494c3e7733fc96483ba8b4cecd8ff5'
V2_DRAWS_SHA256 = '1b1ea9eea179c9b644b9049ede9aa8042cc1e76a5f4ab0457b775165b6f85730'
N_EXPECTED, N_BOOT_EXPECTED, N_DRAWS_EXPECTED, PER_CLASS_DRAW = 2527, 2000, 200, 137
N_CHECKSUM_FILES_EXPECTED = 47
# Vínculo con la extracción auditada y con los archivos del repositorio (commit fijado) que usa el análisis.
H_CHECKSUMS_SHA256 = '409ff94360f15f08e7ad79468863bfb740be04ac57bb020be74cd97a24b770e4'
REPO_FILES_SHA256 = {
    'data/manifests/original_image_index.csv': '796f11e8cf3307068824bfe868902cff23cdeaf56a68628c1e50f8193b43257f',
    'data/derived_metrics/r1/baseline_pair_similarity_samples.npz':
        '084dd90e2089a62e8c35c1b484c1c137d4426b657e5aaaa112061fee23b2f482',
    'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv':
        '819619d07936fc18e2e4fe80db2881fbcf2ac49ccf6a4ddd507c1ac1e7bfe367',
}
# Coseno de los 100 000 pares de referencia recalculado con los originales nuevos frente al publicado:
# diferencia absoluta máxima admitida (la misma escala que la tolerancia de A_adj del control de embeddings).
TOL_REFERENCE_PAIRS = 1e-5
# Hallazgo post-extracción comunicado el 23-11-2026: tres pares de imágenes idénticas en clases distintas.
EXPECTED_DUPLICATES = [('glass/glass115.jpg', 'metal/metal91.jpg'),
                       ('glass/glass176.jpg', 'plastic/plastic152.jpg'),
                       ('glass/glass389.jpg', 'plastic/plastic332.jpg')]

SEED = 2026
N_PAIRS = 100_000
BLOCKS = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5']
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LEVELS = ['low', 'medium', 'high']
EXISTING = {'brightness': [1.2, 1.4, 1.6], 'contrast': [1.2, 1.4, 1.6], 'rotation': [10, 30, 60],
            'zoom': [1.10, 1.25, 1.50]}
ANGLE = {'low': 10, 'medium': 30, 'high': 60}
ZOOMF = {'low': 1.10, 'medium': 1.25, 'high': 1.50}
BLOCK_REGION = np.array([0, 0, 1, 2, 2])
REGION_NAME = {0: 'superficial', 1: 'intermedia', 2: 'profunda'}
F4_SEEDS = list(range(2026, 2046))
F4_DIM = 64
EPS = 1e-8
VERDICT_ORDER = ['Robusta', 'Parcialmente robusta', 'Dependiente de la métrica',
                 'Región estable, efecto no relevante', 'No sostenida']
AUX_R2 = ['r2_mae_R', 'r2_mae_G', 'r2_mae_B', 'r2_max_abs_diff', 'r2_pct_pixels_different']
AUX_GAMMA = ['gamma_Y_mean_original', 'gamma_Y_mean_transformed', 'gamma_Y_std_original',
             'gamma_Y_std_transformed', 'gamma_newly_saturated_pixel_fraction',
             'gamma_newly_saturated_component_fraction']

# Controles de región: bloque de referencia (índice), región esperada, bloques fuera de la región para la
# relevancia y bloques de dirección (criterios v1.3 de C1, C3 y C5, tal como los aplica la adenda).
CONTROL_RULES = {
    'gamma': dict(ref=1, region=0, outside=[2, 3, 4], direction=[], claim='C1'),
    'R1': dict(ref=2, region=1, outside=[0, 1, 3, 4], direction=[0, 4], claim='C3'),
    'R2': dict(ref=2, region=1, outside=[0, 1, 3, 4], direction=[0, 4], claim='C3'),
    'Z1': dict(ref=4, region=2, outside=[0, 1, 2], direction=[0], claim='C5'),
    'Z1r': dict(ref=4, region=2, outside=[0, 1, 2], direction=[0], claim='C5'),
    'Z2': dict(ref=4, region=2, outside=[0, 1, 2], direction=[0], claim='C5'),
}


def build_set_names() -> list[str]:
    names = ['original_uint8'] + [f'{t}_{lv}' for t in EXISTING for lv in LEVELS] + ['original_float']
    names += [f'gamma_{lv}' for lv in LEVELS]
    names += [f'rot{v}_{lv}' for v in ['R1', 'R2', 'R3'] for lv in LEVELS]
    names += [f'zoom{v}_{lv}' for v in ['Z1', 'Z1r', 'Z2'] for lv in LEVELS]
    assert len(names) == 35
    return names


SET_NAMES = build_set_names()
TRANSFORMED = [s for s in SET_NAMES if not s.startswith('original')]            # 33
LOGIT_CONDITIONS = [s for s in TRANSFORMED if not s.startswith('rotR2')]       # 30


def route_original(cond: str) -> str:
    return 'original_float' if cond.startswith('gamma') else 'original_uint8'


# ----------------------------------------------------------------------------- utilidades
class Logger:
    def __init__(self, path: Path):
        self.path, self.t0 = path, time.time()
        path.write_text('', encoding='utf-8')

    def __call__(self, msg: str) -> None:
        line = f'[{time.time() - self.t0:8.1f} s] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def fail(log, code: int, msg: str):
    log(f'ERROR (código {code}): {msg}')
    log('Se detiene sin analizar.')
    sys.exit(code)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    """Idéntica a la del script de extracción v2 (huella de ejecución)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':'))


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def cos_rows(x, y):
    return np.sum(normalize(x) * normalize(y), axis=1)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.empty_like(q)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(q) - rank) * q[i])
        adj[i] = min(running, 1.0)
    out[ok] = adj
    return out


def paired_rank_biserial(d: np.ndarray) -> float:
    """r_rb pareada con signo: positiva cuando la primera serie es mayor (d = primera − segunda)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos, neg = r[d > 0].sum(), r[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def indep_rank_biserial(a, b) -> tuple[float, float]:
    """r_rb para muestras independientes, positiva si a tiende a ser mayor que b; Mann–Whitney bilateral."""
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    return float(2 * res.statistic / (len(a) * len(b)) - 1), float(res.pvalue)


def region_of(means: np.ndarray) -> np.ndarray:
    return BLOCK_REGION[np.argmax(means, axis=-1)]


def band(x: float) -> str:
    """Bandas de la v1.3 para |ρ| y |r_rb|."""
    if x is None or np.isnan(x):
        return 'no disponible'
    a = abs(x)
    return 'despreciable' if a < 0.10 else 'pequeña' if a < 0.30 else 'moderada' if a < 0.50 else 'grande'


def boot_means(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return (W @ A.reshape(n, -1) / n).reshape((W.shape[0],) + A.shape[1:])


def boot_spearman(a: np.ndarray, b: np.ndarray, IDX: np.ndarray):
    """ρ de Spearman puntual e IC percentil 2,5–97,5 con las réplicas congeladas; excluye réplicas sin varianza."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan, np.nan, 0
    rho = float(stats.spearmanr(a, b).statistic)
    ra, rb = stats.rankdata(a[IDX], axis=1), stats.rankdata(b[IDX], axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt(np.sum(ra ** 2, 1) * np.sum(rb ** 2, 1))
    valid = den > 0
    if not valid.any():
        return rho, np.nan, np.nan, 0
    r = np.sum(ra[valid] * rb[valid], 1) / den[valid]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return rho, float(lo), float(hi), int(valid.sum())


# ----------------------------------------------------------------------------- 0. validación
def validate_checksums(ext: Path, log, test_mode: bool) -> dict:
    h_file = sha256(ext / 'H_checksums.sha256')
    if h_file != H_CHECKSUMS_SHA256:
        if not test_mode:
            fail(log, 2, f'H_checksums.sha256 ({h_file}) no es el de la extracción auditada ({H_CHECKSUMS_SHA256})')
        log(f'0: (modo de prueba) H_checksums.sha256 distinto del auditado: {h_file}')
    lines = [ln for ln in (ext / 'H_checksums.sha256').read_text(encoding='utf-8').splitlines() if ln.strip()]
    listed = {}
    for ln in lines:
        h, rel = ln.split('  ', 1)
        listed[rel] = h
    bad = [rel for rel, h in listed.items() if not (ext / rel).is_file() or sha256(ext / rel) != h]
    present = {str(p.relative_to(ext)) for p in ext.rglob('*') if p.is_file() and p.name != 'H_checksums.sha256'}
    extra, missing = sorted(present - set(listed)), sorted(set(listed) - present)
    if bad or extra or missing:
        fail(log, 2, f'checksums de la extracción: {len(bad)} distintos, {len(extra)} archivos no listados, '
                     f'{len(missing)} ausentes. Distintos: {bad[:5]}; no listados: {extra[:5]}')
    if not test_mode and len(listed) != N_CHECKSUM_FILES_EXPECTED:
        fail(log, 2, f'se esperaban {N_CHECKSUM_FILES_EXPECTED} archivos en los checksums y hay {len(listed)}')
    log(f'0: checksums de la extracción verificados: {len(listed)}/{len(listed)}; ningún archivo de más; '
        f'H_checksums.sha256 = {h_file}.')
    return {'checksum_files_verified': len(listed), 'H_checksums_sha256': h_file}


def validate_repo_files(repo: Path, log, test_mode: bool) -> dict:
    got = {rel: sha256(repo / rel) for rel in REPO_FILES_SHA256}
    bad = [rel for rel, h in got.items() if h != REPO_FILES_SHA256[rel]]
    if bad and not test_mode:
        fail(log, 5, f'archivos del repositorio distintos de los fijados: {bad}')
    if bad:
        log(f'0: (modo de prueba) archivos del repositorio distintos de los fijados: {bad}')
    else:
        log('0: índice, pares de referencia y tabla CKA del repositorio verificados por SHA-256.')
    return got


def validate_fingerprint(ext: Path, log, test_mode: bool) -> dict:
    fp_file = json.loads((ext / 'run_fingerprint.json').read_text(encoding='utf-8'))
    man = json.loads((ext / 'H_manifiesto.json').read_text(encoding='utf-8'))
    fp = fp_file['fingerprint']
    fid = hashlib.sha256(canonical(fp).encode()).hexdigest()
    hard = []
    if fid != fp_file['fingerprint_id']:
        hard.append('run_fingerprint.json no reproduce su propio fingerprint_id')
    if man.get('fingerprint_id') != fid or canonical(man.get('fingerprint')) != canonical(fp):
        hard.append('la huella del manifiesto no coincide con run_fingerprint.json')
    if not man.get('embedding_gate_passed'):
        hard.append('el control de embeddings no está aprobado')
    if not (man.get('pixel_gate', {}).get('mode') == 'completo' and man.get('pixel_gate', {}).get('passed') is True):
        hard.append(f"control de píxeles no completo o no aprobado: {man.get('pixel_gate')}")
    if not man.get('valid_for_analysis'):
        hard.append('el manifiesto no declara la extracción válida para el análisis')
    gate = pd.read_csv(ext / 'F_control_embeddings.csv')
    if not gate.passed.all():
        hard.append('F_control_embeddings.csv tiene filas no aprobadas')
    identity = {
        'script de extracción v2': fp.get('script_sha256') == EXTRACTION_SCRIPT_SHA256,
        'adenda v1.2': fp.get('addendum_sha256') == ADDENDUM_SHA256,
        'criterios v1.3': fp.get('criteria_sha256') == CRITERIA_SHA256,
        'commit del repositorio': fp.get('repo', {}).get('commit') == REPO_COMMIT,
        'ejecución completa (sin --smoke)': fp.get('mode', {}).get('smoke_images_per_class') == 0,
        '2527 imágenes': fp.get('mode', {}).get('n_images') == N_EXPECTED,
    }
    soft = [k for k, ok in identity.items() if not ok]
    if hard:
        fail(log, 3, '; '.join(hard))
    if soft and not test_mode:
        fail(log, 3, 'la huella no corresponde a la extracción aprobada: ' + ', '.join(soft))
    if soft:
        log(f'0: (modo de prueba) identidad no comprobada: {", ".join(soft)}')
    log(f"0: huella {fid} verificada; extracción completa; controles de píxeles y embeddings aprobados; "
        f"GPU {fp.get('environment', {}).get('gpu')}.")
    return {'fingerprint_id': fid, 'fingerprint': fp, 'extraction_manifest': man, 'identity_checks': identity}


def load_sets(ext: Path, fid: str, n: int, log) -> dict:
    with np.load(ext / 'E_umbrales_casi_cero.npz', allow_pickle=False) as z:
        tdig = str(z['thresholds_digest'])
        if str(z['fingerprint_id']) != fid:
            fail(log, 4, 'E_umbrales_casi_cero.npz pertenece a otra huella')
    cal = json.loads((ext / 'D_calibracion_gamma.json').read_text(encoding='utf-8'))
    gammas = {lv: cal['levels'][lv]['gamma'] for lv in LEVELS}
    data = {}
    for name in SET_NAMES:
        with np.load(ext / 'conjuntos' / f'{name}.npz', allow_pickle=False) as z:
            problems = []
            if str(z['set_name']) != name:
                problems.append('nombre')
            if str(z['fingerprint_id']) != fid:
                problems.append('huella')
            if str(z['thresholds_digest']) != tdig:
                problems.append('umbrales')
            if not np.array_equal(z['row_index'], np.arange(n)):
                problems.append('filas')
            g = float(z['gamma'])
            if name.startswith('gamma'):
                if g != gammas[name.split('_')[1]]:
                    problems.append('gamma')
            elif not np.isnan(g):
                problems.append('gamma no vacío')
            if problems:
                fail(log, 4, f'{name}: metadatos incompatibles ({", ".join(problems)})')
            d = {'gap': [z[f'gap_{b}'] for b in BLOCKS], 's_exact': np.stack([z[f's_exact_{b}'] for b in BLOCKS], 1),
                 's_near': np.stack([z[f's_near_{b}'] for b in BLOCKS], 1), 'logits': z['logits']}
            for k in AUX_R2 + AUX_GAMMA:
                if k in z.files:
                    d[k] = z[k]
            data[name] = d
    log(f'0: 35 conjuntos cargados; nombre, huella, filas, umbrales y gamma verificados en cada uno.')
    return data, cal, gammas


def load_frozen(v2: Path, index: pd.DataFrame, log, test_mode: bool):
    n = len(index)
    p_idx, p_draws = v2 / 'bootstrap_indices.npy', v2 / '10_extracciones_balanceadas_ids.csv'
    h_idx, h_draws = sha256(p_idx), sha256(p_draws)
    if not test_mode and (h_idx != V2_BOOTSTRAP_SHA256 or h_draws != V2_DRAWS_SHA256):
        fail(log, 5, 'bootstrap_indices.npy o 10_extracciones_balanceadas_ids.csv no coinciden con el paquete v2')
    IDX = np.load(p_idx).astype(np.int64)
    labels = index['class'].to_numpy()
    sizes = np.array([np.sum(labels == c) for c in CLASSES])
    lab_code = np.searchsorted(CLASSES, labels)
    ok_strat = all(np.array_equal(np.bincount(lab_code[IDX[r]], minlength=6), sizes) for r in range(len(IDX)))
    if IDX.shape[1] != n or not ok_strat or (not test_mode and IDX.shape[0] != N_BOOT_EXPECTED):
        fail(log, 5, f'bootstrap congelado incompatible: forma {IDX.shape}, estratificado {ok_strat}')
    W = np.stack([np.bincount(IDX[r], minlength=n) for r in range(len(IDX))]).astype(np.float64)
    dr = pd.read_csv(p_draws)
    pos = pd.Series(np.arange(n), index=index.image_id.to_numpy() + '|' + index['class'].to_numpy())
    keys = dr.image_id.to_numpy() + '|' + dr['class'].to_numpy()
    if not np.isin(keys, pos.index).all():
        fail(log, 5, 'hay identificadores de las extracciones balanceadas que no están en el índice')
    rows = pos.loc[keys].to_numpy()
    n_draws = dr.draw.nunique()
    per = dr.groupby('draw').size()
    if per.nunique() != 1:
        fail(log, 5, 'las extracciones balanceadas no tienen todas el mismo tamaño')
    draws = rows.reshape(n_draws, int(per.iloc[0]))
    k = int(per.iloc[0]) // len(CLASSES)
    if not test_mode and (n_draws != N_DRAWS_EXPECTED or k != PER_CLASS_DRAW):
        fail(log, 5, f'se esperaban {N_DRAWS_EXPECTED} extracciones de {PER_CLASS_DRAW} por clase')
    log(f'0: bootstrap congelado ({IDX.shape[0]} réplicas estratificadas) y {n_draws} extracciones balanceadas '
        f'({k} por clase) verificados por SHA-256.')
    return IDX, W, draws, k, {'bootstrap_indices_sha256': h_idx, 'balanced_draws_sha256': h_draws}


def reference_pairs(labels: np.ndarray):
    n = len(labels)
    pair_i, pair_j = np.triu_indices(n, k=1)
    same = labels[pair_i] == labels[pair_j]
    eligible = {'random': np.arange(len(pair_i)), 'same_class': np.flatnonzero(same),
                'different_class': np.flatnonzero(~same)}
    rng = np.random.default_rng(SEED)
    sel = {k: rng.choice(v, size=min(N_PAIRS, len(v)), replace=False) for k, v in eligible.items()}
    return pair_i[sel['random']], pair_j[sel['random']]


# ----------------------------------------------------------------------------- 1. duplicados
def duplicates_finding(ext: Path, index: pd.DataFrame, data: dict, draws: np.ndarray, pi, pj, log):
    jp = pd.read_csv(ext / 'A_jpeg_sha256.csv')
    expected_files = (index['class'] + '/' + index.original_filename).tolist()
    if jp.file.tolist() != expected_files or not np.array_equal(jp.row_index.to_numpy(), np.arange(len(index))):
        fail(log, 3, 'A_jpeg_sha256.csv no sigue el orden de original_image_index.csv')
    gap = np.concatenate([g.astype(np.float64) for g in data['original_uint8']['gap']], 1)
    groups = {}
    for i, row in enumerate(gap):
        groups.setdefault(row.tobytes(), []).append(i)
    pairs = set()
    for rows in list(groups.values()) + [g.index.tolist() for _, g in jp.groupby('sha256') if len(g) > 1]:
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                pairs.add(tuple(sorted((rows[a], rows[b]))))
    pair_set = set(zip(np.minimum(pi, pj).tolist(), np.maximum(pi, pj).tolist()))
    exp = {tuple(sorted(p)) for p in EXPECTED_DUPLICATES}
    out = []
    for a, b in sorted(pairs):
        fa, fb = expected_files[a], expected_files[b]
        in_a, in_b = (draws == a).any(1), (draws == b).any(1)
        out.append({'row_a': a, 'file_a': fa, 'class_a': index['class'][a], 'row_b': b, 'file_b': fb,
                    'class_b': index['class'][b], 'cross_class': index['class'][a] != index['class'][b],
                    'jpeg_sha256_identical': jp.sha256[a] == jp.sha256[b],
                    'gap_identical_all_blocks': bool(np.array_equal(gap[a], gap[b])),
                    'max_abs_gap_diff': float(np.max(np.abs(gap[a] - gap[b]))),
                    'in_expected_list': tuple(sorted((fa, fb))) in exp,
                    'n_balanced_draws_with_both': int(np.sum(in_a & in_b)),
                    'n_balanced_draws_with_either': int(np.sum(in_a | in_b)),
                    'pair_in_100k_reference_pairs': (a, b) in pair_set})
    dup = pd.DataFrame(out)
    n_cross = int(dup.cross_class.sum()) if len(dup) else 0
    found = ({tuple(sorted((r.file_a, r.file_b))) for r in dup.itertuples() if r.cross_class}
             if len(dup) else set())
    log(f'1: HALLAZGO POST-EXTRACCIÓN: {len(dup)} pares de imágenes idénticas ({n_cross} entre clases distintas). '
        'Se registran y NO se excluyen del análisis principal.')
    if found != exp:
        log(f'1: ATENCIÓN: los pares entre clases encontrados no coinciden con los tres comunicados. '
            f'Encontrados: {sorted(found)}')
    return dup, {'n_pairs': len(dup), 'n_cross_class': n_cross, 'matches_expected_three': found == exp}


# ----------------------------------------------------------------------------- 2. diagnósticos
def diagnostics(ext: Path, data: dict, cal: dict, pi, pj, log):
    geo = json.loads((ext / 'C_rotacion_mascaras.json').read_text(encoding='utf-8'))
    rows, gsat = [], {}
    per = pd.read_csv(ext / 'D_calibracion_gamma_por_imagen.csv')
    if not np.array_equal(per.row_index.to_numpy(), np.arange(len(data['original_uint8']['logits']))):
        fail(log, 3, 'D_calibracion_gamma_por_imagen.csv no sigue el orden del índice')

    def mq(x):
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        return med, q1, q3

    for lv in LEVELS:
        c = cal['levels'][lv]
        g = data[f'gamma_{lv}']
        d_sd_b = per[f'brightness_{lv}_Y_std'].to_numpy() - per['Y_std_original'].to_numpy()
        d_sd_g = g['gamma_Y_std_transformed'].astype(np.float64) - g['gamma_Y_std_original'].astype(np.float64)
        d_y_g = g['gamma_Y_mean_transformed'].astype(np.float64) - g['gamma_Y_mean_original'].astype(np.float64)
        sp, sc = g['gamma_newly_saturated_pixel_fraction'], g['gamma_newly_saturated_component_fraction']
        gsat[lv] = {'pixel_mean': float(sp.mean()), 'pixel_max': float(sp.max()),
                    'component_mean': float(sc.mean()), 'component_max': float(sc.max()),
                    'component_global_from_histogram': float(c.get('gamma_newly_saturated_component_fraction', np.nan))}
        b = mq(d_sd_b)
        gg = mq(d_sd_g)
        rows.append({'section': 'gamma', 'level': lv,
                     **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()},
                     'brightness_delta_SD_Y_median': b[0], 'brightness_delta_SD_Y_q1': b[1],
                     'brightness_delta_SD_Y_q3': b[2],
                     'gamma_delta_SD_Y_median': gg[0], 'gamma_delta_SD_Y_q1': gg[1], 'gamma_delta_SD_Y_q3': gg[2],
                     'gamma_delta_Y_mean_from_extraction_median': float(np.median(d_y_g)),
                     'gamma_newly_saturated_pixel_fraction_mean': gsat[lv]['pixel_mean'],
                     'gamma_newly_saturated_pixel_fraction_max': gsat[lv]['pixel_max'],
                     'gamma_newly_saturated_component_fraction_mean': gsat[lv]['component_mean'],
                     'gamma_newly_saturated_component_fraction_max': gsat[lv]['component_max'],
                     'max_abs_diff_Y_mean_original_D_vs_extraction':
                         float(np.max(np.abs(per['Y_mean_original'].to_numpy()
                                             - g['gamma_Y_mean_original'].astype(np.float64))))})
    for lv in LEVELS:
        rows.append({'section': 'rotacion_mascaras', 'level': lv, **geo[lv]})
    diag = pd.DataFrame(rows)
    # B_float(b), B(b) recalculado con la ruta uint8 y equivalencia de rutas para los originales.
    route = []
    for bi, b in enumerate(BLOCKS):
        u = normalize(data['original_uint8']['gap'][bi].astype(np.float64))
        f = normalize(data['original_float']['gap'][bi].astype(np.float64))
        c = np.sum(u * f, 1)
        route.append({'block': b, 'B_uint8_recomputed': float(np.median(np.sum(u[pi] * u[pj], 1))),
                      'B_float': float(np.median(np.sum(f[pi] * f[pj], 1))),
                      'cos_uint8_vs_float_mean': float(c.mean()), 'cos_uint8_vs_float_p1': float(np.percentile(c, 1)),
                      'cos_uint8_vs_float_min': float(c.min())})
    route = pd.DataFrame(route)
    route['routes_equivalent_block'] = (route.cos_uint8_vs_float_mean >= 0.9985) & (route.cos_uint8_vs_float_p1 >= 0.9949)
    # R2 frente a R0 (recalculado desde los conjuntos, no copiado de G).
    req = []
    for lv in LEVELS:
        r0, r2 = data[f'rotation_{lv}'], data[f'rotR2_{lv}']
        for bi, b in enumerate(BLOCKS):
            c = cos_rows(r0['gap'][bi].astype(np.float64), r2['gap'][bi].astype(np.float64))
            req.append({'level': lv, 'angle': ANGLE[lv], 'block': b, 'cos_mean': float(c.mean()),
                        'cos_p1': float(np.percentile(c, 1)), 'cos_min': float(c.min()),
                        'MAE_R': float(r2['r2_mae_R'].mean()), 'MAE_G': float(r2['r2_mae_G'].mean()),
                        'MAE_B': float(r2['r2_mae_B'].mean()), 'max_abs_diff': float(r2['r2_max_abs_diff'].max()),
                        'pct_pixels_different_mean': float(r2['r2_pct_pixels_different'].mean())})
    r2 = pd.DataFrame(req)
    r2['equivalent_cell'] = (r2.cos_mean >= 0.9985) & (r2.cos_p1 >= 0.9949) & (r2[['MAE_R', 'MAE_G', 'MAE_B']].max(axis=1) <= 1)
    r2_eq = bool(r2.equivalent_cell.all())
    sat_max = max(max(v['pixel_max'], v['component_max'], v['component_global_from_histogram']
                      if not np.isnan(v['component_global_from_histogram']) else 0.0) for v in gsat.values())
    log(f"2: B_float = {', '.join(f'{v:.4f}' for v in route.B_float)}; rutas equivalentes en los 5 bloques: "
        f"{bool(route.routes_equivalent_block.all())}; R2 equivalente a R0: {r2_eq}; "
        f"saturación nueva en gamma (máxima por imagen, píxel o componente): {sat_max:.3g}.")
    return diag, route, r2, r2_eq, gsat


# ----------------------------------------------------------------------------- 3. familias y controles de región
def families(orig: list, trans: dict, B_F1: np.ndarray, pi, pj):
    """F1–F4 de la v1.3 para las condiciones de un control, con los originales de su misma ruta."""
    n, K = orig[0].shape[0], len(BLOCKS)
    conds = list(trans)
    C = len(conds)
    S = np.empty((n, C, K))
    for ci, c in enumerate(conds):
        for bi in range(K):
            S[:, ci, bi] = cos_rows(orig[bi], trans[c][bi])
    fam = {'F1': (1 - S) / (1 - B_F1)}
    A2, A3, F3_min = np.empty((n, C, K)), np.empty((n, C, K)), np.empty((C, K))
    for bi in range(K):
        mu = orig[bi].mean(axis=0)
        xc = normalize(orig[bi] - mu)
        B2 = np.median(np.sum(xc[pi] * xc[pj], 1))
        xn = normalize(orig[bi])
        D = np.sort(1 - np.sum(xn[pi] * xn[pj], 1))
        for ci, c in enumerate(conds):
            A2[:, ci, bi] = (1 - np.sum(xc * normalize(trans[c][bi] - mu), 1)) / (1 - B2)
            cnt = np.searchsorted(D, 1 - S[:, ci, bi], side='right')
            A3[:, ci, bi] = (0.5 + cnt) / (len(D) + 1)
            F3_min[ci, bi] = np.mean(cnt == 0)
    fam['F2'], fam['F3'] = A2, A3
    A4 = np.empty((len(F4_SEEDS), n, C, K))
    for si, seed in enumerate(F4_SEEDS):
        rng = np.random.default_rng(seed)
        for bi in range(K):
            Q, _ = np.linalg.qr(rng.standard_normal((orig[bi].shape[1], F4_DIM)))
            xp = normalize(orig[bi] @ Q)
            B4 = np.median(np.sum(xp[pi] * xp[pj], 1))
            for ci, c in enumerate(conds):
                A4[si, :, ci, bi] = (1 - np.sum(xp * normalize(trans[c][bi] @ Q), 1)) / (1 - B4)
    fam['F4'] = A4.mean(axis=0)
    F3_noninf = (F3_min > 0.5).sum(axis=1) >= 3
    return S, fam, A4, F3_noninf, F3_min


def region_control(control: str, conds: list, orig: list, trans: dict, B_F1: np.ndarray, pi, pj, IDX, W,
                   G: list, draws, first_k, log, reference_label: str = 'principal'):
    """Un control de región (gamma, R1, R2, Z1, Z1r o Z2) con las reglas de la v1.3."""
    rule = CONTROL_RULES[control]
    rb, exp_reg = rule['ref'], rule['region']
    K = len(BLOCKS)
    S, fam, A4, F3_noninf, F3_min = families(orig, trans, B_F1, pi, pj)
    A1 = fam['F1']
    bm = {f: boot_means(A, W) for f, A in fam.items()}
    prof, regs, cts, crows, comp, seeds_rows = [], [], [], [], [], []
    seeds_reg = region_of(A4.mean(axis=1))                         # (20, C)
    # B recalculado en cada extracción balanceada (todos los pares únicos), igual para todas las condiciones
    iu = np.triu_indices(draws.shape[1], k=1)
    B_draw = np.array([[np.median(G[bi][np.ix_(d, d)][iu]) for bi in range(K)] for d in draws])
    iu_k = np.triu_indices(len(first_k), k=1)
    Bk = np.array([np.median(G[bi][np.ix_(first_k, first_k)][iu_k]) for bi in range(K)])
    for ci, cond in enumerate(conds):
        lv = cond.split('_')[-1]
        base = {'control': control, 'reference': reference_label, 'condition': cond, 'level': lv}
        for f, A in fam.items():
            mean, med = A[:, ci].mean(0), np.median(A[:, ci], 0)
            lo, hi = np.percentile(bm[f][:, ci], 2.5, 0), np.percentile(bm[f][:, ci], 97.5, 0)
            for bi, b in enumerate(BLOCKS):
                prof.append({**base, 'family': f, 'block': b, 'mean': mean[bi], 'median': med[bi],
                             'mean_ci_low': lo[bi], 'mean_ci_high': hi[bi]})
            point = int(region_of(mean))
            stab = float(np.mean(region_of(bm[f][:, ci]) == exp_reg))
            row = {**base, 'family': f, 'expected_region': REGION_NAME[exp_reg],
                   'peak_block_mean': BLOCKS[int(np.argmax(mean))], 'peak_region_mean': REGION_NAME[point],
                   'stability_expected_region_mean': stab,
                   'peak_region_median': REGION_NAME[int(region_of(med))]}
            if f == 'F1':
                med_boot = np.stack([np.median(A[:, ci, bi][IDX], axis=1) for bi in range(K)], axis=1)
                row['stability_expected_region_median'] = float(np.mean(region_of(med_boot) == exp_reg))
            if f == 'F4':
                row['F4_seeds_in_expected_region'] = int(np.sum(seeds_reg[:, ci] == exp_reg))
                row['F4_seed_peak_regions'] = json.dumps([REGION_NAME[int(v)] for v in seeds_reg[:, ci]],
                                                         ensure_ascii=False)
                for si, seed in enumerate(F4_SEEDS):
                    m_seed = A4[si, :, ci].mean(0)
                    seeds_rows.append({**base, 'seed': seed, 'peak_block': BLOCKS[int(np.argmax(m_seed))],
                                       'peak_region': REGION_NAME[int(seeds_reg[si, ci])],
                                       'in_expected_region': bool(seeds_reg[si, ci] == exp_reg),
                                       **{f'mean_{b}': float(m_seed[bi]) for bi, b in enumerate(BLOCKS)}})
            if f == 'F3':
                row['F3_noninformative'] = bool(F3_noninf[ci])
                row['F3_share_at_minimum_by_block'] = json.dumps([round(float(v), 4) for v in F3_min[ci]])
            regs.append(row)
        # contrastes del bloque de referencia (F1)
        ct = {}
        for ob in range(K):
            if ob == rb:
                continue
            d = A1[:, ci, rb] - A1[:, ci, ob]
            mb = np.median(d[IDX], axis=1)
            lo, hi = np.percentile(mb, [2.5, 97.5])
            r = {**base, 'reference_block': BLOCKS[rb], 'other_block': BLOCKS[ob],
                 'other_outside_region': ob in rule['outside'], 'F1_median_diff': float(np.median(d)),
                 'F1_mean_diff': float(d.mean()), 'r_rb': paired_rank_biserial(d),
                 'wilcoxon_p': float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue),
                 'median_diff_ci_low': float(lo), 'median_diff_ci_high': float(hi)}
            for f in ['F2', 'F3', 'F4']:
                r[f'{f}_median_diff'] = float(np.median(fam[f][:, ci, rb] - fam[f][:, ci, ob]))
            r['relevant'] = bool(r['r_rb'] >= 0.30 and lo > 0)
            ct[ob] = r
            cts.append(r)
        f1 = [x for x in regs if x['condition'] == cond and x['family'] == 'F1' and x['reference'] == reference_label][-1]
        f1_point_ok = f1['peak_region_mean'] == REGION_NAME[exp_reg]
        f1_stab = f1['stability_expected_region_mean']
        effect_ok = all(ct[ob]['relevant'] for ob in rule['outside'])
        direction_ok = all(ct[ob]['median_diff_ci_low'] > 0 for ob in rule['direction'])
        cumple = bool(f1_point_ok and f1_stab >= 0.90 and effect_ok and direction_ok)
        min_rrb = min(ct[ob]['r_rb'] for ob in ct)
        status, n_inf, n_other = {}, 0, 0
        for f in ['F2', 'F3', 'F4']:
            fr = [x for x in regs if x['condition'] == cond and x['family'] == f and x['reference'] == reference_label][-1]
            if f == 'F3' and F3_noninf[ci]:
                status[f] = 'no informativa'
                continue
            n_inf += 1
            other_region = fr['peak_region_mean'] != REGION_NAME[exp_reg]
            n_other += int(other_region)
            ok = (not other_region) and fr['stability_expected_region_mean'] >= 0.70
            if f == 'F4':
                ok = ok and fr['F4_seeds_in_expected_region'] >= 16
            ok = ok and all(ct[ob][f'{f}_median_diff'] > 0 for ob in rule['direction'])
            status[f] = 'concuerda' if ok else 'discrepa'
        n_disc = sum(v == 'discrepa' for v in status.values())
        if f1_stab < 0.90 or not f1_point_ok:
            verdict = 'No sostenida'
        elif not (effect_ok and direction_ok):
            verdict = 'Región estable, efecto no relevante'
        else:
            verdict = ['Robusta', 'Parcialmente robusta'][n_disc] if n_disc < 2 else 'Dependiente de la métrica'
        if n_inf < 2:
            norm_label = 'evidencia insuficiente para evaluar dependencia de la normalización'
        elif n_other >= 2:
            norm_label = 'dependiente de la normalización'
        elif n_other == 1:
            norm_label = 'discordancia aislada'
        else:
            norm_label = 'sin discordancia de región'
        # composición de clases: 200 extracciones congeladas, B fijo y B recalculado con la misma ruta
        fixed = np.array([region_of(A1[d, ci].mean(0)) for d in draws])
        oms = 1 - S[:, ci]
        rec = np.array([region_of(oms[d].mean(0) / (1 - B_draw[k])) for k, d in enumerate(draws)])
        cons_f, cons_r = float(np.mean(fixed == exp_reg)), float(np.mean(rec == exp_reg))
        comp.append({**base, 'expected_region': REGION_NAME[exp_reg], 'conservation_B_fixed': cons_f,
                     'conservation_B_recomputed': cons_r,
                     'first_k_region_B_fixed': REGION_NAME[int(region_of(A1[first_k, ci].mean(0)))],
                     'first_k_region_B_recomputed': REGION_NAME[int(region_of(oms[first_k].mean(0) / (1 - Bk)))]})
        crows.append({**base, 'claim_criteria': rule['claim'], 'reference_block': BLOCKS[rb],
                      'expected_region': REGION_NAME[exp_reg], 'F1_peak_block': f1['peak_block_mean'],
                      'F1_peak_region': f1['peak_region_mean'], 'F1_stability': f1_stab,
                      'F1_effect_ok': effect_ok, 'F1_direction_ok': direction_ok, 'cumple': cumple,
                      'min_r_rb_reference': min_rrb, 'plateau': bool(min_rrb < 0.10),
                      'F2': status['F2'], 'F3': status['F3'], 'F4': status['F4'],
                      'n_informative_F2_F4': n_inf, 'n_informative_other_region': n_other,
                      'normalization_label': norm_label, 'verdict_v13': verdict,
                      'conservation_B_fixed': cons_f, 'conservation_B_recomputed': cons_r,
                      'class_balance_sensitive': bool(cons_f < 0.90 or cons_r < 0.90)})
    log(f"3: {control} ({reference_label}): " + '; '.join(
        f"{r['level']}: pico {r['F1_peak_block']}, estab. {r['F1_stability']:.3f}, cumple {r['cumple']}" for r in crows))
    return {k: pd.DataFrame(v) for k, v in
            {'profiles': prof, 'regions': regs, 'contrasts': cts, 'conditions': crows, 'composition': comp,
             'f4_seeds': seeds_rows}.items()}


# ----------------------------------------------------------------------------- 4. CKA
def cka_bootstrap(orig: list, trans: dict, W: np.ndarray):
    """CKA lineal centrado; HSIC ponderado por multiplicidades bootstrap (misma fórmula que el paquete v2)."""
    n = W.shape[1]
    Wt = W.T.copy()
    ones = np.ones((n, 1))
    rows, boots = [], {}

    def hsic(Mw, Pw, MPw, w):
        r, q = Mw / n, Pw / n
        s, t = np.sum(w * r, 0) / n, np.sum(w * q, 0) / n
        return np.sum(w * MPw, 0) - 2 * n * np.sum(w * q * r, 0) + n * n * s * t

    for bi, b in enumerate(BLOCKS):
        mu = orig[bi].mean(axis=0)
        X = orig[bi] - mu
        Kx = X @ X.T
        Kw, KKw, Kw1, KKw1 = Kx @ Wt, (Kx * Kx) @ Wt, Kx @ ones, (Kx * Kx) @ ones
        hxx_b, hxx_1 = hsic(Kw, Kw, KKw, Wt), hsic(Kw1, Kw1, KKw1, ones)
        for cond, Y_all in trans.items():
            Y = Y_all[bi] - mu
            Ky = Y @ Y.T
            Lw, LLw, KLw = Ky @ Wt, (Ky * Ky) @ Wt, (Kx * Ky) @ Wt
            Lw1, LLw1, KLw1 = Ky @ ones, (Ky * Ky) @ ones, (Kx * Ky) @ ones
            cb = hsic(Kw, Lw, KLw, Wt) / np.sqrt(hxx_b * hsic(Lw, Lw, LLw, Wt))
            c1 = float((hsic(Kw1, Lw1, KLw1, ones) / np.sqrt(hxx_1 * hsic(Lw1, Lw1, LLw1, ones)))[0])
            boots[(cond, bi)] = cb
            rows.append({'condition': cond, 'block': b, 'cka': c1, 'ci_low': float(np.percentile(cb, 2.5)),
                         'ci_high': float(np.percentile(cb, 97.5))})
    return pd.DataFrame(rows), boots


def cka_ratio(cka: pd.DataFrame, boots: dict, variant: str) -> pd.DataFrame:
    """C4 para la variante (R1 o R2) y razón q = D_variante / D_R0 con las mismas réplicas."""
    out = []
    for lv in LEVELS:
        c0, cv = f'rotation_{lv}', f'rot{variant}_{lv}'
        pt = cka.set_index(['condition', 'block']).cka
        D0, Dv = pt[(c0, 'Block1')] - pt[(c0, 'Block5')], pt[(cv, 'Block1')] - pt[(cv, 'Block5')]
        D0b = boots[(c0, 0)] - boots[(c0, 4)]
        Dvb = boots[(cv, 0)] - boots[(cv, 4)]
        lo_v, hi_v = np.percentile(Dvb, [2.5, 97.5])
        valid = D0b > 0
        qb = Dvb[valid] / D0b[valid]
        q_lo, q_hi = (np.percentile(qb, [2.5, 97.5]) if valid.any() else (np.nan, np.nan))
        mono = bool(np.all(np.diff([pt[(cv, b)] for b in BLOCKS]) <= 0))
        out.append({'variant': variant, 'level': lv, 'angle': ANGLE[lv], 'D_R0': D0, f'D_{variant}': Dv,
                    'D_ci_low': lo_v, 'D_ci_high': hi_v, 'threshold_ok': bool(Dv >= 0.02 and lo_v > 0),
                    'monotone_nonincreasing_point': mono, 'q': Dv / D0 if D0 > 0 else np.nan,
                    'q_ci_low': q_lo, 'q_ci_high': q_hi, 'n_boot_valid': int(valid.sum()), 'n_boot': len(D0b)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- 5. sparsity
def sparsity_analysis(data: dict, A_pub: dict, IDX, log):
    desc, corr, assoc = [], [], []
    for name in SET_NAMES:
        s, sn = data[name]['s_exact'].astype(np.float64), data[name]['s_near'].astype(np.float64)
        for bi, b in enumerate(BLOCKS):
            z = (data[name]['gap'][bi] == 0).mean(1)
            q1, q2, q3 = np.percentile(s[:, bi], [25, 50, 75])
            desc.append({'set': name, 'block': b, 's_mean': s[:, bi].mean(), 's_q1': q1, 's_median': q2, 's_q3': q3,
                         's_near_mean': np.nanmean(sn[:, bi]), 'gap_zero_channel_share_mean': z.mean()})
            r = stats.spearmanr(s[:, bi], z).statistic if np.ptp(s[:, bi]) > 0 and np.ptp(z) > 0 else np.nan
            corr.append({'set': name, 'block': b, 'rho_s_vs_gap_zero_share': r})
    for cond in TRANSFORMED:
        s0 = data[route_original(cond)]['s_exact'].astype(np.float64)
        st = data[cond]['s_exact'].astype(np.float64)
        n0 = data[route_original(cond)]['s_near'].astype(np.float64)
        nt = data[cond]['s_near'].astype(np.float64)
        for bi, b in enumerate(BLOCKS):
            a = A_pub[cond][:, bi]
            ds = np.abs(st[:, bi] - s0[:, bi])
            rho, lo, hi, nv = boot_spearman(a, ds, IDX)
            informative = bool(np.mean(ds == 0) < 0.90 and nv >= 0.95 * len(IDX))
            sd = st[:, bi] - s0[:, bi]
            dn = np.abs(nt[:, bi] - n0[:, bi])
            assoc.append({'condition': cond, 'block': b, 'mean_abs_ds': ds.mean(), 'share_images_ds_zero': np.mean(ds == 0),
                          'rho_abs': rho, 'ci_low': lo, 'ci_high': hi, 'n_boot_valid': nv, 'informative': informative,
                          'relevant': bool(informative and not np.isnan(rho) and abs(rho) >= 0.30
                                           and (lo > 0 or hi < 0)),
                          'magnitude': band(rho),
                          'rho_signed_descriptive': float(stats.spearmanr(a, sd).statistic) if np.ptp(sd) > 0 else np.nan,
                          'rho_abs_near_zero_descriptive': (float(stats.spearmanr(a, dn).statistic)
                                                            if np.ptp(dn) > 0 and np.isfinite(dn).all() else np.nan)})
    assoc = pd.DataFrame(assoc)
    log(f'5: sparsity: {int(assoc.relevant.sum())} asociaciones relevantes de {len(assoc)} '
        f'({int(assoc.informative.sum())} informativas).')
    return pd.DataFrame(desc), pd.DataFrame(corr), assoc


# ----------------------------------------------------------------------------- 6. cabeza ImageNet
def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def head_metrics(lo_o: np.ndarray, lo_t: np.ndarray) -> dict:
    """Métricas por imagen a partir de logits float32 (softmax a temperatura 1 en float64)."""
    lpo, lpt = log_softmax(lo_o), log_softmax(lo_t)
    po, pt = np.exp(lpo), np.exp(lpt)
    ko, kt = lo_o.argmax(1), lo_t.argmax(1)
    t5o, t5t = np.argsort(-lo_o, 1)[:, :5], np.argsort(-lo_t, 1)[:, :5]
    overlap = (t5o[:, :, None] == t5t[:, None, :]).any(2).sum(1) / 5
    r = np.arange(len(ko))
    return {'top1_change': ko != kt, 'KL': np.sum(po * (lpo - lpt), 1),
            'delta_p_top1_original': pt[r, ko] - po[r, ko], 'confidence_change': pt.max(1) - po.max(1),
            'top5_overlap': overlap}


def imagenet_head(data: dict, A_F1: dict, IDX, W, log):
    rows, blocks = [], []
    per_cond = {}
    for cond in LOGIT_CONDITIONS:
        m = head_metrics(data[route_original(cond)]['logits'], data[cond]['logits'])
        per_cond[cond] = m
        ch = m['top1_change'].astype(np.float64)
        a5 = A_F1[cond][:, 4]
        pb = W @ ch / len(ch)
        klb = np.median(m['KL'][IDX], axis=1)
        rho, lo, hi, nv = boot_spearman(a5, m['KL'], IDX)
        row = {'condition': cond, 'n_top1_changed': int(ch.sum()), 'prop_top1_change': ch.mean(),
               'prop_ci_low': np.percentile(pb, 2.5), 'prop_ci_high': np.percentile(pb, 97.5),
               'median_KL': float(np.median(m['KL'])), 'median_KL_ci_low': np.percentile(klb, 2.5),
               'median_KL_ci_high': np.percentile(klb, 97.5),
               'median_delta_p_top1_original': float(np.median(m['delta_p_top1_original'])),
               'median_confidence_change': float(np.median(m['confidence_change'])),
               'mean_top5_overlap': float(m['top5_overlap'].mean()),
               'mean_A_adj_B5': float(a5.mean()), 'rho_A_adj_B5_vs_KL': rho, 'rho_ci_low': lo, 'rho_ci_high': hi,
               'rho_n_boot_valid': nv, 'rho_magnitude': band(rho)}
        g1, g0 = a5[m['top1_change']], a5[~m['top1_change']]
        if len(g1) == 0 or len(g0) == 0:
            row.update({'mw_applicable': False, 'r_rb_changed_vs_unchanged': np.nan, 'mw_p': np.nan})
        else:
            r, p = indep_rank_biserial(g1, g0)
            row.update({'mw_applicable': True, 'r_rb_changed_vs_unchanged': r, 'mw_p': p})
        row['r_rb_magnitude'] = band(row['r_rb_changed_vs_unchanged'])
        rows.append(row)
        for bi, b in enumerate(BLOCKS):
            a = A_F1[cond][:, bi]
            blocks.append({'condition': cond, 'block': b,
                           'rho_A_adj_vs_KL': float(stats.spearmanr(a, m['KL']).statistic),
                           'rho_magnitude': band(float(stats.spearmanr(a, m['KL']).statistic))})
    head = pd.DataFrame(rows)
    head['mw_p_holm'] = holm(head.mw_p.to_numpy())
    # Exploratorio: Spearman entre condiciones (30), media de A_adj B5 frente a proporción de cambio top-1.
    A5 = np.stack([A_F1[c][:, 4] for c in LOGIT_CONDITIONS], 1)
    CH = np.stack([per_cond[c]['top1_change'].astype(np.float64) for c in LOGIT_CONDITIONS], 1)
    n = A5.shape[0]
    mA, mC = W @ A5 / n, W @ CH / n
    rho_pt = float(stats.spearmanr(A5.mean(0), CH.mean(0)).statistic)
    rb = np.array([stats.spearmanr(mA[r], mC[r]).statistic for r in range(len(W))])
    rb = rb[np.isfinite(rb)]
    across = pd.DataFrame([{'n_conditions': len(LOGIT_CONDITIONS), 'rho_mean_A_adj_B5_vs_prop_top1_change': rho_pt,
                            'ci_low': np.percentile(rb, 2.5), 'ci_high': np.percentile(rb, 97.5),
                            'n_boot_valid': len(rb), 'magnitude': band(rho_pt),
                            'note': 'exploratorio; no decide ninguna interpretación'}])
    log(f'6: cabeza ImageNet: {int(head.mw_applicable.sum())}/30 comparaciones de Mann–Whitney aplicables; '
        f'Spearman entre condiciones = {rho_pt:.3f} (exploratorio).')
    return head, pd.DataFrame(blocks), across


# ----------------------------------------------------------------------------- 7. redacción
def interpretations(cond_tables: dict, gamma_ref: pd.DataFrame, c4: pd.DataFrame, r2_eq: bool, sparsity: pd.DataFrame,
                    head: pd.DataFrame, gamma_sat: dict) -> pd.DataFrame:
    out = []

    def add(control, decision, text, detail=''):
        out.append({'control': control, 'decision': decision, 'text': text, 'detail': detail})

    def transversal(df: pd.DataFrame) -> str:
        parts = []
        for r in df.itertuples():
            lab = r.normalization_label
            if lab.startswith('evidencia insuficiente') and r.n_informative_other_region > 0:
                lab += ' (con discordancia de región)'
            bal = 'sensible al balance de clases' if r.class_balance_sensitive else 'estable ante el balance de clases'
            mes = '; meseta (se formula solo por región)' if getattr(r, 'plateau', False) else ''
            parts.append(f'{r.level}: {lab}; {bal} (conservación {r.conservation_B_fixed:.2f} con B fijo, '
                         f'{r.conservation_B_recomputed:.2f} con B recalculado){mes}')
        return ' | '.join(parts)

    # C1 con gamma (B_float principal)
    g = cond_tables['gamma']
    ok = g[g.cumple.astype(bool)].level.tolist()
    # «sin saturación nueva» solo si la proporción máxima por imagen es 0 por píxel y por componente en esos niveles
    no_sat = all(gamma_sat[lv]['pixel_max'] == 0 and gamma_sat[lv]['component_max'] == 0 for lv in (ok or LEVELS))
    sat_txt = ('sin saturación nueva' if no_sat else 'con una saturación nueva máxima por imagen de '
               + ', '.join(f"{lv}: {gamma_sat[lv]['pixel_max']:.2g} de los píxeles" for lv in (ok or LEVELS)))
    if len(ok) == 3:
        add('C1 (gamma)', 'se conserva', 'El brillo actúa en la región superficial, no solo por el clipping: la respuesta '
            f'superficial persiste con gamma, {sat_txt}, en los tres niveles.')
    elif ok:
        add('C1 (gamma)', 'se limita', f'La respuesta superficial persiste con gamma, {sat_txt}, solo en los '
            f'niveles {", ".join(ok)}; en los demás no se cumplen los criterios de la v1.3.')
    else:
        add('C1 (gamma)', 'se restringe', 'C1 se formula solo para el brillo multiplicativo con clipping; el clipping '
            'puede explicar la respuesta superficial.')
    peaks = '; '.join(f'{r.level}: {r.F1_peak_block}' for r in g.itertuples())
    out[-1]['detail'] = (f'Bloque pico con gamma: {peaks}. Saturación nueva (media/máxima por imagen; píxel | '
                         f'componente): ' + '; '.join(f"{lv}: {gamma_sat[lv]['pixel_mean']:.2g}/"
                                                      f"{gamma_sat[lv]['pixel_max']:.2g} | "
                                                      f"{gamma_sat[lv]['component_mean']:.2g}/"
                                                      f"{gamma_sat[lv]['component_max']:.2g}" for lv in LEVELS)
                         + f'. Transversal: {transversal(g)}')
    same = bool(gamma_ref.same_region.all() and gamma_ref.same_cumple.all() and gamma_ref.same_verdict.all())
    add('C1 (gamma): referencia', 'no cambia las conclusiones' if same else 'cambia las conclusiones',
        'La elección de referencia (B_float frente a B del manuscrito) no cambia las conclusiones: misma región pico y '
        'mismo veredicto en los tres niveles.' if same else
        'Con B(b) del manuscrito la región pico o el veredicto cambian en algún nivel; la conclusión de gamma se '
        'informa como dependiente de la referencia.',
        '; '.join(f'{r.level}: región {r.region_B_float}/{r.region_B_manuscript}, cumple {r.cumple_B_float}/'
                  f'{r.cumple_B_manuscript}' for r in gamma_ref.itertuples()))
    # C3 con R1
    r1 = cond_tables['R1'].set_index('level')
    if r1.loc['medium', 'cumple'] and r1.loc['high', 'cumple']:
        add('C3 (R1)', 'se conserva', 'La rotación actúa en la región intermedia; el efecto es compatible con la rotación '
            'sin bordes negros, no exclusivo de la orientación.')
    else:
        add('C3 (R1)', 'se restringe', 'C3 se formula como «rotación con relleno cero».')
    out[-1]['detail'] = (f"10°: cumple {bool(r1.loc['low', 'cumple'])} (informativo); 30°: {bool(r1.loc['medium', 'cumple'])}; "
                         f"60°: {bool(r1.loc['high', 'cumple'])}. Transversal: {transversal(cond_tables['R1'])}")
    if not r2_eq and 'R2' in cond_tables:
        add('R2 (no equivalente a R0)', 'informativo', 'R2 no es operacionalmente equivalente a R0 y se analiza como '
            'R1, con su propia familia de Holm.', transversal(cond_tables['R2']))
    # C4
    c = c4[c4.variant == 'R1']
    if not c.threshold_ok.all():
        add('C4 (R1)', 'se restringe', 'C4 se formula como «rotación con relleno cero».')
    elif (c.q_ci_low >= 0.50).all():
        add('C4 (R1)', 'se conserva', 'El CKA bajo rotación disminuye con la profundidad también sin bordes negros.')
    elif (c.q_ci_high < 0.50).any():
        add('C4 (R1)', 'se limita', 'El descenso del CKA con la profundidad se mantiene sin bordes negros, pero los '
            'bordes explican parte del descenso.')
    else:
        add('C4 (R1)', 'incierto', 'El descenso del CKA se mantiene sin bordes negros, pero el IC de la razón de '
            'descensos cruza 0,50: el resultado es incierto.')
    out[-1]['detail'] = '; '.join(f'{r.angle}°: D={r.D_R1:.3f} [{r.D_ci_low:.3f}, {r.D_ci_high:.3f}], q={r.q:.2f} '
                                  f'[{r.q_ci_low:.2f}, {r.q_ci_high:.2f}], réplicas válidas {r.n_boot_valid}'
                                  for r in c.itertuples())
    # Zoom
    z1, z1r, z2 = (cond_tables[v].set_index('level') for v in ['Z1', 'Z1r', 'Z2'])
    n_ok = int(z1.cumple.sum())
    same_reg = all(z1.loc[lv, 'F1_peak_region'] == z1r.loc[lv, 'F1_peak_region'] for lv in LEVELS)
    if n_ok == 3 and same_reg:
        add('Zoom: mecanismo', 'se conserva', 'El efecto profundo del zoom es compatible con la pérdida de contenido '
            'periférico, sin afirmar causalidad.')
    elif n_ok >= 1 or (n_ok == 3 and not same_reg):
        add('Zoom: mecanismo', 'se limita', f'Z1 cumple en {n_ok} de 3 factores'
            + ('' if same_reg else ' y Z1 y Z1r discrepan en la región pico') + '; el mecanismo se formula con cautela.')
    else:
        z2ok = [lv for lv in LEVELS if z2.loc[lv, 'cumple']]
        add('Zoom: mecanismo', 'se elimina el lenguaje mecanicista', 'El mecanismo específico del efecto profundo del '
            'zoom no queda identificado.' + (f' Como descriptivo: el remuestreo por sí solo (Z2) reproduce el aumento '
                                             f'profundo en {", ".join(z2ok)}.' if z2ok else ''))
    out[-1]['detail'] = ('; '.join(f'{lv}: Z1 {bool(z1.loc[lv, "cumple"])} ({z1.loc[lv, "F1_peak_region"]}), '
                                   f'Z1r {z1r.loc[lv, "F1_peak_region"]}, Z2 {bool(z2.loc[lv, "cumple"])}' for lv in LEVELS)
                         + f'. Z1 y Z1r coinciden en la región: {same_reg}. Transversal Z1: '
                         + transversal(cond_tables['Z1']))
    # Bloque 5 y cabeza ImageNet
    sep = head[head.rho_magnitude.isin(['despreciable', 'pequeña'])].condition.tolist()
    add('Bloque 5 y cabeza ImageNet', 'descriptivo (sin veredicto)',
        'La sensibilidad representacional no equivale a vulnerabilidad de la tarea, y la cabeza ImageNet no mide '
        'rendimiento en TrashNet; se habla de «cambio en la salida de la cabeza ImageNet», nunca de accuracy. '
        + (f'En {len(sep)} condiciones la asociación por imagen entre A_adj del bloque 5 y KL es pequeña o despreciable, '
           'y en ellas los dos fenómenos son separables.' if sep else ''),
        '; '.join(f'{r.condition}: ρ={r.rho_A_adj_B5_vs_KL:.2f} ({r.rho_magnitude}), r_rb='
                  f'{r.r_rb_changed_vs_unchanged:.2f} ({r.r_rb_magnitude})' for r in head.itertuples()))
    # Sparsity
    rel = sparsity[sparsity.relevant]
    if len(rel):
        add('Sparsity antes del GAP', 'asociación en algunos bloques', 'Donde la asociación es relevante, el cambio de '
            'sparsity se asocia con la sensibilidad en ese bloque; no se formula como causa.',
            '; '.join(f'{r.condition} {r.block}: ρ={r.rho_abs:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]'
                      for r in rel.itertuples()))
    else:
        add('Sparsity antes del GAP', 'sin asociación relevante', 'No se encontró evidencia de que los cambios de '
            'sparsity siguieran los perfiles de sensibilidad.')
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--extraction', required=True, type=Path, help='Carpeta Revision_Adenda1_extraccion_v2')
    ap.add_argument('--v2-package', required=True, type=Path, help='Carpeta run_v2 del paquete de robustez v2')
    ap.add_argument('--repo', required=True, type=Path, help='Repositorio en el commit fijado')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--test-fixture', action='store_true',
                    help='Solo para las pruebas con datos sintéticos: la identidad de la extracción y del paquete v2 '
                         'se informa pero no detiene. Las salidas quedan marcadas como no válidas.')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        print('ERROR: la carpeta de salida no está vacía.', flush=True)
        sys.exit(6)
    log = Logger(out / 'run_log.txt')
    script = Path(__file__).resolve()
    log(f'Script {script.name}, SHA-256 {sha256(script)}')
    log(f'Adenda 1 v1.2 SHA-256 {ADDENDUM_SHA256}; criterios v1.3 SHA-256 {CRITERIA_SHA256}')
    if args.test_fixture:
        log('*** MODO DE PRUEBA CON DATOS SINTÉTICOS: las salidas NO son válidas para el manuscrito. ***')

    # ---- 0. validación
    checks = validate_checksums(args.extraction, log, args.test_fixture)
    fpinfo = validate_fingerprint(args.extraction, log, args.test_fixture)
    repo_hashes = validate_repo_files(args.repo, log, args.test_fixture)
    index = pd.read_csv(args.repo / 'data/manifests/original_image_index.csv')
    n = len(index)
    if not args.test_fixture and n != N_EXPECTED:
        fail(log, 3, 'el índice no tiene 2527 imágenes')
    data, cal, gammas = load_sets(args.extraction, fpinfo['fingerprint_id'], n, log)
    IDX, W, draws, k_draw, frozen = load_frozen(args.v2_package, index, log, args.test_fixture)
    labels = index['class'].to_numpy()
    pi, pj = reference_pairs(labels)
    pub = np.load(args.repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz')
    B_pub = np.array([float(np.median(pub[f'{b}_random'].astype(np.float64))) for b in BLOCKS])
    err = max(float(np.max(np.abs(np.sum(normalize(data['original_uint8']['gap'][bi].astype(np.float64))[pi]
                                         * normalize(data['original_uint8']['gap'][bi].astype(np.float64))[pj], 1)
                                  - pub[f'{b}_random']))) for bi, b in enumerate(BLOCKS))
    checks['reference_pairs_new_uint8_vs_published_max_abs_err'] = err
    checks['reference_pairs_tolerance'] = TOL_REFERENCE_PAIRS
    if not err <= TOL_REFERENCE_PAIRS:
        fail(log, 3, f'los pares de referencia recalculados con los originales nuevos difieren de los publicados en '
                     f'{err:.2e} (> {TOL_REFERENCE_PAIRS:.0e})')
    log(f"0: B(b) del manuscrito = {', '.join(f'{v:.4f}' for v in B_pub)}; pares de referencia con los originales "
        f'nuevos frente a los publicados: error máximo {err:.2e} (≤ {TOL_REFERENCE_PAIRS:.0e}).')
    num = index.original_filename.str.extract(r'(\d+)')[0].astype(int).to_numpy()
    first_k = np.flatnonzero(num <= k_draw)
    if not args.test_fixture and len(first_k) != 822:
        fail(log, 5, 'el subconjunto secuencial 1–137 no tiene 822 imágenes')

    # ---- 1. duplicados
    dup, dupinfo = duplicates_finding(args.extraction, index, data, draws, pi, pj, log)

    # ---- 2. diagnósticos
    diag, route, r2tab, r2_eq, sat = diagnostics(args.extraction, data, cal, pi, pj, log)
    no_new_saturation = all(v['pixel_max'] == 0 and v['component_max'] == 0 for v in sat.values())
    B_float = route.B_float.to_numpy()

    # ---- 3. controles de región
    gap64 = {name: [g.astype(np.float64) for g in data[name]['gap']] for name in SET_NAMES}
    G = {r: [normalize(gap64[r][bi]) @ normalize(gap64[r][bi]).T for bi in range(5)]
         for r in ['original_uint8', 'original_float']}
    controls = {'gamma': ([f'gamma_{lv}' for lv in LEVELS], 'original_float', B_float),
                'R1': ([f'rotR1_{lv}' for lv in LEVELS], 'original_uint8', B_pub),
                'Z1': ([f'zoomZ1_{lv}' for lv in LEVELS], 'original_uint8', B_pub),
                'Z1r': ([f'zoomZ1r_{lv}' for lv in LEVELS], 'original_uint8', B_pub),
                'Z2': ([f'zoomZ2_{lv}' for lv in LEVELS], 'original_uint8', B_pub)}
    if not r2_eq:
        controls['R2'] = ([f'rotR2_{lv}' for lv in LEVELS], 'original_uint8', B_pub)
    res = {}
    for ctl, (conds, o, B) in controls.items():
        res[ctl] = region_control(ctl, conds, gap64[o], {c: gap64[c] for c in conds}, B, pi, pj, IDX, W, G[o],
                                  draws, first_k, log)
    res_gamma_pub = region_control('gamma', controls['gamma'][0], gap64['original_float'],
                                   {c: gap64[c] for c in controls['gamma'][0]}, B_pub, pi, pj, IDX, W,
                                   G['original_float'], draws, first_k, log, reference_label='B_manuscrito')
    cat = lambda key: pd.concat([res[c][key] for c in res] + [res_gamma_pub[key]], ignore_index=True)
    profiles, regions, contrasts, conds_tab, comp, f4seeds = (
        cat(k) for k in ['profiles', 'regions', 'contrasts', 'conditions', 'composition', 'f4_seeds'])
    fam_of = contrasts.control.map({'gamma': 'gamma', 'R1': 'R1', 'R2': 'R2', 'Z1': 'zoom', 'Z1r': 'zoom', 'Z2': 'zoom'})
    contrasts['holm_family'] = np.where(contrasts.reference == 'principal', fam_of, 'gamma_B_manuscrito (sensibilidad)')
    contrasts['p_holm'] = np.nan
    for fam_name, idx in contrasts.groupby('holm_family').groups.items():
        if fam_name.startswith('gamma_B_manuscrito'):
            continue
        contrasts.loc[idx, 'p_holm'] = holm(contrasts.loc[idx, 'wilcoxon_p'].to_numpy())
    sizes = contrasts[contrasts.reference == 'principal'].groupby('holm_family').size().to_dict()
    log(f'3: familias de Holm: {sizes}')
    gp, gb = res['gamma']['conditions'].set_index('level'), res_gamma_pub['conditions'].set_index('level')
    gamma_ref = pd.DataFrame([{'level': lv, 'region_B_float': gp.loc[lv, 'F1_peak_region'],
                               'region_B_manuscript': gb.loc[lv, 'F1_peak_region'],
                               'cumple_B_float': bool(gp.loc[lv, 'cumple']), 'cumple_B_manuscript': bool(gb.loc[lv, 'cumple']),
                               'verdict_B_float': gp.loc[lv, 'verdict_v13'], 'verdict_B_manuscript': gb.loc[lv, 'verdict_v13']}
                              for lv in LEVELS])
    gamma_ref['same_region'] = gamma_ref.region_B_float == gamma_ref.region_B_manuscript
    gamma_ref['same_cumple'] = gamma_ref.cumple_B_float == gamma_ref.cumple_B_manuscript
    gamma_ref['same_verdict'] = gamma_ref.verdict_B_float == gamma_ref.verdict_B_manuscript

    # ---- 4. CKA
    rot_conds = [f'rotation_{lv}' for lv in LEVELS] + [f'rotR1_{lv}' for lv in LEVELS]
    if not r2_eq:
        rot_conds += [f'rotR2_{lv}' for lv in LEVELS]
    cka, cboots = cka_bootstrap(gap64['original_uint8'], {c: gap64[c] for c in rot_conds}, W)
    pubcka = pd.read_csv(args.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv')
    r0 = cka[cka.condition.str.startswith('rotation_')].assign(level=lambda d: d.condition.str.split('_').str[1])
    mm = r0.merge(pubcka[pubcka.transformation == 'rotation'][['level', 'block', 'linear_CKA']], on=['level', 'block'])
    checks['CKA_R0_new_vs_published_max_abs_err'] = float(np.max(np.abs(mm.cka - mm.linear_CKA))) if len(mm) else np.nan
    c4 = pd.concat([cka_ratio(cka, cboots, v) for v in (['R1'] if r2_eq else ['R1', 'R2'])], ignore_index=True)
    log('4: CKA: ' + '; '.join(f'{r.angle}°: D_R1={r.D_R1:.3f}, q={r.q:.2f} [{r.q_ci_low:.2f}, {r.q_ci_high:.2f}], '
                               f'válidas {r.n_boot_valid}' for r in c4[c4.variant == 'R1'].itertuples())
        + f"; CKA R0 frente a la tabla publicada: error máximo {checks['CKA_R0_new_vs_published_max_abs_err']:.2e}")

    # ---- 5. sparsity (A_adj con B del manuscrito; Spearman no depende de B dentro de un bloque)
    A_pub, A_F1 = {}, {}
    for cond in TRANSFORMED:
        o = route_original(cond)
        S = np.stack([cos_rows(gap64[o][bi], gap64[cond][bi]) for bi in range(5)], 1)
        A_pub[cond] = (1 - S) / (1 - B_pub)
        A_F1[cond] = (1 - S) / (1 - (B_float if cond.startswith('gamma') else B_pub))
    sp_desc, sp_corr, sp_assoc = sparsity_analysis(data, A_pub, IDX, log)

    # ---- 6. cabeza ImageNet
    head, head_blocks, head_across = imagenet_head(data, A_F1, IDX, W, log)

    # ---- 7. redacción
    cond_tables = {c: res[c]['conditions'] for c in res}
    interp = interpretations(cond_tables, gamma_ref, c4, r2_eq, sp_assoc, head, sat)
    for r in interp.itertuples():
        log(f'7: {r.control}: {r.decision}.')

    # ---- guardado
    tables = {
        '01_hallazgo_duplicados.csv': dup,
        '02_diagnosticos_extraccion.csv': diag,
        '03_B_float_y_equivalencia_rutas.csv': route,
        '04_equivalencia_R2_R0.csv': r2tab,
        '05_perfiles_por_familia.csv': profiles,
        '06_regiones_estabilidad.csv': regions,
        '06b_F4_regiones_por_semilla.csv': f4seeds,
        '07_contrastes_referencia_F1.csv': contrasts,
        '08_condiciones_criterios.csv': conds_tab,
        '09_composicion_clases.csv': comp,
        '10_gamma_B_float_frente_a_B_manuscrito.csv': gamma_ref,
        '11_cka_por_bloque.csv': cka,
        '12_C4_y_razon_q.csv': c4,
        '13_sparsity_descriptivo.csv': sp_desc,
        '14_sparsity_s_frente_a_canales_cero.csv': sp_corr,
        '15_sparsity_asociacion.csv': sp_assoc,
        '16_cabeza_imagenet_por_condicion.csv': head,
        '17_cabeza_imagenet_por_bloque.csv': head_blocks,
        '18_cabeza_imagenet_entre_condiciones.csv': head_across,
        '19_redaccion_interpretaciones.csv': interp,
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format='%.6g')
    md = ['# Redacción resultante (Adenda 1 v1.2)', '']
    if args.test_fixture:
        md += ['**DATOS SINTÉTICOS DE PRUEBA: no válido para el manuscrito.**', '']
    for r in interp.itertuples():
        md += [f'## {r.control}: {r.decision}', '', r.text, '', f'Detalle: {r.detail}' if r.detail else '', '']
    md += ['## Hallazgo post-extracción: imágenes duplicadas entre clases', '',
           f"{dupinfo['n_cross_class']} pares de imágenes idénticas en clases distintas (tabla 01). Se registran como "
           'hallazgo posterior a la extracción y no se excluyen del análisis principal.', '']
    (out / '19_redaccion_interpretaciones.md').write_text('\n'.join(md), encoding='utf-8')

    inputs = {'extraction/H_checksums.sha256': sha256(args.extraction / 'H_checksums.sha256'),
              'extraction/run_fingerprint.json': sha256(args.extraction / 'run_fingerprint.json'),
              'v2/bootstrap_indices.npy': frozen['bootstrap_indices_sha256'],
              'v2/10_extracciones_balanceadas_ids.csv': frozen['balanced_draws_sha256'],
              'repo/original_image_index.csv': sha256(args.repo / 'data/manifests/original_image_index.csv'),
              'repo/baseline_pair_similarity_samples.npz':
                  sha256(args.repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz'),
              'repo/05_CKA_lineal_centrado_por_condicion.csv':
                  sha256(args.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv')}
    manifest = {
        'addendum_sha256': ADDENDUM_SHA256, 'criteria_sha256': CRITERIA_SHA256,
        'script': script.name, 'script_sha256': sha256(script),
        'test_fixture': args.test_fixture, 'valid_for_manuscript': not args.test_fixture,
        'extraction_fingerprint_id': fpinfo['fingerprint_id'], 'identity_checks': fpinfo['identity_checks'],
        'n_boot': int(IDX.shape[0]), 'n_balanced_draws': int(len(draws)), 'per_class_draw': k_draw,
        'B_manuscript': dict(zip(BLOCKS, B_pub)), 'B_float': dict(zip(BLOCKS, B_float)),
        'gammas': gammas, 'R2_equivalent_to_R0': r2_eq, 'gamma_newly_saturated': sat,
        'gamma_no_new_saturation': no_new_saturation, 'H_checksums_sha256_expected': H_CHECKSUMS_SHA256,
        'repo_files_sha256': repo_hashes,
        'holm_family_sizes': sizes, 'duplicates_post_extraction': dupinfo,
        'checks': checks, 'inputs_sha256': inputs,
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__},
        'interpretations_and_deviations': [
            'I1–I2 de la v2 se mantienen: «diferencia pareada» = mediana de las diferencias por imagen; estabilidad y '
            'región de F2–F4 se juzgan respecto de la región esperada de la afirmación.',
            'I3: los rótulos de normalización usan la región pico puntual de cada familia informativa '
            '(«familias F2–F4 informativas que dan otra región pico»); el estado concuerda/discrepa de la v1.3 '
            '(estabilidad ≥ 0,70, 16/20 semillas, signo) se informa aparte en la tabla 08.',
            'I4: en sparsity, A_adj usa B(b) del manuscrito incluso para gamma (texto literal de la adenda); en la '
            'cabeza ImageNet, A_adj del bloque 5 usa la referencia principal de cada condición (B_float para gamma). '
            'Los Spearman y Mann–Whitney por condición no dependen de B, porque B es constante dentro del bloque; '
            'solo el Spearman exploratorio entre condiciones depende de ello.',
            'I5: para gamma, s_0, los logits de referencia y F2–F4 usan los originales de la ruta float; para el '
            'resto, los de la ruta uint8.',
            'I6: «Z1r da la misma región» se evalúa como igualdad de la región pico puntual de F1 de Z1 y Z1r en '
            'cada factor.',
            'I7: la composición de clases recalcula B con todos los pares únicos de cada extracción usando los '
            'originales de la misma ruta que la condición.',
            'H1 (hallazgo post-extracción, no preespecificado): tres pares de imágenes idénticas en clases distintas. '
            'Se registran en la tabla 01 y no se excluyen del análisis principal.',
        ],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    log('8: registro cerrado. A continuación se calculan los checksums de salida.')
    with open(out / 'checksums_outputs.sha256', 'w') as f:
        for p in sorted(out.iterdir()):
            if p.name != 'checksums_outputs.sha256':
                f.write(f'{sha256(p)}  {p.name}\n')


if __name__ == '__main__':
    main()
