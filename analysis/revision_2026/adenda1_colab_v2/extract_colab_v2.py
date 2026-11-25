"""Extracción nueva para la revisión jimaging-4632369, según la Adenda 1 v1.2 congelada.

Adenda 1 v1.2 (.docx), SHA-256 8f0c7b19079c196f2ad6220340b47adb64b7d775c32d974e8d2a6145decda156
Criterios base v1.3 (.docx), SHA-256 492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57

Versión 2 del script. Corrige la implementación tras la auditoría de ChatGPT; la adenda no cambia.
  1. Huella de ejecución (run_fingerprint.json). Nada se reutiliza si no coinciden el script, el repositorio,
     las imágenes, los NPZ de referencia, el modo (prueba reducida o completa), los parámetros, el entorno,
     gamma y los umbrales. Una carpeta con otra huella, o no vacía y sin huella, se rechaza.
  2. Repositorio fijado al commit REPO_COMMIT, con el SHA-256 de generate_transformations_png.py comprobado.
  3. Pruebas locales reproducibles desde el paquete (tests/, con sus sustitutos de torch y torchvision).
  4. Sin matriz «extra»: cada medida auxiliar se guarda con nombre propio, y el esquema va en el manifiesto.

Solo extrae y guarda; no analiza. Etapas, en orden:
  0  Repositorio fijado, entorno, pesos y huella de ejecución
  A  SHA-256 y tamaño de los JPEG de entrada
  B  Control de píxeles: 12 condiciones existentes regeneradas frente a los PNG almacenados
  C  Comprobaciones geométricas de rotación (máscaras R0 y R1)
  D  Calibración de gamma por histogramas exactos
  E  Primera pasada sobre los originales: histogramas para los umbrales de casi cero
  F  Extracción de originales (uint8) y 12 condiciones existentes + control de embeddings
     (si falla, se detiene y no se extrae ninguna condición nueva)
  G  Extracción de los 22 conjuntos restantes; B_float y equivalencia R2–R0
  H  Manifiesto y checksums (después de cerrar el registro)

Códigos de salida: 2 control de píxeles, 3 control de embeddings, 4 carpeta de salida incompatible,
5 repositorio no fijado o modificado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ADDENDUM_SHA256 = '8f0c7b19079c196f2ad6220340b47adb64b7d775c32d974e8d2a6145decda156'
CRITERIA_SHA256 = '492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57'
REPO_COMMIT = '6672c2c7e4b84a5b0381de0d70476a97438223e5'
GEN_SCRIPT_SHA256 = '39f2eb0935d7914370bdff5e095c9d99b14b82016e883f8686aecce93788499b'
INDEX_CSV_SHA256 = '796f11e8cf3307068824bfe868902cff23cdeaf56a68628c1e50f8193b43257f'
SEED = 2026
W, H = 512, 384
N_EXPECTED = 2527
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LEVELS = ['low', 'medium', 'high']
BLOCKS = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5']
LAYER_INDEX = {'Block1': 3, 'Block2': 8, 'Block3': 15, 'Block4': 22, 'Block5': 29}
DIMS = {'Block1': 64, 'Block2': 128, 'Block3': 256, 'Block4': 512, 'Block5': 512}
EXISTING = {
    'brightness': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'contrast': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'rotation': {'low': 10, 'medium': 30, 'high': 60},
    'zoom': {'low': 1.10, 'medium': 1.25, 'high': 1.50},
}
ANGLES = EXISTING['rotation']
ZOOMS = EXISTING['zoom']
FILL_MEAN_RGB = (124, 116, 104)  # round(255 × [0.485, 0.456, 0.406])
PAD_R1 = 256
NBINS, LOG_LO, LOG_HI = 4096, -6.0, 4.0
LUMA = np.array([0.299, 0.587, 0.114])
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
WEIGHTS_FILE = 'vgg16-397923af.pth'
TOL_COS, TOL_NORM, TOL_AADJ = 1e-6, 1e-5, 1e-5  # control de embeddings (adenda, alcance)
N_PAIRS = 100_000
GAMMA_BRACKET, GAMMA_TOL = (0.05, 1.0), 1e-5

# Medidas auxiliares por imagen, con nombre propio (sustituyen a la matriz «extra» de la v1).
AUX_FIELDS = {
    'R2': ['r2_mae_R', 'r2_mae_G', 'r2_mae_B', 'r2_max_abs_diff', 'r2_pct_pixels_different'],
    'gamma': ['gamma_Y_mean_original', 'gamma_Y_mean_transformed', 'gamma_Y_std_original',
              'gamma_Y_std_transformed', 'gamma_newly_saturated_pixel_fraction',
              'gamma_newly_saturated_component_fraction'],
}
N_AUX = max(len(v) for v in AUX_FIELDS.values())

NPZ_SCHEMA = {
    'set_name': 'str: nombre del conjunto de extracción',
    'fingerprint_id': 'str: huella de ejecución con la que se generó (debe coincidir para reutilizarlo)',
    'thresholds_digest': 'str: SHA-256 de los umbrales de casi cero y la máscara de canales vivos usados',
    'gamma': 'float64 escalar: γ aplicado (NaN si el conjunto no es de gamma)',
    'row_index': 'int64 (n,): fila en data/manifests/original_image_index.csv (n = 2527 en la ejecución completa)',
    'gap_BlockK': 'float32 (n, C): GAP de la salida ReLU del bloque K',
    'czero_BlockK': 'float32 (n, C): c_k, fracción de posiciones espaciales exactamente cero del canal k',
    's_exact_BlockK': 'float32 (n,): s, fracción de activaciones exactamente cero en el tensor C × H × W',
    's_near_BlockK': ('float32 (n,): fracción de activaciones ≤ umbral de casi cero de su canal (1 % de la mediana '
                      'de no nulas en los originales), sobre los canales vivos; los ceros exactos cuentan'),
    'logits': 'float32 (n, 1000): salida completa de model(x)',
    'r2_mae_R / r2_mae_G / r2_mae_B': 'float32 (n,), solo R2: MAE por componente RGB frente a R0, escala 0–255',
    'r2_max_abs_diff': 'float32 (n,), solo R2: diferencia absoluta máxima frente a R0, escala 0–255',
    'r2_pct_pixels_different': 'float32 (n,), solo R2: % de píxeles con algún componente distinto de R0',
    'gamma_Y_mean_original / gamma_Y_mean_transformed': 'float32 (n,), solo gamma: luminancia media BT.601 en [0, 1]',
    'gamma_Y_std_original / gamma_Y_std_transformed': 'float32 (n,), solo gamma: desviación estándar de Y',
    'gamma_newly_saturated_pixel_fraction': ('float32 (n,), solo gamma: fracción de píxeles con algún componente '
                                             'que llega a 1 sin valer 1 antes (debe ser 0)'),
    'gamma_newly_saturated_component_fraction': 'float32 (n,), solo gamma: lo mismo por componente RGB (debe ser 0)',
}


# ============================================================================ utilidades
class Logger:
    """Registro con búfer: se escribe en disco solo después de validar la carpeta de salida."""

    def __init__(self):
        self.path, self.buffer, self.t0 = None, [], time.time()

    def attach(self, path: Path) -> None:
        self.path = path
        with open(path, 'a', encoding='utf-8') as f:
            f.writelines(line + '\n' for line in self.buffer)
        self.buffer = []

    def __call__(self, msg: str) -> None:
        line = f'[{time.time() - self.t0:8.1f} s] {msg}'
        print(line, flush=True)
        if self.path is None:
            self.buffer.append(line)
        else:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':'))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


# ============================================================================ repositorio fijado
_GEN = None  # módulo scripts/generate_transformations_png.py del repositorio


def repo_info(repo: Path) -> dict:
    def git(*a):
        r = subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    gen, idx = repo / 'scripts' / 'generate_transformations_png.py', repo / 'data/manifests/original_image_index.csv'
    return {'commit': git('rev-parse', 'HEAD'),
            'tracked_changes': git('status', '--porcelain', '--untracked-files=no'),
            'generate_transformations_png_sha256': sha256(gen),
            'original_image_index_sha256': sha256(idx)}


def repo_problems(info: dict) -> list[str]:
    p = []
    if info['commit'] != REPO_COMMIT:
        p.append(f"commit {info['commit']} ≠ {REPO_COMMIT}")
    if info['tracked_changes']:
        p.append('el repositorio tiene cambios sin confirmar en archivos versionados')
    if info['generate_transformations_png_sha256'] != GEN_SCRIPT_SHA256:
        p.append('SHA-256 de generate_transformations_png.py distinto del fijado')
    if info['original_image_index_sha256'] != INDEX_CSV_SHA256:
        p.append('SHA-256 de original_image_index.csv distinto del fijado')
    return p


def load_repo_transform(repo: Path) -> None:
    """Las 12 condiciones existentes se generan con la función transform() del propio repositorio."""
    import importlib.util
    global _GEN
    spec = importlib.util.spec_from_file_location('gen', repo / 'scripts' / 'generate_transformations_png.py')
    _GEN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_GEN)
    assert _GEN.CONDITIONS == EXISTING, 'Los parámetros del repositorio no coinciden con los de la adenda'


# ============================================================================ operadores (PIL)
def _tv():
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F
    return F, InterpolationMode


def op_existing(img: Image.Image, t: str, v: float) -> Image.Image:
    return _GEN.transform(img, t, v)


def zoom_box(z: float) -> tuple[int, int, int, int]:
    cw, ch = max(1, int(W / z)), max(1, int(H / z))
    return (W - cw) // 2, (H - ch) // 2, cw, ch


def op_z1(img, z):
    left, top, cw, ch = zoom_box(z)
    canvas = Image.new('RGB', (W, H), FILL_MEAN_RGB)
    canvas.paste(img.crop((left, top, left + cw, top + ch)), (left, top))
    return canvas


def op_z1r(img, z):
    left, top, cw, ch = zoom_box(z)
    arr = np.asarray(img.crop((left, top, left + cw, top + ch)))
    pad = ((top, H - ch - top), (left, W - cw - left), (0, 0))
    return Image.fromarray(np.pad(arr, pad, mode='reflect'))


def op_z2(img, z):
    _, _, cw, ch = zoom_box(z)
    return img.resize((cw, ch), resample=Image.Resampling.BILINEAR).resize((W, H), resample=Image.Resampling.BILINEAR)


def op_r1(img, a):
    F, IM = _tv()
    padded = F.pad(img, [PAD_R1, PAD_R1, PAD_R1, PAD_R1], padding_mode='reflect')
    rot = F.rotate(padded, angle=a, interpolation=IM.BILINEAR, expand=False, fill=0)
    return F.center_crop(rot, [H, W])


def op_r2(img, a):
    F, IM = _tv()
    rot = F.rotate(img, angle=a, interpolation=IM.BILINEAR, expand=True, fill=0)
    return F.center_crop(rot, [H, W])


def op_r3(img, a):
    F, IM = _tv()
    return F.rotate(img, angle=a, interpolation=IM.BILINEAR, expand=True, fill=0)


# ============================================================================ conjuntos
def build_sets() -> list[dict]:
    """35 conjuntos de extracción, en el orden de ejecución."""
    sets = [{'name': 'original_uint8', 'group': 'original', 'route': 'uint8', 'op': None, 'param': None}]
    for t, levels in EXISTING.items():
        for lv, v in levels.items():
            sets.append({'name': f'{t}_{lv}', 'group': 'existing', 'route': 'uint8', 'op': ('existing', t), 'param': v})
    sets.append({'name': 'original_float', 'group': 'original', 'route': 'float', 'op': None, 'param': None})
    for lv in LEVELS:
        sets.append({'name': f'gamma_{lv}', 'group': 'gamma', 'route': 'float', 'op': ('gamma', lv), 'param': None})
    for var in ['R1', 'R2', 'R3']:
        for lv, a in ANGLES.items():
            sets.append({'name': f'rot{var}_{lv}', 'group': 'rotation', 'route': 'uint8', 'op': (var, None), 'param': a})
    for var in ['Z1', 'Z1r', 'Z2']:
        for lv, z in ZOOMS.items():
            sets.append({'name': f'zoom{var}_{lv}', 'group': 'zoom', 'route': 'uint8', 'op': (var, None), 'param': z})
    assert len(sets) == 35
    return sets


def aux_kind(spec) -> str | None:
    if spec['op'] and spec['op'][0] == 'R2':
        return 'R2'
    if spec['op'] and spec['op'][0] == 'gamma':
        return 'gamma'
    return None


def apply_uint8_op(img, spec):
    kind, t = spec['op'] if spec['op'] else (None, None)
    v = spec['param']
    if kind is None:
        return img
    if kind == 'existing':
        return op_existing(img, t, v)
    return {'R1': op_r1, 'R2': op_r2, 'R3': op_r3, 'Z1': op_z1, 'Z1r': op_z1r, 'Z2': op_z2}[kind](img, v)


def r2_aux(img, a) -> np.ndarray:
    d = np.abs(np.asarray(op_r2(img, a), dtype=np.int16) - np.asarray(op_existing(img, 'rotation', a), dtype=np.int16))
    mae = d.reshape(-1, 3).mean(axis=0)
    return np.array([mae[0], mae[1], mae[2], d.max(), 100.0 * np.mean(d.max(axis=2) > 0)])


def gamma_aux(x01: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x01, y: (3, H, W) en [0, 1]."""
    new = (y >= 1.0) & (x01 < 1.0)
    l0 = np.tensordot(LUMA, x01, axes=1)
    l1 = np.tensordot(LUMA, y, axes=1)
    return np.array([l0.mean(), l1.mean(), l0.std(), l1.std(), new.any(axis=0).mean(), new.mean()])


# ============================================================================ luminancia y gamma
def channel_hist(arr: np.ndarray) -> np.ndarray:
    return np.stack([np.bincount(arr[..., c].ravel(), minlength=256) for c in range(3)]).astype(np.int64)


def luminance(arr01: np.ndarray) -> np.ndarray:
    return arr01[..., 0] * LUMA[0] + arr01[..., 1] * LUMA[1] + arr01[..., 2] * LUMA[2]


_V = np.arange(256) / 255.0


def delta_y_from_hist(hist: np.ndarray, gamma: float) -> np.ndarray:
    """ΔY medio exacto a partir de histogramas por canal; hist (…, 3, 256)."""
    gain = _V ** gamma - _V
    npix = hist[..., 0, :].sum(axis=-1)
    per_c = (hist * gain).sum(axis=-1)
    return (per_c @ LUMA) / npix


def calibrate_gamma(total_hist: np.ndarray, target: float, lo=GAMMA_BRACKET[0], hi=GAMMA_BRACKET[1],
                    tol=GAMMA_TOL, max_iter=200) -> float:
    """Bisección en (lo, hi) hasta |ΔY(γ) − ΔY_objetivo| < tol (adenda, sección de gamma)."""
    f = lambda g: float(delta_y_from_hist(total_hist, g)) - target
    if f(lo) < 0:
        raise RuntimeError(f'Objetivo ΔY={target} inalcanzable con gamma ≥ {lo}')
    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fm > 0:
            a = m
        else:
            b = m
    raise RuntimeError('La bisección de gamma no convergió')


def newly_saturated_component_fraction_from_hist(hist: np.ndarray, gamma: float) -> float:
    """Con la misma aritmética float32 que la ruta float: componentes < 255 que pasarían a valer 1."""
    v = (np.arange(256, dtype=np.float32) / np.float32(255.0)) ** np.float32(gamma)
    new = (v >= 1.0) & (np.arange(256) < 255)
    return float((hist * new).sum() / hist.sum())


# ============================================================================ umbrales casi cero
def thresholds_from_counts(counts: np.ndarray):
    """counts (C, NBINS) de activaciones no nulas → mediana aproximada, umbral 1 % y canales vivos."""
    total = counts.sum(axis=1)
    live = total > 0
    med = np.full(counts.shape[0], np.nan)
    cum = np.cumsum(counts, axis=1)
    for c in np.flatnonzero(live):
        k = int(np.searchsorted(cum[c], total[c] / 2.0))
        med[c] = 10 ** (LOG_LO + (k + 0.5) * (LOG_HI - LOG_LO) / NBINS)
    return med, 0.01 * med, live


def thresholds_digest(thr: dict, live: dict) -> str:
    h = hashlib.sha256()
    for b in BLOCKS:
        h.update(np.ascontiguousarray(thr[b], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(live[b], dtype=np.bool_).tobytes())
    return h.hexdigest()


# ============================================================================ comprobaciones geométricas
def rotation_mask_checks() -> dict:
    F, IM = _tv()
    out = {}
    for lv, a in ANGLES.items():
        full = Image.new('L', (W + 2 * PAD_R1, H + 2 * PAD_R1), 255)
        r1_fill = np.asarray(F.center_crop(F.rotate(full, angle=a, interpolation=IM.BILINEAR, expand=False, fill=0), [H, W]))
        core = Image.new('L', (W + 2 * PAD_R1, H + 2 * PAD_R1), 0)
        core.paste(255, (PAD_R1, PAD_R1, PAD_R1 + W, PAD_R1 + H))
        r1_core = np.asarray(F.center_crop(F.rotate(core, angle=a, interpolation=IM.BILINEAR, expand=False, fill=0), [H, W]))
        r0 = np.asarray(F.rotate(Image.new('L', (W, H), 255), angle=a, interpolation=IM.BILINEAR, expand=False, fill=0))
        out[lv] = {'angle': a,
                   'R1_pixels_touched_by_fill0': int(np.sum(r1_fill < 255)),
                   'R1_fraction_from_reflection_any': float(np.mean(r1_core < 255)),
                   'R1_fraction_from_reflection_full': float(np.mean(r1_core == 0)),
                   'R0_fraction_fill_any': float(np.mean(r0 < 255)),
                   'R0_fraction_fill_full': float(np.mean(r0 == 0))}
        assert out[lv]['R1_pixels_touched_by_fill0'] == 0, f'R1 contiene relleno fill=0 a {a}°'
    out['note'] = ('La geometría es la misma para todas las imágenes porque todas miden 512 × 384 '
                   '(comprobado en la etapa A); por eso la máscara se verifica una vez por ángulo.')
    return out


# ============================================================================ referencia (pares) y control de embeddings
def reference_pairs(labels: np.ndarray):
    n = len(labels)
    pair_i, pair_j = np.triu_indices(n, k=1)
    same = labels[pair_i] == labels[pair_j]
    eligible = {'random': np.arange(len(pair_i)), 'same_class': np.flatnonzero(same),
                'different_class': np.flatnonzero(~same)}
    rng = np.random.default_rng(SEED)
    sel = {k: rng.choice(v, size=min(N_PAIRS, len(v)), replace=False) for k, v in eligible.items()}
    return pair_i[sel['random']], pair_j[sel['random']]


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def embedding_gate(new: dict, old: dict, labels: np.ndarray, sel=None) -> tuple[pd.DataFrame, bool]:
    """new: {set: {block: (n, C)}}; old: NPZ verificados completos (2527 filas); sel: filas extraídas.

    B(b) se calcula siempre con los 2527 originales verificados (los 100 000 pares de referencia).
    """
    pi, pj = reference_pairs(labels)
    sel = np.arange(len(labels)) if sel is None else sel
    rows, ok = [], True
    B = {}
    for b in BLOCKS:
        n0 = _norm(old['original_uint8'][b].astype(np.float64))
        B[b] = float(np.median(np.sum(n0[pi] * n0[pj], 1)))
    for name in new:
        for b in BLOCKS:
            xn, xo = new[name][b].astype(np.float64), old[name][b][sel].astype(np.float64)
            cos = np.sum(_norm(xn) * _norm(xo), 1)
            nrm = (np.abs(np.linalg.norm(xn, axis=1) - np.linalg.norm(xo, axis=1))
                   / np.maximum(np.linalg.norm(xo, axis=1), 1e-12))
            row = {'set': name, 'block': b, 'B_reference': B[b], 'min_cos_new_vs_old': float(cos.min()),
                   'max_rel_norm_diff': float(nrm.max()), 'max_abs_A_adj_diff': np.nan}
            if name != 'original_uint8':
                on, oo = new['original_uint8'][b].astype(np.float64), old['original_uint8'][b][sel].astype(np.float64)
                a_new = (1 - np.sum(_norm(on) * _norm(xn), 1)) / (1 - B[b])
                a_old = (1 - np.sum(_norm(oo) * _norm(xo), 1)) / (1 - B[b])
                row['max_abs_A_adj_diff'] = float(np.max(np.abs(a_new - a_old)))
            row['passed'] = bool(row['min_cos_new_vs_old'] >= 1 - TOL_COS and row['max_rel_norm_diff'] <= TOL_NORM
                                 and (np.isnan(row['max_abs_A_adj_diff']) or row['max_abs_A_adj_diff'] <= TOL_AADJ))
            ok &= row['passed']
            rows.append(row)
    return pd.DataFrame(rows), bool(ok)


def post_summaries(sets_dir: Path, labels: np.ndarray, out: Path, log) -> dict:
    """B_float(b), coseno uint8–float de los originales y equivalencia R2–R0 (tablas de la adenda)."""
    def load(name):
        with np.load(sets_dir / f'{name}.npz', allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if len(labels) == N_EXPECTED:
        pi, pj = reference_pairs(labels)
    else:  # solo en la prueba reducida (--smoke): todos los pares del subconjunto
        pi, pj = np.triu_indices(len(labels), k=1)
    gu, gf = load('original_uint8'), load('original_float')
    rows = []
    for b in BLOCKS:
        nu, nf = _norm(gu[f'gap_{b}'].astype(np.float64)), _norm(gf[f'gap_{b}'].astype(np.float64))
        cos = np.sum(nu * nf, 1)
        rows.append({'block': b, 'B_uint8_recomputed': float(np.median(np.sum(nu[pi] * nu[pj], 1))),
                     'B_float': float(np.median(np.sum(nf[pi] * nf[pj], 1))),
                     'cos_uint8_vs_float_mean': float(cos.mean()), 'cos_uint8_vs_float_min': float(cos.min()),
                     'cos_uint8_vs_float_p1': float(np.percentile(cos, 1))})
    bf = pd.DataFrame(rows)
    bf.to_csv(out / 'G_B_float_y_rutas.csv', index=False)
    req = []
    for lv in LEVELS:
        z0, z2 = load(f'rotation_{lv}'), load(f'rotR2_{lv}')
        for b in BLOCKS:
            cos = np.sum(_norm(z0[f'gap_{b}'].astype(np.float64)) * _norm(z2[f'gap_{b}'].astype(np.float64)), 1)
            req.append({'level': lv, 'angle': ANGLES[lv], 'block': b, 'cos_mean': float(cos.mean()),
                        'cos_p1': float(np.percentile(cos, 1)), 'cos_min': float(cos.min()),
                        'MAE_R_mean': float(z2['r2_mae_R'].mean()), 'MAE_G_mean': float(z2['r2_mae_G'].mean()),
                        'MAE_B_mean': float(z2['r2_mae_B'].mean()),
                        'max_abs_diff_max': float(z2['r2_max_abs_diff'].max()),
                        'pct_pixels_different_mean': float(z2['r2_pct_pixels_different'].mean())})
    r2 = pd.DataFrame(req)
    r2['equivalent_cell'] = ((r2.cos_mean >= 0.9985) & (r2.cos_p1 >= 0.9949)
                             & (r2[['MAE_R_mean', 'MAE_G_mean', 'MAE_B_mean']].max(axis=1) <= 1.0))
    r2.to_csv(out / 'G_equivalencia_R2_R0.csv', index=False)
    eq = bool(r2.equivalent_cell.all())
    log(f"G: B_float = {', '.join(f'{v:.4f}' for v in bf.B_float)}; R2 operacionalmente equivalente a R0: {eq}.")
    return {'B_float': dict(zip(bf.block, bf.B_float)), 'R2_equivalent_to_R0': eq}


# ============================================================================ huella de ejecución
def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = canonical(v)
    return out


def fingerprint_diff(a: dict, b: dict) -> list[str]:
    fa, fb = flatten(a), flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def check_out_folder(out: Path, fp: dict) -> tuple[str, bool]:
    """Devuelve (fingerprint_id, reanudación). Sale con código 4 si la carpeta es incompatible."""
    fid = sha256_bytes(canonical(fp).encode())
    fpath = out / 'run_fingerprint.json'
    if fpath.exists():
        stored = json.loads(fpath.read_text(encoding='utf-8'))
        if stored.get('fingerprint_id') != fid:
            diff = fingerprint_diff(stored.get('fingerprint', {}), fp)
            print('ERROR: la carpeta de salida pertenece a otra ejecución. Campos distintos de la huella:\n  '
                  + '\n  '.join(diff[:50]) + '\nUse una carpeta nueva.', flush=True)
            sys.exit(4)
        return fid, True
    if any(out.iterdir()):
        print('ERROR: la carpeta de salida no está vacía y no tiene run_fingerprint.json. Use una carpeta nueva.',
              flush=True)
        sys.exit(4)
    write_json(fpath, {'fingerprint_id': fid, 'fingerprint': fp})
    return fid, False


# ============================================================================ etapa B: control de píxeles
def pixel_gate(paths, index, png_root: Path | None, log, workers: int, smoke: bool) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    def png(i, t, lv):
        return png_root / f'{t}_{lv}' / index['class'][i] / f'{Path(paths[i]).stem}.png'

    jobs = [(i, t, lv, v) for t, levels in EXISTING.items() for lv, v in levels.items() for i in range(len(paths))]
    if png_root is None or not png_root.exists():
        log('B: sin carpeta de PNG; el control de píxeles no está disponible (el control descansa en los embeddings).')
        return {'mode': 'no_disponible', 'n_compared': 0, 'passed': None, 'failures': []}
    complete = all(png(i, t, lv).exists() for (i, t, lv, _) in jobs)
    if complete:
        mode = 'prueba_reducida' if smoke else 'completo'
    else:
        mode = 'diagnostico'
        jobs = [j for j in jobs if png(j[0], j[1], j[2]).exists()]
        log(f'B: PNG incompletos; se compara lo disponible ({len(jobs)}) y el control queda como solo diagnóstico.')

    class DS(Dataset):
        def __len__(self):
            return len(jobs)

        def __getitem__(self, k):
            i, t, lv, v = jobs[k]
            with Image.open(paths[i]) as im:
                regen = np.asarray(op_existing(im.convert('RGB'), t, v), dtype=np.int16)
            with Image.open(png(i, t, lv)) as st:
                stored = np.asarray(st.convert('RGB'), dtype=np.int16)
            d = int(np.abs(regen - stored).max()) if regen.shape == stored.shape else 999
            return torch.tensor([k, d])

    maxd = np.zeros(len(jobs), dtype=np.int64)
    for batch in DataLoader(DS(), batch_size=64, num_workers=max(workers, 8)):  # lectura de Drive: limitada por E/S
        b = np.asarray(batch.numpy())
        maxd[b[:, 0]] = b[:, 1]
    rows = pd.DataFrame([{'row': i, 'file': Path(paths[i]).name, 'transformation': t, 'level': lv,
                          'max_abs_diff': int(d)} for (i, t, lv, _), d in zip(jobs, maxd)])
    summary = rows.groupby(['transformation', 'level']).max_abs_diff.agg(['size', 'max']).reset_index()
    passed = bool((maxd == 0).all())
    log(f'B: control de píxeles ({mode}): {len(jobs)} comparaciones, diferencia máxima global = {int(maxd.max())}.')
    return {'mode': mode, 'n_compared': len(jobs), 'passed': passed,
            'by_condition': summary.to_dict(orient='records'),
            'failures': rows[rows.max_abs_diff > 0].to_dict(orient='records')}


# ============================================================================ etapa D: calibración de gamma
def gamma_calibration(paths, log, workers: int):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                img = im.convert('RGB')
            arr = np.asarray(img)
            y0 = luminance(arr / 255.0)
            feats = [y0.mean(), y0.std()]
            for v in EXISTING['brightness'].values():
                yb = luminance(np.asarray(op_existing(img, 'brightness', v)) / 255.0)
                feats += [yb.mean(), yb.std()]
            return i, torch.from_numpy(channel_hist(arr)), torch.tensor(feats, dtype=torch.float64)

    hists = np.zeros((len(paths), 3, 256), dtype=np.int64)
    feats = np.zeros((len(paths), 8))
    for i, h, f in DataLoader(DS(), batch_size=32, num_workers=workers):
        i = np.asarray(i.numpy() if hasattr(i, 'numpy') else i)
        hists[i] = np.asarray(h.numpy())
        feats[i] = np.asarray(f.numpy())
    total = hists.sum(axis=0)
    res = {'luma_weights': LUMA.tolist(), 'bracket': list(GAMMA_BRACKET), 'tolerance_delta_Y': GAMMA_TOL,
           'levels': {}}
    per_image = pd.DataFrame({'Y_mean_original': feats[:, 0], 'Y_std_original': feats[:, 1]})
    for k, (lv, v) in enumerate(EXISTING['brightness'].items()):
        dy_b = feats[:, 2 + 2 * k] - feats[:, 0]
        target = float(dy_b.mean())
        g = calibrate_gamma(total, target)
        dy_g = delta_y_from_hist(hists, g)
        res['levels'][lv] = {
            'brightness_factor': v, 'target_delta_Y': target, 'gamma': g,
            'achieved_delta_Y_global': float(delta_y_from_hist(total, g)),
            'brightness_delta_Y_per_image_median': float(np.median(dy_b)),
            'brightness_delta_Y_per_image_iqr': [float(np.percentile(dy_b, 25)), float(np.percentile(dy_b, 75))],
            'gamma_delta_Y_per_image_median': float(np.median(dy_g)),
            'gamma_delta_Y_per_image_iqr': [float(np.percentile(dy_g, 25)), float(np.percentile(dy_g, 75))],
            'gamma_newly_saturated_component_fraction': newly_saturated_component_fraction_from_hist(total, g)}
        per_image[f'brightness_{lv}_delta_Y'] = dy_b
        per_image[f'brightness_{lv}_Y_std'] = feats[:, 3 + 2 * k]
        per_image[f'gamma_{lv}_delta_Y'] = dy_g
        log(f'D: gamma {lv}: objetivo ΔY = {target:.6f}, γ = {g:.6f}')
    return res, per_image


# ============================================================================ red
def setup_torch():
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def environment(torch) -> dict:
    import PIL
    import torchvision
    # Sin platform.platform(): la versión del núcleo cambia entre máquinas de Colab y no afecta al cálculo.
    return {'python': sys.version.split()[0], 'numpy': np.__version__,
            'pandas': pd.__version__, 'Pillow': PIL.__version__, 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32}


def build_model(device):
    import torch
    from torchvision.models import VGG16_Weights, vgg16
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wpath = Path(torch.hub.get_dir()) / 'checkpoints' / WEIGHTS_FILE
    # El GAP y la sparsity se calculan dentro del gancho, antes de que continúe la pasada (adenda).
    # state['mode']: 'hist' (primera pasada) o 'extract'; los resultados quedan en state['res'].
    state = {'mode': None, 'res': {}, 'counts': {}, 'thr': {}, 'live': {}, 'n_live': {}}

    def hook(module, inputs, A, name):
        if state['mode'] == 'hist':
            C = A.shape[1]
            bins = ((torch.log10(A.clamp_min(1e-30)) - LOG_LO) / (LOG_HI - LOG_LO) * NBINS).floor().long()
            bins = torch.where(A > 0, bins.clamp(0, NBINS - 1), torch.full_like(bins, NBINS))
            idx = (torch.arange(C, device=A.device).view(1, C, 1, 1) * (NBINS + 1) + bins).flatten()
            state['counts'][name] += torch.bincount(idx, minlength=C * (NBINS + 1)).view(C, NBINS + 1)
        elif state['mode'] == 'extract':
            hw = A.shape[2] * A.shape[3]
            cz = (A == 0).float().mean(dim=(2, 3))
            n_live = state['n_live'][name]
            near = ((A <= state['thr'][name]) & state['live'][name]).float().sum(dim=(1, 2, 3)) / max(n_live * hw, 1)
            state['res'][name] = {'gap': A.mean(dim=(2, 3)), 'czero': cz, 's_exact': cz.mean(dim=1),
                                  's_near': near if n_live > 0 else torch.full_like(near, float('nan'))}

    for b, idx in LAYER_INDEX.items():
        model.features[idx].register_forward_hook(lambda m, i, o, name=b: hook(m, i, o, name))
    return model, state, wpath


def make_dataset(paths, spec, gammas):
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms as T
    F, IM = _tv()
    pre_uint8 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    kind = aux_kind(spec)

    def pre_float(t):
        t = F.resize(t, [224, 224], interpolation=IM.BILINEAR, antialias=True)
        return F.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                img = im.convert('RGB')
            aux = np.full(N_AUX, np.nan, dtype=np.float64)
            if spec['route'] == 'uint8':
                out = apply_uint8_op(img, spec)
                if kind == 'R2':
                    aux[:len(AUX_FIELDS['R2'])] = r2_aux(img, spec['param'])
                x = pre_uint8(out)
            else:
                x01 = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
                y = x01
                if kind == 'gamma':
                    y = x01.pow(gammas[spec['op'][1]])
                    aux[:len(AUX_FIELDS['gamma'])] = gamma_aux(x01.numpy(), y.numpy())
                x = pre_float(y)
            return x, i, torch.from_numpy(aux)

    return DS()


def histogram_pass(model, state, paths, device, log, workers, batch):
    import torch
    from torch.utils.data import DataLoader
    ds = make_dataset(paths, {'route': 'uint8', 'op': None, 'param': None}, None)
    state['counts'] = {b: torch.zeros(DIMS[b], NBINS + 1, dtype=torch.int64, device=device) for b in BLOCKS}
    state['mode'] = 'hist'
    with torch.inference_mode():
        for x, _, _ in DataLoader(ds, batch_size=batch, num_workers=workers):
            model(x.to(device))
    state['mode'] = None
    log('E: primera pasada (histogramas de activaciones no nulas de los originales) completada.')
    return {b: state['counts'][b][:, :NBINS].cpu().numpy() for b in BLOCKS}


def extract_set(model, state, paths, rows_sel, spec, gammas, thr, live, device, workers, batch, fid, tdig):
    import torch
    from torch.utils.data import DataLoader
    n = len(paths)
    kind = aux_kind(spec)
    out = {f'gap_{b}': np.zeros((n, DIMS[b]), np.float32) for b in BLOCKS}
    out.update({f'czero_{b}': np.zeros((n, DIMS[b]), np.float32) for b in BLOCKS})
    out.update({f's_exact_{b}': np.zeros(n, np.float32) for b in BLOCKS})
    out.update({f's_near_{b}': np.zeros(n, np.float32) for b in BLOCKS})
    out['logits'] = np.zeros((n, 1000), np.float32)
    aux = np.full((n, N_AUX), np.nan)
    # Canales muertos: umbral −1 (nunca se cumple, porque la ReLU no da negativos) y fuera de la máscara.
    state['thr'] = {b: torch.tensor(np.nan_to_num(thr[b], nan=-1.0), dtype=torch.float32, device=device)
                    .view(1, -1, 1, 1) for b in BLOCKS}
    state['live'] = {b: torch.tensor(live[b], device=device).view(1, -1, 1, 1) for b in BLOCKS}
    state['n_live'] = {b: int(live[b].sum()) for b in BLOCKS}
    ds = make_dataset(paths, spec, gammas)
    state['mode'] = 'extract'
    with torch.inference_mode():
        for x, i, a in DataLoader(ds, batch_size=batch, num_workers=workers):
            i = i.numpy()
            state['res'] = {}
            logits = model(x.to(device))
            out['logits'][i] = logits.float().cpu().numpy()
            aux[i] = a.numpy()
            for b in BLOCKS:
                r = state['res'][b]
                for k in ('gap', 'czero', 's_exact', 's_near'):
                    out[f'{k}_{b}'][i] = r[k].cpu().numpy()
    state['mode'] = None
    if kind:
        for c, name in enumerate(AUX_FIELDS[kind]):
            out[name] = aux[:, c].astype(np.float32)
    out.update(set_metadata(spec, rows_sel, gammas, fid, tdig))
    return out


def set_metadata(spec, rows_sel, gammas, fid, tdig) -> dict:
    g = gammas[spec['op'][1]] if aux_kind(spec) == 'gamma' else np.nan
    return {'set_name': np.array(spec['name']), 'fingerprint_id': np.array(fid),
            'thresholds_digest': np.array(tdig), 'gamma': np.array(g, dtype=np.float64),
            'row_index': np.asarray(rows_sel, dtype=np.int64)}


def expected_keys(spec) -> set:
    keys = {'set_name', 'fingerprint_id', 'thresholds_digest', 'gamma', 'row_index', 'logits'}
    for b in BLOCKS:
        keys |= {f'gap_{b}', f'czero_{b}', f's_exact_{b}', f's_near_{b}'}
    kind = aux_kind(spec)
    return keys | set(AUX_FIELDS[kind]) if kind else keys


def valid_set_file(path: Path, spec, rows_sel, gammas, fid: str, tdig: str) -> tuple[bool, str]:
    """Un conjunto se reutiliza solo si coincide en todo con la ejecución actual."""
    if not path.exists():
        return False, 'no existe'
    n = len(rows_sel)
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != expected_keys(spec):
                return False, 'claves distintas del esquema'
            if str(z['set_name']) != spec['name']:
                return False, 'otro conjunto'
            if str(z['fingerprint_id']) != fid:
                return False, 'otra huella de ejecución'
            if str(z['thresholds_digest']) != tdig:
                return False, 'otros umbrales de casi cero'
            g_exp = gammas[spec['op'][1]] if aux_kind(spec) == 'gamma' else np.nan
            g = float(z['gamma'])
            if not ((np.isnan(g) and np.isnan(g_exp)) or g == g_exp):
                return False, 'otro gamma'
            if not np.array_equal(z['row_index'], np.asarray(rows_sel)):
                return False, 'otras imágenes'
            if z['logits'].shape != (n, 1000) or not np.isfinite(z['logits']).all():
                return False, 'logits incompletos'
            for b in BLOCKS:
                if z[f'gap_{b}'].shape != (n, DIMS[b]) or not np.isfinite(z[f'gap_{b}']).all():
                    return False, f'GAP {b} incompleto'
                if z[f'czero_{b}'].shape != (n, DIMS[b]) or z[f's_exact_{b}'].shape != (n,):
                    return False, f'sparsity {b} incompleta'
            return True, 'válido'
    except Exception as exc:
        return False, f'ilegible: {exc!r}'


def save_set(path: Path, out: dict) -> None:
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **out)
    tmp.replace(path)


# ============================================================================ principal
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--png-root', type=Path, default=None)
    ap.add_argument('--npz-root', required=True, type=Path)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--hist-batch', type=int, default=16)
    ap.add_argument('--smoke', type=int, default=0,
                    help='Prueba reducida: K imágenes por clase. Sus salidas NO sirven para el análisis.')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger()
    script = Path(__file__).resolve()
    log(f'Script {script.name}, SHA-256 {sha256(script)}')
    log(f'Adenda 1 v1.2 SHA-256 {ADDENDUM_SHA256}; criterios v1.3 SHA-256 {CRITERIA_SHA256}')

    # ---- 0: repositorio fijado
    rinfo = repo_info(args.repo)
    problems = repo_problems(rinfo)
    if problems:
        print('ERROR: repositorio no fijado o modificado:\n  ' + '\n  '.join(problems)
              + f'\nEjecute: git -C {args.repo} checkout {REPO_COMMIT}', flush=True)
        sys.exit(5)
    log(f"0: repositorio en el commit {rinfo['commit']}; generate_transformations_png.py y el índice verificados.")
    load_repo_transform(args.repo)

    index = pd.read_csv(args.repo / 'data/manifests/original_image_index.csv')
    assert len(index) == N_EXPECTED
    labels_full = index['class'].to_numpy()
    sel = np.arange(N_EXPECTED)
    if args.smoke:
        sel = index.groupby('class', sort=False).head(args.smoke).index.to_numpy()
        log(f'*** PRUEBA REDUCIDA (--smoke {args.smoke}): {len(sel)} imágenes. Salidas solo para comprobar '
            'que el código corre; no sirven para el análisis. ***')
    index = index.iloc[sel].reset_index(drop=True)
    paths = [str(args.dataset_root / c / f) for c, f in zip(index['class'], index.original_filename)]
    labels = labels_full[sel]

    # ---- A: hashes y tamaño de los JPEG
    rows = []
    for p in paths:
        with Image.open(p) as im:
            size = im.size
        rows.append({'row_index': None, 'file': f'{Path(p).parent.name}/{Path(p).name}', 'sha256': sha256(Path(p)),
                     'width': size[0], 'height': size[1]})
    jpeg = pd.DataFrame(rows)
    jpeg['row_index'] = sel
    assert ((jpeg.width == W) & (jpeg.height == H)).all(), 'Hay imágenes que no miden 512 × 384'
    jpeg_digest = sha256_bytes('\n'.join(f'{f} {h}' for f, h in zip(jpeg.file, jpeg.sha256)).encode())

    # NPZ verificados de referencia
    ref_files = {'original_uint8': args.npz_root / 'embeddings_originales_VGG16_R1.npz'}
    for t, levels in EXISTING.items():
        for lv in levels:
            ref_files[f'{t}_{lv}'] = args.npz_root / 'transformadas' / f'embeddings_{t}_{lv}.npz'
    ref_hashes = {k: sha256(p) for k, p in ref_files.items()}

    # ---- entorno, pesos y huella
    torch = setup_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, state, wpath = build_model(device)
    env = environment(torch)
    fp = {
        'script_sha256': sha256(script), 'addendum_sha256': ADDENDUM_SHA256, 'criteria_sha256': CRITERIA_SHA256,
        'repo': {k: rinfo[k] for k in ('commit', 'generate_transformations_png_sha256', 'original_image_index_sha256')},
        'mode': {'smoke_images_per_class': args.smoke, 'n_images': len(sel),
                 'row_index_sha256': sha256_bytes(np.asarray(sel, dtype=np.int64).tobytes())},
        'inputs': {'jpeg_digest': jpeg_digest, 'reference_npz_sha256': ref_hashes,
                   'png_root': str(args.png_root) if args.png_root else None},
        'weights': {'file': WEIGHTS_FILE, 'sha256': sha256(wpath)},
        'parameters': {'existing': EXISTING, 'fill_rgb': FILL_MEAN_RGB, 'pad_r1': PAD_R1,
                       'near_zero_bins': [NBINS, LOG_LO, LOG_HI], 'luma': LUMA.tolist(),
                       'gamma_bracket': GAMMA_BRACKET, 'gamma_tol': GAMMA_TOL,
                       'gate_tolerances': [TOL_COS, TOL_NORM, TOL_AADJ], 'seed': SEED,
                       'batch': args.batch, 'hist_batch': args.hist_batch, 'aux_fields': AUX_FIELDS},
        'environment': env,
    }
    fid, resumed = check_out_folder(out, fp)
    log.attach(out / 'registro_extraccion.txt')
    log(f"{'REANUDACIÓN' if resumed else 'Ejecución nueva'}: huella {fid}. GPU: {env['gpu']}.")
    if device.type != 'cuda':
        log('ADVERTENCIA: no hay GPU; la ejecución será muy lenta.')
    jpeg.to_csv(out / 'A_jpeg_sha256.csv', index=False)
    log(f'A: {len(paths)} JPEG con SHA-256 registrados; todos miden 512 × 384.')

    # ---- B: control de píxeles
    pix_path = out / 'B_control_pixeles.json'
    pix = json.loads(pix_path.read_text()) if pix_path.exists() else None
    if pix is not None and pix.get('fingerprint_id') == fid:
        log(f"B: se reutiliza el control de píxeles de esta misma huella (modo {pix['mode']}).")
    else:
        pix = pixel_gate(paths, index, args.png_root, log, args.workers, smoke=bool(args.smoke))
        pix['fingerprint_id'] = fid
        write_json(pix_path, pix)
    if pix['mode'] in ('completo', 'prueba_reducida') and not pix['passed']:
        log(f"B: FALLO del control de píxeles ({len(pix['failures'])} comparaciones distintas). Se detiene la ejecución.")
        sys.exit(2)

    # ---- C: geometría de rotación
    geo = rotation_mask_checks()
    write_json(out / 'C_rotacion_mascaras.json', geo)
    log('C: R1 sin píxeles de fill=0 en los tres ángulos; proporciones de reflexión y de relleno de R0 registradas.')

    # ---- D: calibración de gamma (determinista: se recalcula siempre)
    cal, cal_img = gamma_calibration(paths, log, args.workers)
    write_json(out / 'D_calibracion_gamma.json', cal)
    cal_img.insert(0, 'row_index', sel)
    cal_img.insert(1, 'image_id', index.image_id)
    cal_img.to_csv(out / 'D_calibracion_gamma_por_imagen.csv', index=False)
    gammas = {lv: cal['levels'][lv]['gamma'] for lv in LEVELS}

    # ---- E: umbrales de casi cero
    thr_path = out / 'E_umbrales_casi_cero.npz'
    counts = None
    if thr_path.exists():
        with np.load(thr_path, allow_pickle=False) as z:
            if str(z['fingerprint_id']) == fid:
                counts = {b: z[f'counts_{b}'] for b in BLOCKS}
                log('E: se reutilizan los histogramas de esta misma huella.')
    if counts is None:
        counts = histogram_pass(model, state, paths, device, log, args.workers, args.hist_batch)
    thr, live, med = {}, {}, {}
    for b in BLOCKS:
        med[b], thr[b], live[b] = thresholds_from_counts(counts[b])
    tdig = thresholds_digest(thr, live)
    tmp = thr_path.with_name('E_umbrales_casi_cero.tmp.npz')
    np.savez_compressed(tmp, fingerprint_id=np.array(fid), thresholds_digest=np.array(tdig),
                        **{f'counts_{b}': counts[b] for b in BLOCKS},
                        **{f'median_{b}': med[b] for b in BLOCKS}, **{f'threshold_{b}': thr[b] for b in BLOCKS},
                        **{f'live_{b}': live[b] for b in BLOCKS},
                        bin_edges_log10=np.linspace(LOG_LO, LOG_HI, NBINS + 1))
    tmp.replace(thr_path)
    log(f"E: canales vivos por bloque: {', '.join(f'{b} {int(live[b].sum())}/{DIMS[b]}' for b in BLOCKS)}.")

    sets = build_sets()
    sets_dir = out / 'conjuntos'
    sets_dir.mkdir(exist_ok=True)
    timings = {}

    def run(spec):
        path = sets_dir / f"{spec['name']}.npz"
        ok_file, why = valid_set_file(path, spec, sel, gammas, fid, tdig)
        if ok_file:
            log(f"   {spec['name']}: ya existe y coincide con esta ejecución; se omite.")
            return
        if path.exists():
            log(f"   {spec['name']}: el archivo existente no sirve ({why}); se rehace.")
        t0 = time.time()
        res = extract_set(model, state, paths, sel, spec, gammas, thr, live, device, args.workers, args.batch,
                          fid, tdig)
        save_set(path, res)
        timings[spec['name']] = round(time.time() - t0, 1)
        log(f"   {spec['name']}: extraído en {timings[spec['name']]} s.")

    # ---- F: originales y 12 existentes + control de embeddings
    log('F: extracción de originales (uint8) y de las 12 condiciones existentes.')
    for spec in sets[:13]:
        run(spec)
    new, old = {}, {}
    for spec in sets[:13]:
        with np.load(sets_dir / f"{spec['name']}.npz", allow_pickle=False) as z:
            new[spec['name']] = {b: z[f'gap_{b}'] for b in BLOCKS}
        with np.load(ref_files[spec['name']], allow_pickle=False) as z:
            old[spec['name']] = {b: z[b] for b in BLOCKS}
    gate, ok = embedding_gate(new, old, labels_full, sel)
    gate.to_csv(out / 'F_control_embeddings.csv', index=False)
    log(f"F: control de embeddings {'APROBADO' if ok else 'FALLIDO'}: coseno mínimo {gate.min_cos_new_vs_old.min():.9f}; "
        f"norma relativa máx. {gate.max_rel_norm_diff.max():.2e}; A_adj máx. {np.nanmax(gate.max_abs_A_adj_diff):.2e}.")
    ctx = dict(out=out, args=args, log=log, pix=pix, geo=geo, gammas=gammas, timings=timings, wpath=wpath,
               fid=fid, fp=fp, tdig=tdig)
    if not ok and args.smoke:
        log('F: (prueba reducida) el control falla, pero se continúa solo para probar el código.')
    elif not ok:
        log('F: se detiene la ejecución; no se extrae ninguna condición nueva (adenda, sección de alcance).')
        finish(**ctx, gate_ok=False)
        sys.exit(3)

    # ---- G: conjuntos restantes
    log('G: extracción de los 22 conjuntos restantes.')
    for spec in sets[13:]:
        run(spec)
    post = post_summaries(sets_dir, labels, out, log)
    finish(**ctx, gate_ok=ok, post=post)


def finish(out, args, log, pix, geo, gammas, timings, wpath, fid, fp, tdig, gate_ok, post=None):
    manifest = {
        'fingerprint_id': fid, 'fingerprint': fp,
        'smoke_test_images_per_class': args.smoke, 'valid_for_analysis': (args.smoke == 0 and gate_ok),
        'embedding_gate_passed': gate_ok,
        'pixel_gate': {k: pix[k] for k in ('mode', 'n_compared', 'passed')},
        'gammas': gammas, 'thresholds_digest': tdig, 'post_summaries': post, 'timings_s': timings,
        'weights_path': str(wpath), 'platform': platform.platform(),
        'operators': {'zoom_box': 'cw=int(512/z), ch=int(384/z), left=(512-cw)//2, top=(384-ch)//2',
                      'Z1_fill_rgb': FILL_MEAN_RGB, 'R1_pad': PAD_R1, 'near_zero_bins': [NBINS, LOG_LO, LOG_HI]},
        'sets': [s['name'] for s in build_sets()],
        'npz_schema': NPZ_SCHEMA, 'aux_fields_by_set_kind': AUX_FIELDS,
    }
    write_json(out / 'H_manifiesto.json', manifest)
    log('H: registro cerrado. A continuación se calculan los checksums de salida.')
    with open(out / 'H_checksums.sha256', 'w') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name != 'H_checksums.sha256' and not p.name.endswith('.tmp.npz'):
                f.write(f'{sha256(p)}  {p.relative_to(out)}\n')


if __name__ == '__main__':
    main()
