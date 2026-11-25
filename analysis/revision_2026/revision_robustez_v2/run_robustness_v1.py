"""Análisis de robustez preespecificado — revisión jimaging-4632369.

Implementa los criterios congelados v1.3
(SHA-256 del .docx: 492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57).

Entradas (solo lectura):
  --emb-root  carpeta con embeddings_originales_VGG16_R1.npz y transformadas/embeddings_{t}_{lv}.npz
  --repo      clon del repositorio público (manifiesto, CSV R1, pares de referencia, R3, R5)
Salidas: --out (tablas CSV, registro de ejecución, manifiesto con SHA-256).

Uso:  python run_robustness_v1.py --emb-root E --repo R --out O [--quick]
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

CRITERIA_VERSION = '1.3'
CRITERIA_SHA256 = '492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57'
SEED = 2026
N_PAIRS = 100_000
BLOCKS = ['Block1', 'Block2', 'Block3', 'Block4', 'Block5']
DIMS = {'Block1': 64, 'Block2': 128, 'Block3': 256, 'Block4': 512, 'Block5': 512}
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
TRANSFORMS = ['brightness', 'contrast', 'rotation', 'zoom']
LEVELS = ['low', 'medium', 'high']
CONDS = [(t, lv) for t in TRANSFORMS for lv in LEVELS]
BLOCK_REGION = np.array([0, 0, 1, 2, 2])  # 0 superficial (B1-B2), 1 intermedia (B3), 2 profunda (B4-B5)
REGION_NAME = {0: 'superficial', 1: 'intermedia', 2: 'profunda'}
REF_BLOCK = {'brightness': 1, 'contrast': 0, 'rotation': 2, 'zoom': 4}  # índices: B2, B1, B3, B5
CLAIM = {'brightness': 'C1', 'contrast': 'C2', 'rotation': 'C3', 'zoom': 'C5'}
F4_SEEDS = list(range(2026, 2046))
F4_DIM = 64
EPS = 1e-8

VERDICT_ORDER = ['Robusta', 'Parcialmente robusta', 'Dependiente de la métrica',
                 'Región estable, efecto no relevante', 'No sostenida']


# ----------------------------------------------------------------------------- utilidades
class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.t0 = time.time()
        path.write_text('', encoding='utf-8')

    def __call__(self, msg: str) -> None:
        line = f'[{time.time() - self.t0:8.1f} s] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def cos_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(normalize(x) * normalize(y), axis=1)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_rank_biserial(d: np.ndarray) -> float:
    """r_rb pareada con signo: positiva cuando la primera serie es mayor (d = primera − segunda)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos, neg = r[d > 0].sum(), r[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def indep_rank_biserial(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """r_rb para muestras independientes, positiva cuando a tiende a ser mayor que b; p bilateral."""
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 2 * res.statistic / (len(a) * len(b)) - 1
    return float(r), float(res.pvalue)


def eta2_h(H: float, k: int, n: int) -> float:
    return max(0.0, (H - k + 1) / (n - k))


def region_of(means: np.ndarray) -> np.ndarray:
    """Región pico (0/1/2) a partir de medias por bloque en el último eje."""
    return BLOCK_REGION[np.argmax(means, axis=-1)]


# ----------------------------------------------------------------------------- carga
def load_inputs(emb_root: Path, repo: Path, log: Logger):
    index = pd.read_csv(repo / 'data/manifests/original_image_index.csv')
    n = len(index)
    assert n == 2527
    orig = load_npz(emb_root / 'embeddings_originales_VGG16_R1.npz')
    trans = {}
    for t, lv in CONDS:
        trans[(t, lv)] = load_npz(emb_root / 'transformadas' / f'embeddings_{t}_{lv}.npz')
    for b in BLOCKS:
        assert orig[b].shape == (n, DIMS[b])
    log(f'Embeddings cargados: 13 archivos, n = {n}')

    # Pares de referencia: reproducción exacta del muestreo del notebook 02 (semilla 2026).
    pair_i, pair_j = np.triu_indices(n, k=1)
    labels = index['class'].to_numpy()
    same = labels[pair_i] == labels[pair_j]
    eligible = {'random': np.arange(len(pair_i)), 'same_class': np.flatnonzero(same),
                'different_class': np.flatnonzero(~same)}
    rng = np.random.default_rng(SEED)
    sel = {k: rng.choice(v, size=min(N_PAIRS, len(v)), replace=False) for k, v in eligible.items()}
    pi, pj = pair_i[sel['random']], pair_j[sel['random']]
    published_pairs = np.load(repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz')
    for b in BLOCKS:
        xn = normalize(orig[b])
        c = np.sum(xn[pi] * xn[pj], axis=1).astype(np.float32)
        err = float(np.max(np.abs(c.astype(np.float64) - published_pairs[f'{b}_random'])))
        assert err <= 1e-6, (b, err)
    log('Pares aleatorios de referencia reproducidos exactamente (5 bloques).')
    B_pub = np.array([float(np.median(published_pairs[f'{b}_random'].astype(np.float64))) for b in BLOCKS])
    return index, orig, trans, pi, pj, B_pub


# ----------------------------------------------------------------------------- familias
def compute_families(index, orig, trans, pi, pj, B_pub, repo: Path, log: Logger):
    n = len(index)
    C, K = len(CONDS), len(BLOCKS)
    S = np.empty((n, C, K))
    REL = np.empty((n, C, K))
    for ci, cond in enumerate(CONDS):
        for bi, b in enumerate(BLOCKS):
            x, y = orig[b], trans[cond][b]
            S[:, ci, bi] = cos_rows(x, y)
            REL[:, ci, bi] = np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + EPS)

    fam = {}
    # F1
    A1 = (1 - S) / (1 - B_pub)
    fam['F1'] = A1
    # Validación contra el CSV R1 publicado.
    pub = pd.read_csv(repo / 'data/derived_metrics/r1/image_level_adjusted_sensitivity.csv.gz')
    key = index[['image_id', 'class']].copy()
    max_err = 0.0
    for ci, (t, lv) in enumerate(CONDS):
        for bi, b in enumerate(BLOCKS):
            sub = pub[(pub.transformation == t) & (pub.level == lv) & (pub.block == b)]
            m = key.assign(a=A1[:, ci, bi]).merge(sub[['image_id', 'class', 'baseline_adjusted_sensitivity']],
                                                   on=['image_id', 'class'], validate='one_to_one')
            max_err = max(max_err, float(np.max(np.abs(m.a - m.baseline_adjusted_sensitivity))))
    log(f'F1 frente a A_adj publicado: error absoluto máximo = {max_err:.2e}')
    checks = {'F1_vs_published_A_adj_max_abs_err': max_err}

    # F2: coseno centrado con la media de los originales del bloque.
    A2 = np.empty((n, C, K))
    B2 = np.empty(K)
    for bi, b in enumerate(BLOCKS):
        mu = orig[b].mean(axis=0)
        xc = normalize(orig[b] - mu)
        B2[bi] = np.median(np.sum(xc[pi] * xc[pj], axis=1))
        for ci, cond in enumerate(CONDS):
            sc = np.sum(xc * normalize(trans[cond][b] - mu), axis=1)
            A2[:, ci, bi] = (1 - sc) / (1 - B2[bi])
    fam['F2'] = A2

    # F3: percentil con corrección de medio punto.
    A3 = np.empty((n, C, K))
    F3_at_min = np.empty((C, K))
    for bi, b in enumerate(BLOCKS):
        xn = normalize(orig[b])
        D = np.sort(1 - np.sum(xn[pi] * xn[pj], axis=1))
        for ci in range(C):
            cnt = np.searchsorted(D, 1 - S[:, ci, bi], side='right')
            A3[:, ci, bi] = (0.5 + cnt) / (len(D) + 1)
            F3_at_min[ci, bi] = np.mean(cnt == 0)
    F3_noninf = (F3_at_min > 0.5).sum(axis=1) >= 3
    fam['F3'] = A3

    # F4: proyección semiortogonal a 64 dimensiones, 20 semillas.
    A4 = np.empty((len(F4_SEEDS), n, C, K))
    B4 = np.empty((len(F4_SEEDS), K))
    for si, seed in enumerate(F4_SEEDS):
        rng = np.random.default_rng(seed)
        for bi, b in enumerate(BLOCKS):
            Q, _ = np.linalg.qr(rng.standard_normal((DIMS[b], F4_DIM)))
            xp = normalize(orig[b] @ Q)
            B4[si, bi] = np.median(np.sum(xp[pi] * xp[pj], axis=1))
            for ci, cond in enumerate(CONDS):
                sp = np.sum(xp * normalize(trans[cond][b] @ Q), axis=1)
                A4[si, :, ci, bi] = (1 - sp) / (1 - B4[si, bi])
    fam['F4'] = A4.mean(axis=0)
    # Comprobación: en el bloque 1 la proyección es ortogonal y F4 = F1.
    checks['F4_block1_vs_F1_max_abs_err'] = float(np.max(np.abs(A4[:, :, :, 0] - A1[None, :, :, 0])))
    log(f"F4 bloque 1 frente a F1 (debe ser ~0): {checks['F4_block1_vs_F1_max_abs_err']:.2e}")

    # Variante de sparsity: sin canales nulos en > 50 % de los originales del bloque.
    AS = np.empty((n, C, K))
    BS = np.empty(K)
    kept = {}
    for bi, b in enumerate(BLOCKS):
        keep = (orig[b] == 0).mean(axis=0) <= 0.5
        kept[b] = int(keep.sum())
        xn = normalize(orig[b][:, keep])
        BS[bi] = np.median(np.sum(xn[pi] * xn[pj], axis=1))
        for ci, cond in enumerate(CONDS):
            ss = np.sum(xn * normalize(trans[cond][b][:, keep]), axis=1)
            AS[:, ci, bi] = (1 - ss) / (1 - BS[bi])
    fam['FS'] = AS

    refs = pd.DataFrame({'block': BLOCKS, 'B_F1_published': B_pub, 'B_F2_centered': B2,
                         'B_F4_mean_over_seeds': B4.mean(axis=0), 'B_F4_sd_over_seeds': B4.std(axis=0, ddof=1),
                         'B_FS_sparsity_variant': BS, 'FS_channels_kept': [kept[b] for b in BLOCKS],
                         'channels_total': [DIMS[b] for b in BLOCKS]})
    log('Familias F1–F4 y variante de sparsity calculadas.')
    return S, REL, fam, A4, F3_noninf, F3_at_min, refs, checks


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_indices(index, n_boot: int):
    rng = np.random.default_rng(SEED)
    labels = index['class'].to_numpy()
    groups = [np.flatnonzero(labels == c) for c in CLASSES]
    n = len(labels)
    IDX = np.empty((n_boot, n), dtype=np.int64)
    for r in range(n_boot):
        IDX[r] = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
    W = np.stack([np.bincount(IDX[r], minlength=n) for r in range(n_boot)]).astype(np.float64)
    return IDX, W


def boot_means(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return (W @ A.reshape(n, -1) / n).reshape((W.shape[0],) + A.shape[1:])


# ----------------------------------------------------------------------------- CKA
def cka_bootstrap(orig, trans, W, log: Logger):
    """CKA lineal centrado; HSIC ponderado por multiplicidades bootstrap (fórmula de Gram)."""
    n = W.shape[1]
    Wt = W.T.copy()                      # n × R
    ones = np.ones((n, 1))
    rows = []
    boots = {}
    for bi, b in enumerate(BLOCKS):
        X = orig[b] - orig[b].mean(axis=0)
        Kx = X @ X.T
        Kw, KKw = Kx @ Wt, (Kx * Kx) @ Wt
        Kw1, KKw1 = Kx @ ones, (Kx * Kx) @ ones

        def hsic(Mw, Pw, MPw, w):
            r, q = Mw / n, Pw / n
            s, t = np.sum(w * r, 0) / n, np.sum(w * q, 0) / n
            return np.sum(w * MPw, 0) - 2 * n * np.sum(w * q * r, 0) + n * n * s * t

        hxx_b, hxx_1 = hsic(Kw, Kw, KKw, Wt), hsic(Kw1, Kw1, KKw1, ones)
        for ci, cond in enumerate(CONDS):
            Y = trans[cond][b] - orig[b].mean(axis=0)
            Ky = Y @ Y.T
            Lw, LLw, KLw = Ky @ Wt, (Ky * Ky) @ Wt, (Kx * Ky) @ Wt
            Lw1, LLw1, KLw1 = Ky @ ones, (Ky * Ky) @ ones, (Kx * Ky) @ ones
            cka_b = hsic(Kw, Lw, KLw, Wt) / np.sqrt(hxx_b * hsic(Lw, Lw, LLw, Wt))
            cka_1 = float((hsic(Kw1, Lw1, KLw1, ones) / np.sqrt(hxx_1 * hsic(Lw1, Lw1, LLw1, ones)))[0])
            boots[(ci, bi)] = cka_b
            rows.append({'transformation': cond[0], 'level': cond[1], 'block': b, 'cka': cka_1,
                         'ci_low': float(np.percentile(cka_b, 2.5)), 'ci_high': float(np.percentile(cka_b, 97.5))})
        log(f'CKA bootstrap: {b} listo.')
    return pd.DataFrame(rows), boots


# ----------------------------------------------------------------------------- análisis principal
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb-root', required=True, type=Path)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--quick', action='store_true', help='Prueba de humo: 50 réplicas y 10 extracciones')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'run_log.txt')
    n_boot, n_draws = (50, 10) if args.quick else (2000, 200)
    log(f'Criterios v{CRITERIA_VERSION}, SHA-256 {CRITERIA_SHA256}')
    log(f'Réplicas bootstrap = {n_boot}; extracciones balanceadas = {n_draws}; modo rápido = {args.quick}')

    index, orig, trans, pi, pj, B_pub = load_inputs(args.emb_root, args.repo, log)
    n = len(index)
    labels = index['class'].to_numpy()
    S, REL, fam, A4, F3_noninf, F3_at_min, refs, checks = compute_families(index, orig, trans, pi, pj, B_pub,
                                                                          args.repo, log)
    refs.to_csv(out / '01_referencias_por_familia.csv', index=False)
    A1 = fam['F1']
    C, K = len(CONDS), len(BLOCKS)
    pub_region = np.array([BLOCK_REGION[REF_BLOCK[t]] for t, _ in CONDS])

    # ---- medias puntuales por familia (tabla 1 de salidas)
    rows = []
    for f, A in fam.items():
        mean, med = A.mean(axis=0), np.median(A, axis=0)
        for ci, (t, lv) in enumerate(CONDS):
            for bi, b in enumerate(BLOCKS):
                rows.append({'family': f, 'transformation': t, 'level': lv, 'block': b,
                             'mean': mean[ci, bi], 'median': med[ci, bi]})
    for si, seed in enumerate(F4_SEEDS):
        mean = A4[si].mean(axis=0)
        for ci, (t, lv) in enumerate(CONDS):
            for bi, b in enumerate(BLOCKS):
                rows.append({'family': f'F4_seed{seed}', 'transformation': t, 'level': lv, 'block': b,
                             'mean': mean[ci, bi], 'median': np.nan})
    profiles = pd.DataFrame(rows)

    # ---- bootstrap
    IDX, W = bootstrap_indices(index, n_boot)
    np.save(out / 'bootstrap_indices.npy', IDX.astype(np.int32))
    log('Índices bootstrap generados y guardados.')
    bm = {f: boot_means(A, W) for f, A in fam.items()}
    ci_rows = []
    for f in fam:
        lo, hi = np.percentile(bm[f], 2.5, axis=0), np.percentile(bm[f], 97.5, axis=0)
        for ci, (t, lv) in enumerate(CONDS):
            for bi, b in enumerate(BLOCKS):
                ci_rows.append({'family': f, 'transformation': t, 'level': lv, 'block': b,
                                'mean_ci_low': lo[ci, bi], 'mean_ci_high': hi[ci, bi]})
    profiles = profiles.merge(pd.DataFrame(ci_rows), on=['family', 'transformation', 'level', 'block'], how='left')

    # Regiones y estabilidad (media) + sensibilidad con la mediana.
    reg_rows = []
    for f, A in fam.items():
        point = region_of(A.mean(axis=0))
        stab = (region_of(bm[f]) == pub_region[None, :]).mean(axis=0)
        point_med = region_of(np.median(A, axis=0))
        for ci, (t, lv) in enumerate(CONDS):
            med_boot = np.stack([np.median(A[:, ci, bi][IDX], axis=1) for bi in range(K)], axis=1)
            stab_med = float(np.mean(region_of(med_boot) == pub_region[ci]))
            reg_rows.append({'family': f, 'transformation': t, 'level': lv,
                             'published_region': REGION_NAME[pub_region[ci]],
                             'peak_block_mean': BLOCKS[int(np.argmax(A.mean(axis=0)[ci]))],
                             'peak_region_mean': REGION_NAME[point[ci]],
                             'stability_published_region_mean': float(stab[ci]),
                             'peak_region_median': REGION_NAME[point_med[ci]],
                             'stability_published_region_median': stab_med})
    regions = pd.DataFrame(reg_rows)
    seed_regions = region_of(A4.mean(axis=1))  # (20, C)
    seeds_in_pub = (seed_regions == pub_region[None, :]).sum(axis=0)
    regions['F4_seeds_in_published_region'] = regions.apply(
        lambda r: int(seeds_in_pub[CONDS.index((r.transformation, r.level))]) if r.family == 'F4' else np.nan, axis=1)
    regions['F3_noninformative'] = regions.apply(
        lambda r: bool(F3_noninf[CONDS.index((r.transformation, r.level))]) if r.family == 'F3' else np.nan, axis=1)
    log('Regiones pico y estabilidad bootstrap calculadas.')

    # ---- Pruebas de F1: Friedman (12) y contrastes del bloque de referencia (48)
    fr_rows, ct_rows = [], []
    for ci, (t, lv) in enumerate(CONDS):
        chi2, p = stats.friedmanchisquare(*[A1[:, ci, bi] for bi in range(K)])
        fr_rows.append({'transformation': t, 'level': lv, 'chi2': chi2, 'p': p, 'kendall_W': chi2 / (n * (K - 1))})
        rb = REF_BLOCK[t]
        for ob in range(K):
            if ob == rb:
                continue
            row = {'transformation': t, 'level': lv, 'reference_block': BLOCKS[rb], 'other_block': BLOCKS[ob],
                   'other_outside_region': bool(BLOCK_REGION[ob] != BLOCK_REGION[rb])}
            for f in ['F1', 'F2', 'F3', 'F4', 'FS']:
                d = fam[f][:, ci, rb] - fam[f][:, ci, ob]
                row[f'{f}_median_diff'] = float(np.median(d))
            d = A1[:, ci, rb] - A1[:, ci, ob]
            row['F1_mean_diff'] = float(d.mean())
            row['r_rb'] = paired_rank_biserial(d)
            row['wilcoxon_p'] = float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)
            mb = np.median(d[IDX], axis=1)
            row['median_diff_ci_low'], row['median_diff_ci_high'] = np.percentile(mb, [2.5, 97.5])
            ct_rows.append(row)
    friedman = pd.DataFrame(fr_rows)
    friedman['p_holm'] = holm(friedman.p)
    contrasts = pd.DataFrame(ct_rows)
    contrasts['p_holm'] = holm(contrasts.wilcoxon_p)
    contrasts['relevant'] = (contrasts.r_rb >= 0.30) & (contrasts.median_diff_ci_low > 0)
    log('Friedman (12) y contrastes de referencia (48) calculados.')

    # ---- Sensibilidad: B(b) re-estimado en cada réplica (pares ponderados m_i·m_j)
    G = {b: normalize(orig[b]) @ normalize(orig[b]).T for b in BLOCKS}
    rng_pairs = np.random.default_rng([SEED, 1])
    one_minus_S_boot = boot_means(1 - S, W)            # R × C × K
    Brep = np.empty((n_boot, K))
    for r in range(n_boot):
        pos = IDX[r]
        a = rng_pairs.integers(0, n, size=30_000)
        c = rng_pairs.integers(0, n, size=30_000)
        ok = (a != c) & (pos[a] != pos[c])
        a, c = pos[a[ok]][:20_000], pos[c[ok]][:20_000]
        assert len(a) == 20_000
        for bi, b in enumerate(BLOCKS):
            Brep[r, bi] = np.median(G[b][a, c])
    m_rep = one_minus_S_boot / (1 - Brep[:, None, :])
    stab_Brep = (region_of(m_rep) == pub_region[None, :]).mean(axis=0)
    regions = regions.merge(pd.DataFrame({'family': 'F1', 'transformation': [t for t, _ in CONDS],
                                          'level': [lv for _, lv in CONDS],
                                          'stability_F1_B_reestimated': stab_Brep}),
                            on=['family', 'transformation', 'level'], how='left')
    refs_boot = pd.DataFrame({'block': BLOCKS, 'B_published': B_pub, 'B_reest_median': np.median(Brep, 0),
                              'B_reest_ci_low': np.percentile(Brep, 2.5, 0), 'B_reest_ci_high': np.percentile(Brep, 97.5, 0)})
    refs_boot.to_csv(out / '02_B_reestimado_bootstrap.csv', index=False)
    log('Sensibilidad con B(b) re-estimado completada.')

    # ---- CKA
    cka, cka_boot = cka_bootstrap(orig, trans, W, log)
    pubcka = pd.read_csv(args.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv')
    mm = cka.merge(pubcka[['transformation', 'level', 'block', 'linear_CKA']], on=['transformation', 'level', 'block'],
                   validate='one_to_one')
    checks['CKA_vs_published_max_abs_err'] = float(np.max(np.abs(mm.cka - mm.linear_CKA)))
    log(f"CKA frente a la tabla publicada: error absoluto máximo = {checks['CKA_vs_published_max_abs_err']:.2e}")
    c4_rows = []
    for lv in LEVELS:
        ci = CONDS.index(('rotation', lv))
        diff_b = cka_boot[(ci, 0)] - cka_boot[(ci, 4)]
        d1 = cka[(cka.transformation == 'rotation') & (cka.level == lv)].set_index('block').cka
        diff = float(d1['Block1'] - d1['Block5'])
        lo, hi = np.percentile(diff_b, [2.5, 97.5])
        mono = bool(np.all(np.diff(d1[BLOCKS].to_numpy()) <= 0))
        c4_rows.append({'level': lv, 'cka_B1_minus_B5': diff, 'ci_low': lo, 'ci_high': hi,
                        'monotone_nonincreasing_point': mono, 'passes': (diff >= 0.02) and (lo > 0)})
    c4 = pd.DataFrame(c4_rows)

    # ---- Composición de clases: 200 extracciones balanceadas
    rng_draw = np.random.default_rng(SEED)
    groups = {c: np.flatnonzero(labels == c) for c in CLASSES}
    draws = np.stack([np.concatenate([np.sort(rng_draw.choice(groups[c], 137, replace=False)) for c in CLASSES])
                      for _ in range(n_draws)])
    pd.DataFrame({'draw': np.repeat(np.arange(n_draws), draws.shape[1]),
                  'image_id': index.image_id.to_numpy()[draws.ravel()],
                  'class': labels[draws.ravel()]}).to_csv(out / '10_extracciones_balanceadas_ids.csv', index=False)
    first137 = np.flatnonzero(index.original_filename.str.extract(r'(\d+)')[0].astype(int).to_numpy() <= 137)
    assert len(first137) == 822
    iu = np.triu_indices(822, k=1)

    def draw_regions(idx):
        fixed = region_of(A1[idx].mean(axis=0))
        Bd = np.array([np.median(G[b][np.ix_(idx, idx)][iu]) for b in BLOCKS])
        recomputed = region_of((1 - S[idx]).mean(axis=0) / (1 - Bd))
        return fixed, recomputed, Bd

    comp_fixed = np.empty((n_draws, C), int)
    comp_rec = np.empty((n_draws, C), int)
    for d in range(n_draws):
        comp_fixed[d], comp_rec[d], _ = draw_regions(draws[d])
    f137_fixed, f137_rec, B137 = draw_regions(first137)
    comp = pd.DataFrame({'transformation': [t for t, _ in CONDS], 'level': [lv for _, lv in CONDS],
                         'published_region': [REGION_NAME[r] for r in pub_region],
                         'conservation_B_fixed': (comp_fixed == pub_region).mean(axis=0),
                         'conservation_B_recomputed': (comp_rec == pub_region).mean(axis=0),
                         'first137_region_B_fixed': [REGION_NAME[r] for r in f137_fixed],
                         'first137_region_B_recomputed': [REGION_NAME[r] for r in f137_rec]})
    comp['sensitive_to_class_balance'] = (comp.conservation_B_fixed < 0.90) | (comp.conservation_B_recomputed < 0.90)
    log('Sensibilidad a la composición de clases completada.')

    # ---- Veredictos por condición y por afirmación
    reg = regions.set_index(['family', 'transformation', 'level'])
    vrows = []
    for ci, (t, lv) in enumerate(CONDS):
        claim = CLAIM[t]
        rb = REF_BLOCK[t]
        ct = contrasts[(contrasts.transformation == t) & (contrasts.level == lv)].set_index('other_block')
        f1_stab = reg.loc[('F1', t, lv), 'stability_published_region_mean']
        f1_point_ok = reg.loc[('F1', t, lv), 'peak_region_mean'] == REGION_NAME[pub_region[ci]]
        outside = ct[ct.other_outside_region]
        effect_ok = bool(outside.relevant.all())
        if claim == 'C3':
            dir_blocks = ['Block1', 'Block5']
        elif claim == 'C5':
            dir_blocks = ['Block1']
        else:
            dir_blocks = []
        direction_ok = bool(all(ct.loc[b, 'median_diff_ci_low'] > 0 for b in dir_blocks))
        plateau = bool(ct.r_rb.min() < 0.10)
        fam_status = {}
        for f in ['F2', 'F3', 'F4']:
            if f == 'F3' and F3_noninf[ci]:
                fam_status[f] = 'no informativa'
                continue
            ok = (reg.loc[(f, t, lv), 'peak_region_mean'] == REGION_NAME[pub_region[ci]]) and \
                 (reg.loc[(f, t, lv), 'stability_published_region_mean'] >= 0.70)
            if f == 'F4':
                ok = ok and seeds_in_pub[ci] >= 16
            ok = ok and all(ct.loc[b, f'{f}_median_diff'] > 0 for b in dir_blocks)
            fam_status[f] = 'concuerda' if ok else 'discrepa'
        n_disc = sum(v == 'discrepa' for v in fam_status.values())
        if f1_stab < 0.90 or not f1_point_ok:
            verdict = 'No sostenida'
        elif not (effect_ok and direction_ok):
            verdict = 'Región estable, efecto no relevante'
        elif n_disc == 0:
            verdict = 'Robusta'
        elif n_disc == 1:
            verdict = 'Parcialmente robusta'
        else:
            verdict = 'Dependiente de la métrica'
        crow = comp.iloc[ci]
        vrows.append({'claim': claim, 'transformation': t, 'level': lv, 'reference_block': BLOCKS[rb],
                      'published_region': REGION_NAME[pub_region[ci]], 'F1_stability': f1_stab,
                      'F1_effect_ok': effect_ok, 'F1_direction_ok': direction_ok, 'plateau': plateau,
                      'min_r_rb_reference': float(ct.r_rb.min()),
                      'F2': fam_status['F2'], 'F3': fam_status['F3'], 'F4': fam_status['F4'],
                      'n_discordant': n_disc, 'verdict': verdict,
                      'sparsity_variant_region': reg.loc[('FS', t, lv), 'peak_region_mean'],
                      'sparsity_sensitive': reg.loc[('FS', t, lv), 'peak_region_mean'] != REGION_NAME[pub_region[ci]],
                      'class_balance_sensitive': bool(crow.sensitive_to_class_balance),
                      'F1_stability_B_reestimated': reg.loc[('F1', t, lv), 'stability_F1_B_reestimated']})
    vcond = pd.DataFrame(vrows)

    def claim_verdict(vs: list[str]) -> str:
        if all(v == 'Robusta' for v in vs):
            return 'Robusta'
        if sum(v in ('Robusta', 'Parcialmente robusta') for v in vs) >= 2:
            return 'Parcialmente robusta'
        return max(vs, key=VERDICT_ORDER.index)

    crows = []
    for t in TRANSFORMS:
        sub = vcond[vcond.transformation == t]
        v = claim_verdict(sub.verdict.tolist())
        if v == 'Robusta' and sub.class_balance_sensitive.any():
            v = 'Robusta bajo la composición original, pero sensible al balance de clases'
        crows.append({'claim': CLAIM[t], 'transformation': t, 'verdict': v,
                      'by_level': '; '.join(f'{r.level}: {r.verdict}' for r in sub.itertuples()),
                      'plateau_any': bool(sub.plateau.any()), 'sparsity_sensitive_any': bool(sub.sparsity_sensitive.any()),
                      'class_balance_sensitive_any': bool(sub.class_balance_sensitive.any())})
    c4_ok = bool(c4.passes.all())
    crows.append({'claim': 'C4', 'transformation': 'rotation (CKA)',
                  'verdict': 'Sostenida (criterio CKA)' if c4_ok else 'No sostenida (criterio CKA)',
                  'by_level': '; '.join(f'{r.level}: {"cumple" if r.passes else "no cumple"}' for r in c4.itertuples())})
    held = {'Robusta', 'Parcialmente robusta', 'Robusta bajo la composición original, pero sensible al balance de clases'}
    vmap = {r['claim']: r['verdict'] for r in crows}
    c6 = (vmap['C1'] in held or vmap['C2'] in held) and vmap['C5'] in held
    crows.append({'claim': 'C6', 'transformation': '(C1 o C2) y C5', 'verdict': 'Sostenida' if c6 else 'No sostenida',
                  'by_level': f"C1={vmap['C1']}; C2={vmap['C2']}; C5={vmap['C5']}"})
    vclaim = pd.DataFrame(crows)
    log('Veredictos calculados.')

    # ---- Clases (bloque 5)
    S5 = S[:, :, 4]
    Bcls = {}
    for c in CLASSES:
        g = groups[c]
        Bcls[c] = float(np.median(G['Block5'][np.ix_(g, g)][np.triu_indices(len(g), k=1)]))
    A5 = {'random': (1 - S5) / (1 - B_pub[4]),
          'same_class': (1 - S5) / (1 - np.array([Bcls[c] for c in labels]))[:, None]}
    cls_rows, kw_rows = [], []
    for den, A in A5.items():
        for ci, (t, lv) in enumerate(CONDS):
            samples = [A[groups[c], ci] for c in CLASSES]
            H, p = stats.kruskal(*samples)
            kw_rows.append({'denominator': den, 'transformation': t, 'level': lv, 'H': H, 'p': p,
                            'eta2_H': eta2_h(H, 6, n)})
            for c, s in zip(CLASSES, samples):
                q1, med, q3 = np.percentile(s, [25, 50, 75])
                cls_rows.append({'denominator': den, 'transformation': t, 'level': lv, 'class': c,
                                 'B_class': Bcls[c] if den == 'same_class' else B_pub[4],
                                 'median': med, 'q1': q1, 'q3': q3, 'n': len(s)})
    cls_summary = pd.DataFrame(cls_rows)
    kw = pd.DataFrame(kw_rows)
    kw['p_holm'] = kw.groupby('denominator').p.transform(holm)
    top_rows = []
    for den, A in A5.items():
        top = np.empty((n_draws, C), dtype=object)
        eta = np.empty((n_draws, C))
        for d in range(n_draws):
            idx = draws[d]
            lab = labels[idx]
            for ci in range(C):
                meds = [np.median(A[idx[lab == c], ci]) for c in CLASSES]
                top[d, ci] = CLASSES[int(np.argmax(meds))]
                H, _ = stats.kruskal(*[A[idx[lab == c], ci] for c in CLASSES])
                eta[d, ci] = eta2_h(H, 6, len(idx))
        for ci, (t, lv) in enumerate(CONDS):
            row = {'denominator': den, 'transformation': t, 'level': lv,
                   'eta2_H_median': float(np.median(eta[:, ci])),
                   'eta2_H_p2_5': float(np.percentile(eta[:, ci], 2.5)),
                   'eta2_H_p97_5': float(np.percentile(eta[:, ci], 97.5))}
            for c in CLASSES:
                row[f'freq_top_{c}'] = float(np.mean(top[:, ci] == c))
            top_rows.append(row)
    cls_draws = pd.DataFrame(top_rows)
    k1 = cls_draws[(cls_draws.transformation == 'zoom') & (cls_draws.level.isin(['low', 'medium']))]
    k1_ok = bool((k1.freq_top_cardboard >= 0.90).all())
    k2_rows = []
    ci_zh = CONDS.index(('zoom', 'high'))
    for den, A in A5.items():
        for other in [c for c in CLASSES if c != 'cardboard']:
            r, p = indep_rank_biserial(A[groups['cardboard'], ci_zh], A[groups[other], ci_zh])
            k2_rows.append({'denominator': den, 'other_class': other, 'r_rb_cardboard_vs_other': r, 'p': p})
    k2 = pd.DataFrame(k2_rows)
    k2['p_holm'] = k2.groupby('denominator').p.transform(holm)
    log('Análisis por clase completados.')

    # ---- K3: gradiente periférico ajustado por bounding box
    ext = pd.read_csv(args.repo / 'data/derived_metrics/r5/01_object_extent_by_image.csv')
    grad = pd.read_csv(args.repo / 'data/derived_metrics/r5/02_peripheral_gradient_by_image_and_zoom.csv')
    ext['edge_contact'] = (ext.bbox_x0 <= 0) | (ext.bbox_y0 <= 0) | (ext.bbox_x1 >= ext.width) | (ext.bbox_y1 >= ext.height)
    ext['bbox_ge_095'] = ext.bbox_area_fraction >= 0.95
    qc = ext.groupby('class').agg(n=('image_id', 'size'), edge_contact_rate=('edge_contact', 'mean'),
                                  bbox_ge_095_rate=('bbox_ge_095', 'mean'),
                                  either_rate=('edge_contact', lambda s: np.mean(s | ext.loc[s.index, 'bbox_ge_095']))).reset_index()
    m = grad.merge(ext[['image_id', 'class', 'bbox_area_fraction', 'edge_contact', 'bbox_ge_095']],
                   on=['image_id', 'class'], validate='many_to_one')
    k3_rows = []
    for subset, mask_fn in [('todas', lambda d: np.ones(len(d), bool)),
                            ('sin_contacto_ni_bbox_ge_095', lambda d: ~(d.edge_contact | d.bbox_ge_095).to_numpy())]:
        for lv in LEVELS:
            d = m[m.level == lv]
            d = d[mask_fn(d)].copy()
            rg, rbx = stats.rankdata(d.gradient_removed_pct), stats.rankdata(d.bbox_area_fraction)
            slope, intercept = np.polyfit(rbx, rg, 1)
            d['resid'] = rg - (intercept + slope * rbx)
            H, p_kw = stats.kruskal(*[d.resid[d['class'] == c] for c in CLASSES])
            meds = d.groupby('class').resid.median()
            res = []
            for other in [c for c in CLASSES if c != 'cardboard']:
                r, p = indep_rank_biserial(d.resid[d['class'] == 'cardboard'].to_numpy(),
                                           d.resid[d['class'] == other].to_numpy())
                res.append((other, r, p))
            ph = holm([x[2] for x in res])
            ok = (meds.idxmax() == 'cardboard') and all(r >= 0.10 and p < 0.05 for (_, r, _), p in zip(res, ph))
            for (other, r, p), pa in zip(res, ph):
                k3_rows.append({'subset': subset, 'level': lv, 'n': len(d), 'kw_H': H, 'kw_p': p_kw,
                                'eta2_H': eta2_h(H, 6, len(d)), 'cardboard_highest_median_resid': meds.idxmax() == 'cardboard',
                                'other_class': other, 'r_rb_cardboard_vs_other': r, 'p': p, 'p_holm': pa,
                                'K3_holds_at_level': ok})
    k3 = pd.DataFrame(k3_rows)
    log('K3 y control de calidad de máscaras completados.')

    # ---- Sparsity
    sp_rows, assoc_rows, chan_rows = [], [], []
    z0 = {b: (orig[b] == 0).mean(axis=1) for b in BLOCKS}
    for bi, b in enumerate(BLOCKS):
        X = orig[b]
        zero_frac_chan = (X == 0).mean(axis=0)
        dead = zero_frac_chan == 1.0
        nz_med = np.array([np.median(X[X[:, j] > 0, j]) if (X[:, j] > 0).any() else np.nan for j in range(X.shape[1])])
        thr = 0.01 * nz_med
        ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
        chan_rows.append({'block': b, 'channels': X.shape[1], 'dead_channels': int(dead.sum()),
                          'channels_zero_in_some_images': int(((zero_frac_chan > 0) & ~dead).sum()),
                          'channels_never_zero': int((zero_frac_chan == 0).sum()),
                          'channels_zero_gt50pct': int((zero_frac_chan > 0.5).sum()),
                          'participation_ratio': float(ev.sum() ** 2 / np.sum(ev ** 2))})
        for ci, cond in enumerate([('original', '')] + CONDS):
            Y = X if ci == 0 else trans[cond][b]
            near = (Y[:, ~dead] <= thr[None, ~dead]).mean()
            sp_rows.append({'block': b, 'transformation': cond[0], 'level': cond[1],
                            'exact_zero_pair_fraction': float((Y == 0).mean()),
                            'near_zero_pair_fraction_live_channels': float(near)})
        for ci, cond in enumerate(CONDS):
            zt = (trans[cond][b] == 0).mean(axis=1)
            dz = np.abs(zt - z0[b])
            a = A1[:, ci, bi]
            row = {'block': b, 'transformation': cond[0], 'level': cond[1],
                   'mean_abs_dz': float(dz.mean()), 'share_images_dz_zero': float(np.mean(dz == 0))}
            if np.ptp(dz) == 0:
                row.update({'rho_abs': np.nan, 'ci_low': np.nan, 'ci_high': np.nan})
            else:
                row['rho_abs'] = float(stats.spearmanr(a, dz).statistic)
                ra, rd = stats.rankdata(a[IDX], axis=1), stats.rankdata(dz[IDX], axis=1)
                ra -= ra.mean(1, keepdims=True); rd -= rd.mean(1, keepdims=True)
                rb_ = np.sum(ra * rd, 1) / np.sqrt(np.sum(ra ** 2, 1) * np.sum(rd ** 2, 1))
                row['ci_low'], row['ci_high'] = np.nanpercentile(rb_, [2.5, 97.5])
            sd = zt - z0[b]
            row['rho_signed_descriptive'] = float(stats.spearmanr(a, sd).statistic) if np.ptp(sd) > 0 else np.nan
            assoc_rows.append(row)
    sparsity_desc = pd.DataFrame(sp_rows)
    sparsity_chan = pd.DataFrame(chan_rows)
    sparsity_assoc = pd.DataFrame(assoc_rows)
    log('Sparsity completada.')

    # ---- Clipping (R3-2)
    clip = pd.read_csv(args.repo / 'data/derived_metrics/r3/01_brightness_clipping_by_image.csv')
    cl_rows, prof_rows = [], []
    for lv in LEVELS:
        cmap = clip[clip.level == lv].set_index(['class', 'original_filename']).newly_any255_proportion
        cvec = cmap.loc[list(zip(index['class'], index.original_filename))].to_numpy()
        ci = CONDS.index(('brightness', lv))
        for bi, b in enumerate(BLOCKS):
            for metric, M in [('A_adj', A1), ('relative_euclidean', REL)]:
                r = stats.spearmanr(cvec, M[:, ci, bi])
                cl_rows.append({'level': lv, 'block': b, 'metric': metric, 'rho': r.statistic, 'p': r.pvalue})
        rc = stats.rankdata(cvec)
        terc = pd.qcut(cvec, 3, labels=['T1_bajo', 'T2_medio', 'T3_alto'])
        for bi, b in enumerate(BLOCKS):
            a = A1[:, ci, bi]
            slope, intercept = np.polyfit(rc, a, 1)
            adj = a - (intercept + slope * rc) + a.mean()
            prof_rows.append({'level': lv, 'block': b, 'group': 'todas', 'mean_A_adj': a.mean(),
                              'mean_A_adj_residualized_literal': adj.mean(),
                              'mean_clipping': cvec.mean()})
            for tname in ['T1_bajo', 'T2_medio', 'T3_alto']:
                msk = np.asarray(terc == tname)
                prof_rows.append({'level': lv, 'block': b, 'group': tname, 'mean_A_adj': a[msk].mean(),
                                  'mean_A_adj_residualized_literal': np.nan, 'mean_clipping': cvec[msk].mean()})
    clipping = pd.DataFrame(cl_rows)
    clipping['p_holm'] = holm(clipping.p)
    clip_prof = pd.DataFrame(prof_rows)
    peak_by_tercile = (clip_prof.loc[clip_prof.groupby(['level', 'group']).mean_A_adj.idxmax(), ['level', 'group', 'block']]
                       .rename(columns={'block': 'peak_block'}))
    log('Clipping completado.')

    # ---- Guardado
    tables = {
        '03_perfiles_por_familia.csv': profiles,
        '04_regiones_estabilidad.csv': regions,
        '05_friedman_F1.csv': friedman,
        '06_contrastes_referencia_F1.csv': contrasts,
        '07_cka_por_bloque.csv': cka,
        '08_C4_cka_B1_menos_B5.csv': c4,
        '09_composicion_clases.csv': comp,
        '11_veredictos_por_condicion.csv': vcond,
        '12_veredictos_por_afirmacion.csv': vclaim,
        '13_clases_resumen_bloque5.csv': cls_summary,
        '14_clases_kruskal_bloque5.csv': kw,
        '15_clases_extracciones_bloque5.csv': cls_draws,
        '16_K2_zoom_alto_cardboard.csv': k2,
        '17_K3_gradiente_ajustado.csv': k3,
        '18_K3_qc_mascaras.csv': qc,
        '19_sparsity_descriptivo.csv': sparsity_desc,
        '20_sparsity_canales_dimension.csv': sparsity_chan,
        '21_sparsity_asociacion.csv': sparsity_assoc,
        '22_clipping_spearman.csv': clipping,
        '23_clipping_perfiles.csv': clip_prof,
        '24_clipping_pico_por_tercil.csv': peak_by_tercile,
        '25_F3_fraccion_en_minimo.csv': pd.DataFrame(F3_at_min, columns=BLOCKS).assign(
            transformation=[t for t, _ in CONDS], level=[lv for _, lv in CONDS], noninformative=F3_noninf),
    }
    for name, df in tables.items():
        df.to_csv(out / name, index=False, float_format='%.6g')
    checks['K1_holds'] = k1_ok
    checks['B_first137_recomputed'] = B137.tolist()

    script = Path(__file__).resolve()
    inputs = sorted(args.emb_root.rglob('*.npz')) + [
        args.repo / 'data/manifests/original_image_index.csv',
        args.repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz',
        args.repo / 'data/derived_metrics/r1/image_level_adjusted_sensitivity.csv.gz',
        args.repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv',
        args.repo / 'data/derived_metrics/r3/01_brightness_clipping_by_image.csv',
        args.repo / 'data/derived_metrics/r5/01_object_extent_by_image.csv',
        args.repo / 'data/derived_metrics/r5/02_peripheral_gradient_by_image_and_zoom.csv']
    manifest = {
        'criteria_version': CRITERIA_VERSION, 'criteria_docx_sha256': CRITERIA_SHA256,
        'script': script.name, 'script_sha256': sha256(script),
        'quick_mode': args.quick, 'n_boot': n_boot, 'n_draws': n_draws,
        'seeds': {'bootstrap': SEED, 'balanced_draws': SEED, 'pair_sampling_B_reestimation': [SEED, 1],
                  'reference_pairs': SEED, 'F4_projection': F4_SEEDS},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__},
        'inputs_sha256': {str(p.relative_to(args.emb_root) if args.emb_root in p.parents else
                              p.relative_to(args.repo)): sha256(p) for p in inputs},
        'checks': checks,
        'interpretations_and_deviations': [
            'D1 (clipping): un ajuste por residuos centrados en la media no puede cambiar la media por bloque; '
            'se informa la versión literal (idéntica por construcción) y, como complemento descriptivo no '
            'preespecificado, perfiles por tercil de clipping.',
            'I1: "diferencia pareada" en los criterios de dirección se implementa como la mediana de las '
            'diferencias por imagen, igual que en la regla de tamaño de efecto.',
            'I2: estabilidad y región de F2–F4 se juzgan respecto de la región informada en el manuscrito enviado.',
        ],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    with open(out / 'checksums_outputs.sha256', 'w') as f:
        for p in sorted(out.iterdir()):
            if p.name != 'checksums_outputs.sha256':
                f.write(f'{sha256(p)}  {p.name}\n')
    log('Ejecución terminada.')


if __name__ == '__main__':
    main()
