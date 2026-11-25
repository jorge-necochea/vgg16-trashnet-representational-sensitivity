"""Pruebas reproducibles de analyze_adenda_v2.py con datos sintéticos.

No necesitan la extracción real ni el repositorio: construyen en una carpeta temporal una extracción
sintética completa (35 conjuntos, huella, checksums), un paquete v2 mínimo (bootstrap estratificado y
extracciones balanceadas) y un repositorio mínimo. Los perfiles sintéticos están diseñados para que gamma
tenga su pico en B2, R1 en B3 y Z1/Z1r en B5, y para que haya tres pares de imágenes duplicadas entre clases.

Comprueban:
  - que el script rechaza la extracción sintética sin --test-fixture (identidad: código 3);
  - los códigos de parada: checksums (2), huella (3), metadatos de un NPZ (4), bootstrap no estratificado (5),
    carpeta de salida no vacía (6);
  - la ejecución completa en modo de prueba: tablas, manifiesto, checksums de salida, familias de Holm,
    veredictos esperados, hallazgo de duplicados;
  - funciones aisladas: Holm, r_rb, bandas, métricas de la cabeza ImageNet, Spearman bootstrap,
    razón q de CKA con réplicas excluidas y todas las ramas de redacción.

Uso:  python tests/test_analisis.py [--repo-real RUTA_AL_REPOSITORIO]
(--repo-real es opcional: comprueba además que los tres archivos fijados del repositorio real pasan.)
"""
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
SCRIPT = HERE.parent / 'analyze_adenda_v2.py'
sys.path.insert(0, str(HERE.parent))
import analyze_adenda_v2 as A  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--repo-real', type=Path, default=None)
cli = ap.parse_args()
results = {}


def check(name, cond):
    results[name] = bool(cond)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ============================================================================ fixture sintético
PER_CLASS, DRAW_K, N_DRAWS, N_BOOT = 40, 20, 12, 60
DIMS = [64, 80, 96, 128, 128]
PROFILES = {  # A_adj medio objetivo por bloque
    'gamma': [0.25, 0.60, 0.12, 0.08, 0.06],
    'R': [0.15, 0.20, 0.60, 0.20, 0.12],
    'Z1': [0.06, 0.08, 0.12, 0.35, 0.70],
    'Z2': [0.60, 0.20, 0.10, 0.08, 0.06],
    'other': [0.30, 0.25, 0.20, 0.15, 0.10],
}
DUP_PAIRS = [(1, 45), (2, 130), (3, 170)]  # filas (clase distinta) que se hacen idénticas


def write_checksums(ext: Path):
    lines = [f'{sha(p)}  {p.relative_to(ext)}' for p in sorted(ext.rglob('*'))
             if p.is_file() and p.name != 'H_checksums.sha256']
    (ext / 'H_checksums.sha256').write_text('\n'.join(lines) + '\n')


def perturb(rng, x, target, B):
    """Transformada x + ruido con A_adj medio ≈ target en un bloque (búsqueda por bisección del ruido)."""
    noise = rng.standard_normal(x.shape)
    xn = np.linalg.norm(x, axis=1, keepdims=True) / np.sqrt(x.shape[1])
    lo, hi = 0.0, 20.0
    for _ in range(40):
        s = 0.5 * (lo + hi)
        y = np.abs(x + s * xn * noise)
        a = ((1 - A.cos_rows(x, y)) / (1 - B)).mean()
        lo, hi = (s, hi) if a < target else (lo, s)
    return y


def make_fixture(root: Path):
    rng = np.random.default_rng(11)
    n = PER_CLASS * 6
    classes = np.repeat(A.CLASSES, PER_CLASS)
    files = [f'{c}{j + 1}.jpg' for c in A.CLASSES for j in range(PER_CLASS)]
    index = pd.DataFrame({'row_index': np.arange(n), 'image_id': [f.split('.')[0] for f in files],
                          'original_filename': files, 'class': classes,
                          'original_relative_path': [f'DatasetOriginalCompleto/{c}/{f}' for c, f in zip(classes, files)]})
    repo = root / 'repo'
    (repo / 'data/manifests').mkdir(parents=True)
    (repo / 'data/derived_metrics/r1').mkdir(parents=True)
    (repo / 'data/summary_tables/r1').mkdir(parents=True)
    index.to_csv(repo / 'data/manifests/original_image_index.csv', index=False)
    # originales con estructura de clase; los pares duplicados son idénticos
    cls_code = np.repeat(np.arange(6), PER_CLASS)
    orig = []
    for d in DIMS:
        centers = np.abs(rng.normal(0, 1, (6, d)))
        x = np.abs(centers[cls_code] + rng.normal(0, 0.8, (n, d)))
        for a, b in DUP_PAIRS:
            x[b] = x[a]
        orig.append(x)
    pi, pj = A.reference_pairs(classes)
    B = np.array([np.median(np.sum(A.normalize(o)[pi] * A.normalize(o)[pj], 1)) for o in orig])
    np.savez(repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz',
             **{f'{b}_random': np.sum(A.normalize(o)[pi] * A.normalize(o)[pj], 1).astype(np.float32)
                for b, o in zip(A.BLOCKS, orig)})
    pd.DataFrame([{'transformation': 'rotation', 'level': lv, 'block': b, 'linear_CKA': 0.5}
                  for lv in A.LEVELS for b in A.BLOCKS]).to_csv(
        repo / 'data/summary_tables/r1/05_CKA_lineal_centrado_por_condicion.csv', index=False)

    def profile(name):
        if name.startswith('gamma'):
            return PROFILES['gamma']
        if name.startswith(('rotation', 'rotR1', 'rotR2')):
            return PROFILES['R']
        if name.startswith(('zoomZ1_', 'zoomZ1r')):
            return PROFILES['Z1']
        if name.startswith('zoomZ2'):
            return PROFILES['Z2']
        return PROFILES['other']

    ext = root / 'extraccion'
    (ext / 'conjuntos').mkdir(parents=True)
    fp = {'script_sha256': 'fixture', 'addendum_sha256': A.ADDENDUM_SHA256, 'criteria_sha256': A.CRITERIA_SHA256,
          'repo': {'commit': 'fixture'}, 'mode': {'smoke_images_per_class': 0, 'n_images': n},
          'environment': {'gpu': 'ninguna (sintético)'}}
    fid = hashlib.sha256(A.canonical(fp).encode()).hexdigest()
    (ext / 'run_fingerprint.json').write_text(json.dumps({'fingerprint_id': fid, 'fingerprint': fp}, indent=2))
    tdig = 'digest-umbrales-sintetico'
    np.savez(ext / 'E_umbrales_casi_cero.npz', fingerprint_id=np.array(fid), thresholds_digest=np.array(tdig))
    gammas = {'low': 0.8, 'medium': 0.65, 'high': 0.55}
    cal = {'levels': {lv: {'gamma': g, 'target_delta_Y': 0.1, 'achieved_delta_Y_global': 0.1,
                           'brightness_delta_Y_per_image_iqr': [0.05, 0.12],
                           'gamma_newly_saturated_component_fraction': 0.0} for lv, g in gammas.items()}}
    (ext / 'D_calibracion_gamma.json').write_text(json.dumps(cal))
    y0, sd0 = rng.uniform(0.3, 0.6, n), rng.uniform(0.15, 0.25, n)
    per = pd.DataFrame({'row_index': np.arange(n), 'image_id': index.image_id, 'Y_mean_original': y0,
                        'Y_std_original': sd0})
    for lv, f in zip(A.LEVELS, [1.2, 1.4, 1.6]):
        per[f'brightness_{lv}_delta_Y'] = y0 * (f - 1)
        per[f'brightness_{lv}_Y_std'] = sd0 * (1 + (f - 1) * 0.6)   # el clipping reduce el aumento de SD
        per[f'gamma_{lv}_delta_Y'] = y0 * (f - 1)
    per.to_csv(ext / 'D_calibracion_gamma_por_imagen.csv', index=False)
    (ext / 'C_rotacion_mascaras.json').write_text(json.dumps(
        {lv: {'angle': a, 'R1_pixels_touched_by_fill0': 0, 'R1_fraction_from_reflection_any': 0.1,
              'R0_fraction_fill_any': 0.1} for lv, a in A.ANGLE.items()}))
    sha_files = [hashlib.sha256(f.encode()).hexdigest() for f in files]
    for a, b in DUP_PAIRS:
        sha_files[b] = sha_files[a]
    pd.DataFrame({'row_index': np.arange(n), 'file': [f'{c}/{f}' for c, f in zip(classes, files)],
                  'sha256': sha_files, 'width': 512, 'height': 384}).to_csv(ext / 'A_jpeg_sha256.csv', index=False)
    pd.DataFrame({'set': ['original_uint8'], 'block': ['Block1'], 'passed': [True]}).to_csv(
        ext / 'F_control_embeddings.csv', index=False)
    (ext / 'registro_extraccion.txt').write_text('sintético\n')
    logits0 = rng.normal(0, 3, (n, 1000)).astype(np.float32)
    s0 = rng.uniform(0.2, 0.6, (n, 5))
    stored = {}
    for name in A.SET_NAMES:
        if name == 'original_uint8':
            gap = orig
        elif name == 'original_float':
            gap = [o * (1 + 1e-6 * rng.standard_normal(o.shape)) for o in orig]
        elif name.startswith('rotR2'):
            gap = None  # se copia de rotation_* más abajo
        else:
            gap = [perturb(rng, o, t, b) for o, t, b in zip(orig, profile(name), B)]
            for a, b_ in DUP_PAIRS:
                for g in gap:
                    g[b_] = g[a]
        if name.startswith('rotR2'):
            gap = [np.asarray(g) for g in stored[f"rotation_{name.split('_')[1]}"]]
        stored_set = gap
        if name.startswith('original'):
            logits, s = logits0, s0
        else:
            a5 = (1 - A.cos_rows(orig[4], gap[4])) / (1 - B[4])
            logits = (logits0 + rng.normal(0, 1, logits0.shape) * (0.3 + 4 * a5[:, None])).astype(np.float32)
            s = np.clip(s0 + rng.normal(0, 0.05, s0.shape), 0, 1)
        z = {'set_name': np.array(name), 'fingerprint_id': np.array(fid), 'thresholds_digest': np.array(tdig),
             'gamma': np.array(gammas[name.split('_')[1]] if name.startswith('gamma') else np.nan),
             'row_index': np.arange(n, dtype=np.int64), 'logits': logits}
        for bi, b in enumerate(A.BLOCKS):
            z[f'gap_{b}'] = gap[bi].astype(np.float32)
            z[f's_exact_{b}'] = s[:, bi].astype(np.float32)
            z[f's_near_{b}'] = (s[:, bi] * 1.1).astype(np.float32)
            z[f'czero_{b}'] = np.zeros((n, DIMS[bi]), np.float32)
        if name.startswith('rotR2'):
            z.update({k: np.zeros(n, np.float32) for k in A.AUX_R2})
        if name.startswith('gamma'):
            g_ = gammas[name.split('_')[1]]
            z.update({'gamma_Y_mean_original': y0.astype(np.float32),
                      'gamma_Y_mean_transformed': (y0 ** g_).astype(np.float32),
                      'gamma_Y_std_original': sd0.astype(np.float32),
                      'gamma_Y_std_transformed': (sd0 * 0.9).astype(np.float32),
                      'gamma_newly_saturated_pixel_fraction': np.zeros(n, np.float32),
                      'gamma_newly_saturated_component_fraction': np.zeros(n, np.float32)})
        np.savez(ext / 'conjuntos' / f'{name}.npz', **z)
        stored[name] = stored_set
    (ext / 'G_B_float_y_rutas.csv').write_text('sintético\n')
    (ext / 'G_equivalencia_R2_R0.csv').write_text('sintético\n')
    man = {'fingerprint_id': fid, 'fingerprint': fp, 'valid_for_analysis': True, 'embedding_gate_passed': True,
           'pixel_gate': {'mode': 'completo', 'n_compared': 12 * n, 'passed': True}}
    (ext / 'H_manifiesto.json').write_text(json.dumps(man, indent=2))
    write_checksums(ext)
    # paquete v2 mínimo
    v2 = root / 'run_v2'
    v2.mkdir()
    groups = [np.flatnonzero(classes == c) for c in A.CLASSES]
    rngb = np.random.default_rng(A.SEED)
    IDX = np.stack([np.concatenate([rngb.choice(g, len(g), replace=True) for g in groups]) for _ in range(N_BOOT)])
    np.save(v2 / 'bootstrap_indices.npy', IDX.astype(np.int32))
    draws = np.stack([np.concatenate([np.sort(rngb.choice(g, DRAW_K, replace=False)) for g in groups])
                      for _ in range(N_DRAWS)])
    pd.DataFrame({'draw': np.repeat(np.arange(N_DRAWS), draws.shape[1]),
                  'image_id': index.image_id.to_numpy()[draws.ravel()],
                  'class': classes[draws.ravel()]}).to_csv(v2 / '10_extracciones_balanceadas_ids.csv', index=False)
    return ext, v2, repo


def run(ext, v2, repo, out, fixture=True):
    cmd = [sys.executable, str(SCRIPT), '--extraction', str(ext), '--v2-package', str(v2), '--repo', str(repo),
           '--out', str(out)] + (['--test-fixture'] if fixture else [])
    return subprocess.run(cmd, capture_output=True, text=True)


tmp = Path(tempfile.mkdtemp(prefix='test_analisis_'))
try:
    base = tmp / 'base'
    base.mkdir()
    ext0, v20, repo0 = make_fixture(base)

    def fresh(tag):
        d = tmp / tag
        shutil.copytree(base, d)
        return d / 'extraccion', d / 'run_v2', d / 'repo', d / 'out'

    # ---------------------------------------------------------------- ejecución completa en modo de prueba
    ext, v2, repo, out = fresh('ok')
    r = run(ext, v2, repo, out)
    check('Ejecución completa con datos sintéticos: termina sin error', r.returncode == 0)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    else:
        man = json.loads((out / 'manifest.json').read_text())
        check('Pares de referencia: error registrado y por debajo de la tolerancia',
              man['checks']['reference_pairs_new_uint8_vs_published_max_abs_err'] <= A.TOL_REFERENCE_PAIRS)
        check('Manifiesto: marcado como prueba y no válido para el manuscrito',
              man['test_fixture'] and not man['valid_for_manuscript'])
        listed = dict(ln.split('  ')[::-1] for ln in (out / 'checksums_outputs.sha256').read_text().splitlines())
        check('Checksums de salida: todos los archivos, incluido el registro cerrado',
              set(listed) == {p.name for p in out.iterdir()} - {'checksums_outputs.sha256'}
              and all(sha(out / k) == v for k, v in listed.items()))
        check('El registro termina antes de los checksums',
              (out / 'run_log.txt').read_text().strip().splitlines()[-1].endswith('checksums de salida.'))
        check('20 tablas CSV (incluida la 06b) y la redacción en Markdown', len(list(out.glob('[01]*.csv'))) == 20
              and (out / '19_redaccion_interpretaciones.md').exists())
        seeds = pd.read_csv(out / '06b_F4_regiones_por_semilla.csv')
        regs = pd.read_csv(out / '06_regiones_estabilidad.csv')
        f4 = regs[regs.family == 'F4']
        cnt = seeds.groupby(['control', 'reference', 'condition']).in_expected_region.sum()
        check('F4: 20 semillas por condición (15 principales + 3 de gamma con B del manuscrito), coherentes con el recuento',
              len(seeds) == 20 * 18 and seeds.groupby(['control', 'reference', 'condition']).size().eq(20).all()
              and all(cnt[(r.control, r.reference, r.condition)] == r.F4_seeds_in_expected_region
                      and len(json.loads(r.F4_seed_peak_regions)) == 20 for r in f4.itertuples()))
        diag = pd.read_csv(out / '02_diagnosticos_extraccion.csv')
        perD = pd.read_csv(ext / 'D_calibracion_gamma_por_imagen.csv')
        gd = diag[diag.section == 'gamma'].set_index('level')
        cols = ['brightness_delta_SD_Y_median', 'brightness_delta_SD_Y_q1', 'brightness_delta_SD_Y_q3',
                'gamma_delta_SD_Y_median', 'gamma_delta_SD_Y_q1', 'gamma_delta_SD_Y_q3',
                'gamma_newly_saturated_pixel_fraction_mean', 'gamma_newly_saturated_pixel_fraction_max',
                'gamma_newly_saturated_component_fraction_mean', 'gamma_newly_saturated_component_fraction_max']
        check('Diagnóstico de gamma: ΔSD(Y) de brillo y gamma (mediana e IQR) y saturación nueva media y máxima',
              gd[cols].notna().all().all() and (gd.brightness_delta_SD_Y_median > 0).all()
              and (gd.gamma_delta_SD_Y_median < 0).all() and np.isclose(
                  gd.loc['medium', 'brightness_delta_SD_Y_median'],
                  np.median(perD['brightness_medium_Y_std'] - perD['Y_std_original']), rtol=1e-5))
        check('Saturación nueva nula: la redacción dice «sin saturación nueva»',
              'sin saturación nueva' in pd.read_csv(out / '19_redaccion_interpretaciones.csv')
              .set_index('control').loc['C1 (gamma)', 'text'] and man['gamma_no_new_saturation'])
        check('Familias de Holm: gamma 12, R1 12, zoom 36 (R2 equivalente, sin familia propia)',
              man['holm_family_sizes'] == {'R1': 12, 'gamma': 12, 'zoom': 36} and man['R2_equivalent_to_R0'])
        cond = pd.read_csv(out / '08_condiciones_criterios.csv')
        prin = cond[cond.reference == 'principal'].set_index(['control', 'level'])
        check('Gamma: pico en B2 y cumple en los 3 niveles',
              all(prin.loc[('gamma', lv), 'F1_peak_block'] == 'Block2' and prin.loc[('gamma', lv), 'cumple']
                  for lv in A.LEVELS))
        check('R1: pico en B3 y cumple a 30° y 60°',
              all(prin.loc[('R1', lv), 'F1_peak_block'] == 'Block3' and prin.loc[('R1', lv), 'cumple']
                  for lv in ['medium', 'high']))
        check('Z1 y Z1r: pico en B5 y cumplen en los 3 factores; Z2 no',
              all(prin.loc[(v, lv), 'cumple'] for v in ['Z1', 'Z1r'] for lv in A.LEVELS)
              and not any(prin.loc[('Z2', lv), 'cumple'] for lv in A.LEVELS))
        check('Gamma también con B del manuscrito como sensibilidad',
              (cond.reference == 'B_manuscrito').sum() == 3)
        interp = pd.read_csv(out / '19_redaccion_interpretaciones.csv').set_index('control')
        check('Redacción: C1 se conserva', interp.loc['C1 (gamma)', 'decision'] == 'se conserva')
        check('Redacción: C3 se conserva', interp.loc['C3 (R1)', 'decision'] == 'se conserva')
        check('Redacción: mecanismo del zoom se conserva', interp.loc['Zoom: mecanismo', 'decision'] == 'se conserva')
        check('Redacción: siempre advierte que la cabeza ImageNet no mide rendimiento ni es accuracy',
              'nunca de accuracy' in interp.loc['Bloque 5 y cabeza ImageNet', 'text'])
        dup = pd.read_csv(out / '01_hallazgo_duplicados.csv')
        check('Duplicados: 3 pares entre clases, con JPEG y GAP idénticos, sin excluirse',
              len(dup) == 3 and dup.cross_class.all() and dup.jpeg_sha256_identical.all()
              and dup.gap_identical_all_blocks.all() and man['duplicates_post_extraction']['n_cross_class'] == 3)
        head = pd.read_csv(out / '16_cabeza_imagenet_por_condicion.csv')
        check('Cabeza ImageNet: 30 condiciones (sin R2) y Holm solo sobre las aplicables',
              len(head) == 30 and not head.condition.str.startswith('rotR2').any()
              and head.mw_p_holm.notna().sum() == head.mw_applicable.sum())
        sp = pd.read_csv(out / '15_sparsity_asociacion.csv')
        check('Sparsity: 33 condiciones × 5 bloques, con réplicas válidas', len(sp) == 165 and (sp.n_boot_valid > 0).all())
        comp = pd.read_csv(out / '09_composicion_clases.csv')
        check('Composición de clases con B fijo y recalculado para cada condición de los controles',
              comp[['conservation_B_fixed', 'conservation_B_recomputed']].notna().all().all() and len(comp) == 18)
        c4 = pd.read_csv(out / '12_C4_y_razon_q.csv')
        check('C4: tres ángulos con D, q, IC y réplicas válidas', len(c4) == 3 and (c4.n_boot == N_BOOT).all())

    # ---------------------------------------------------------------- sin --test-fixture: identidad
    ext, v2, repo, out = fresh('identidad')
    r = run(ext, v2, repo, out, fixture=False)
    check('Sin --test-fixture, la extracción sintética se rechaza antes de analizar (código 2 o 3, sin tablas)',
          r.returncode in (2, 3) and not list(out.glob('*.csv')))

    # ---------------------------------------------------------------- checksums
    ext, v2, repo, out = fresh('byte')
    p = ext / 'conjuntos' / 'zoomZ1_high.npz'
    b = bytearray(p.read_bytes())
    b[-10] ^= 1
    p.write_bytes(bytes(b))
    check('Un byte alterado en un NPZ: código 2', run(ext, v2, repo, out).returncode == 2)
    ext, v2, repo, out = fresh('extra')
    (ext / 'conjuntos' / 'sobrante.npz').write_bytes(b'x')
    check('Un archivo no listado en la extracción: código 2', run(ext, v2, repo, out).returncode == 2)

    # ---------------------------------------------------------------- huella
    ext, v2, repo, out = fresh('huella')
    f = json.loads((ext / 'run_fingerprint.json').read_text())
    f['fingerprint']['mode']['smoke_images_per_class'] = 4
    (ext / 'run_fingerprint.json').write_text(json.dumps(f))
    write_checksums(ext)
    check('Huella modificada (aunque se rehagan los checksums): código 3', run(ext, v2, repo, out).returncode == 3)
    ext, v2, repo, out = fresh('gate')
    m = json.loads((ext / 'H_manifiesto.json').read_text())
    m['embedding_gate_passed'] = False
    (ext / 'H_manifiesto.json').write_text(json.dumps(m))
    write_checksums(ext)
    check('Control de embeddings no aprobado: código 3', run(ext, v2, repo, out).returncode == 3)

    # ---------------------------------------------------------------- metadatos de un NPZ
    ext, v2, repo, out = fresh('meta')
    p = ext / 'conjuntos' / 'gamma_medium.npz'
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d['gamma'] = np.array(0.66)
    np.savez(p, **d)
    write_checksums(ext)
    check('Gamma de un conjunto distinto del calibrado: código 4', run(ext, v2, repo, out).returncode == 4)

    # ---------------------------------------------------------------- bootstrap congelado
    ext, v2, repo, out = fresh('boot')
    I = np.load(v2 / 'bootstrap_indices.npy')
    I[0, 0] = I[0, -1]  # una imagen de cardboard sustituida por una de trash: deja de estar estratificado
    np.save(v2 / 'bootstrap_indices.npy', I)
    check('Bootstrap no estratificado: código 5', run(ext, v2, repo, out).returncode == 5)

    # ---------------------------------------------------------------- pares de referencia fuera de tolerancia
    ext, v2, repo, out = fresh('pares')
    pz = repo / 'data/derived_metrics/r1/baseline_pair_similarity_samples.npz'
    with np.load(pz) as z:
        d = {k: z[k].copy() for k in z.files}
    d['Block3_random'][123] += np.float32(5e-5)
    np.savez(pz, **d)
    check('Pares de referencia que difieren > 1e-5: código 3, también en modo de prueba',
          run(ext, v2, repo, out).returncode == 3)

    # ---------------------------------------------------------------- saturación nueva no nula
    ext, v2, repo, out = fresh('saturacion')
    p = ext / 'conjuntos' / 'gamma_high.npz'
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    sat = np.zeros(len(d['row_index']), np.float32)
    sat[7] = 0.002
    d['gamma_newly_saturated_pixel_fraction'] = sat
    d['gamma_newly_saturated_component_fraction'] = sat / 3
    np.savez(p, **d)
    write_checksums(ext)
    r = run(ext, v2, repo, out)
    txt = (pd.read_csv(out / '19_redaccion_interpretaciones.csv').set_index('control').loc['C1 (gamma)', 'text']
           if r.returncode == 0 else '')
    check('Saturación nueva > 0 en un nivel: la redacción ya no dice «sin saturación nueva»',
          r.returncode == 0 and 'sin saturación nueva' not in txt and 'saturación nueva máxima' in txt)

    # ---------------------------------------------------------------- identidad fijada, fuera del modo de prueba
    try:
        A.validate_checksums(ext0, lambda m: None, test_mode=False)
        check('H_checksums.sha256 distinto del auditado: código 2', False)
    except SystemExit as e:
        check('H_checksums.sha256 distinto del auditado: código 2', e.code == 2)
    try:
        A.validate_repo_files(repo0, lambda m: None, test_mode=False)
        check('Archivos del repositorio distintos de los fijados: código 5', False)
    except SystemExit as e:
        check('Archivos del repositorio distintos de los fijados: código 5', e.code == 5)
    if cli.repo_real:
        got = A.validate_repo_files(cli.repo_real, lambda m: None, test_mode=False)
        check('Repositorio real: índice, pares y tabla CKA coinciden con los SHA-256 fijados',
              got == A.REPO_FILES_SHA256)

    # ---------------------------------------------------------------- carpeta de salida
    ext, v2, repo, out = fresh('salida')
    out.mkdir()
    (out / 'algo.txt').write_text('x')
    check('Carpeta de salida no vacía: código 6', run(ext, v2, repo, out).returncode == 6)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ============================================================================ funciones aisladas
check('Holm: ajuste y NaN conservados', np.allclose(A.holm([0.01, 0.04, np.nan, 0.03])[[0, 1, 3]], [0.03, 0.06, 0.06])
      and np.isnan(A.holm([0.01, np.nan])[1]))
check('r_rb pareada: +1 si todas las diferencias son positivas, −1 si negativas',
      A.paired_rank_biserial(np.array([1., 2, 3])) == 1 and A.paired_rank_biserial(np.array([-1., -2])) == -1)
check('Bandas v1.3', [A.band(x) for x in [0.05, -0.2, 0.3, 0.7]] == ['despreciable', 'pequeña', 'moderada', 'grande'])
lo = np.random.default_rng(0).normal(0, 1, (50, 1000)).astype(np.float32)
m = A.head_metrics(lo, lo)
check('Cabeza ImageNet: sin cambio → top-1 igual, KL 0 y solapamiento top-5 = 1',
      not m['top1_change'].any() and np.allclose(m['KL'], 0) and np.allclose(m['top5_overlap'], 1))
lo2 = lo.copy()
lo2[:, lo.argmax(1)] = -50  # anula la clase top-1 original en todas las filas usadas
m2 = A.head_metrics(lo, lo2)
check('Cabeza ImageNet: anular la clase top-1 → cambio top-1, KL > 0, Δp < 0',
      m2['top1_change'].all() and (m2['KL'] > 0).all() and (m2['delta_p_top1_original'] < 0).all())
x = np.arange(30.0)
IDXs = np.random.default_rng(1).integers(0, 30, (40, 30))
rho, l, h, nv = A.boot_spearman(x, x ** 2, IDXs)
check('Spearman bootstrap: ρ = 1 con IC [1, 1]', rho == 1 and np.isclose(l, 1) and np.isclose(h, 1) and nv == 40)
check('Spearman bootstrap: sin varianza → NaN y 0 réplicas válidas', np.isnan(A.boot_spearman(x, np.ones(30), IDXs)[0]))
# Razón q de CKA: réplicas con D_R0 ≤ 0 excluidas
cka = pd.DataFrame([{'condition': c, 'block': b, 'cka': v} for lv in A.LEVELS for c, vals in
                    [(f'rotation_{lv}', [0.9, 0.8, 0.7, 0.6, 0.5]), (f'rotR1_{lv}', [0.95, 0.9, 0.85, 0.8, 0.75])]
                    for b, v in zip(A.BLOCKS, vals)])
boots = {}
for lv in A.LEVELS:
    for c, d in [(f'rotation_{lv}', np.r_[np.full(8, 0.4), [-0.1, 0.0]]), (f'rotR1_{lv}', np.full(10, 0.2))]:
        boots[(c, 0)], boots[(c, 4)] = d + 0.5, np.full(10, 0.5)
q = A.cka_ratio(cka, boots, 'R1')
check('Razón q: punto 0,2/0,4 = 0,5; réplicas con D_R0 ≤ 0 excluidas (8 de 10 válidas)',
      np.allclose(q.q, 0.5) and (q.n_boot_valid == 8).all() and np.allclose(q.q_ci_low, 0.5))


# ---------------------------------------------------------------- ramas de redacción
def ctab(cumples, peak='profunda', norm='sin discordancia de región'):
    return pd.DataFrame([{'level': lv, 'cumple': bool(c), 'F1_peak_block': 'Block5', 'F1_peak_region': peak,
                          'normalization_label': norm, 'n_informative_other_region': 0,
                          'class_balance_sensitive': False, 'conservation_B_fixed': 1.0,
                          'conservation_B_recomputed': 1.0} for lv, c in zip(A.LEVELS, cumples)])


def c4tab(thr, qlo, qhi):
    return pd.DataFrame([{'variant': 'R1', 'level': lv, 'angle': A.ANGLE[lv], 'D_R1': 0.1, 'D_ci_low': 0.05,
                          'D_ci_high': 0.15, 'threshold_ok': bool(t), 'q': 0.8, 'q_ci_low': a, 'q_ci_high': b,
                          'n_boot_valid': 2000} for lv, t, a, b in zip(A.LEVELS, thr, qlo, qhi)])


gref = pd.DataFrame([{'level': lv, 'region_B_float': 'superficial', 'region_B_manuscript': 'superficial',
                      'cumple_B_float': True, 'cumple_B_manuscript': True, 'same_region': True, 'same_cumple': True,
                      'same_verdict': True} for lv in A.LEVELS])
head = pd.DataFrame([{'condition': 'zoomZ1_low', 'rho_A_adj_B5_vs_KL': 0.05, 'rho_magnitude': 'despreciable',
                      'r_rb_changed_vs_unchanged': 0.1, 'r_rb_magnitude': 'pequeña'}])
spn = pd.DataFrame({'relevant': [False], 'condition': ['x'], 'block': ['Block1'], 'rho_abs': [0.1],
                    'ci_low': [0.0], 'ci_high': [0.2]})
nosat = {lv: {'pixel_mean': 0.0, 'pixel_max': 0.0, 'component_mean': 0.0, 'component_max': 0.0} for lv in A.LEVELS}


def interp(g=(1, 1, 1), r1=(0, 1, 1), z1=(1, 1, 1), z1r_peak='profunda', z2=(0, 0, 0), c4=None, sp=spn):
    t = {'gamma': ctab(g, 'superficial'), 'R1': ctab(r1, 'intermedia'), 'Z1': ctab(z1),
         'Z1r': ctab((1, 1, 1), z1r_peak), 'Z2': ctab(z2)}
    c4 = c4 if c4 is not None else c4tab([1, 1, 1], [0.6] * 3, [1.2] * 3)
    return A.interpretations(t, gref, c4, True, sp, head, nosat).set_index('control').decision.to_dict()


check('C1: 3 niveles → se conserva; 2 → se limita; 0 → se restringe',
      interp()['C1 (gamma)'] == 'se conserva' and interp(g=(1, 0, 1))['C1 (gamma)'] == 'se limita'
      and interp(g=(0, 0, 0))['C1 (gamma)'] == 'se restringe')
check('C3: depende solo de 30° y 60°', interp(r1=(0, 1, 1))['C3 (R1)'] == 'se conserva'
      and interp(r1=(1, 1, 0))['C3 (R1)'] == 'se restringe')
check('C4: conserva / limita / incierto / restringe',
      interp()['C4 (R1)'] == 'se conserva'
      and interp(c4=c4tab([1, 1, 1], [0.3, 0.6, 0.6], [0.45, 1, 1]))['C4 (R1)'] == 'se limita'
      and interp(c4=c4tab([1, 1, 1], [0.3, 0.6, 0.6], [0.8, 1, 1]))['C4 (R1)'] == 'incierto'
      and interp(c4=c4tab([1, 0, 1], [0.6] * 3, [1] * 3))['C4 (R1)'] == 'se restringe')
check('Zoom: conserva; limita si Z1r discrepa o Z1 cumple parcialmente; elimina si Z1 no cumple',
      interp()['Zoom: mecanismo'] == 'se conserva'
      and interp(z1r_peak='intermedia')['Zoom: mecanismo'] == 'se limita'
      and interp(z1=(1, 0, 0))['Zoom: mecanismo'] == 'se limita'
      and interp(z1=(0, 0, 0))['Zoom: mecanismo'] == 'se elimina el lenguaje mecanicista')
check('Sparsity sin asociaciones relevantes: frase fijada', interp()['Sparsity antes del GAP'] == 'sin asociación relevante')
check('canonical() idéntica a la de la extracción v2',
      A.canonical({'b': (1, 2), 'a': 'ñ'}) == '{"a":"ñ","b":[1,2]}')

print()
for k, v in results.items():
    print(('OK    ' if v else 'FALLO ') + k)
n_ok = sum(results.values())
print(f'\n{n_ok}/{len(results)} pruebas superadas')
sys.exit(0 if n_ok == len(results) else 1)
