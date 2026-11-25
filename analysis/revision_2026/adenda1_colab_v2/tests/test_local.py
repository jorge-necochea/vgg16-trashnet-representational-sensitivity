"""Pruebas locales de extract_colab_v2.py, reproducibles desde el paquete.

No necesitan GPU ni datos de TrashNet: generan imágenes sintéticas en una carpeta temporal.
Si torch y torchvision están instalados, se usan los reales; si no, los sustitutos de tests/stubs
(implementan con PIL y NumPy solo las funciones que usan estas pruebas, igual que las rutas PIL
de torchvision 0.26). Las rutas que pasan imágenes por la red no se prueban aquí: las cubre
la prueba reducida en Colab (--smoke).

Uso:
    python tests/test_local.py --repo RUTA_AL_REPOSITORIO [--npz-originales embeddings_originales_VGG16_R1.npz]
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True, type=Path, help='Clon del repositorio en el commit fijado')
ap.add_argument('--npz-originales', type=Path, default=None,
                help='Opcional: NPZ verificado de originales, para comprobar B(b)')
args = ap.parse_args()

if importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'):
    backend = 'torch y torchvision reales'
else:
    sys.path.insert(0, str(HERE / 'stubs'))
    backend = 'sustitutos de tests/stubs (PIL y NumPy)'
if not importlib.util.find_spec('tqdm'):
    sys.path.insert(0, str(HERE / 'stubs'))
sys.path.insert(0, str(HERE.parent))
import extract_colab_v2 as E  # noqa: E402

print(f'Backend: {backend}')
results = {}


def check(name, cond):
    results[name] = bool(cond)


E.load_repo_transform(args.repo)
tmp = Path(tempfile.mkdtemp(prefix='test_colab_v2_'))
log = E.Logger()
log.attach(tmp / 'log.txt')

# ---------------------------------------------------------------- imágenes sintéticas
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:E.H, 0:E.W]
data = tmp / 'DatasetOriginalCompleto'
paths = []
for c, cls in enumerate(E.CLASSES):
    (data / cls).mkdir(parents=True)
    for k in range(2):
        base = np.stack([(xx * (1 + c) + yy * (k + 1)) % 256, (yy * 2 + 40 * c) % 256, (xx + yy) % 256], -1)
        noisy = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
        noisy[50:120, 60:200] = 255  # zona saturada
        p = data / cls / f'{cls}{k}.jpg'
        Image.fromarray(noisy).save(p, quality=90)
        paths.append(str(p))
index = pd.DataFrame({'class': [Path(p).parent.name for p in paths]})
img = Image.open(paths[0]).convert('RGB')

# ---------------------------------------------------------------- 1. repositorio fijado
info = E.repo_info(args.repo)
if info['commit'] is None:
    print('Aviso: la carpeta del repositorio no es un clon de git; se omite la comprobación del commit.')
else:
    check('Repositorio en el commit fijado y sin cambios', E.repo_problems(info) == [])
bad = dict(info, commit='0' * 40, tracked_changes=' M x')
check('Se detectan commit distinto y cambios locales', len(E.repo_problems(bad)) >= 2)
check('SHA-256 de generate_transformations_png.py = fijado',
      info['generate_transformations_png_sha256'] == E.GEN_SCRIPT_SHA256)

# ---------------------------------------------------------------- 2. conjuntos y esquema
sets = E.build_sets()
by = {s['name']: s for s in sets}
check('35 conjuntos; los 13 primeros son originales uint8 y las 12 existentes',
      len(sets) == 35 and sets[0]['name'] == 'original_uint8' and sum(s['group'] == 'existing' for s in sets[:13]) == 12)
check('Medidas auxiliares con nombre: R2 y gamma',
      set(E.AUX_FIELDS['R2']) <= E.expected_keys(by['rotR2_low'])
      and set(E.AUX_FIELDS['gamma']) <= E.expected_keys(by['gamma_high'])
      and not set(E.AUX_FIELDS['R2']) & E.expected_keys(by['rotR1_low']))
check('Sin clave «extra» en ningún esquema', all('extra' not in E.expected_keys(s) for s in sets))

# ---------------------------------------------------------------- 3. zoom
for z in E.ZOOMS.values():
    left, top, cw, ch = E.zoom_box(z)
    box = np.asarray(img)[top:top + ch, left:left + cw]
    z1, z1r, z2 = (np.asarray(f(img, z)) for f in (E.op_z1, E.op_z1r, E.op_z2))
    inside = np.zeros((E.H, E.W), bool)
    inside[top:top + ch, left:left + cw] = True
    check(f'Z1 {z}: caja intacta y periferia (124, 116, 104)',
          np.array_equal(z1[inside].reshape(ch, cw, 3), box) and (z1[~inside] == E.FILL_MEAN_RGB).all())
    check(f'Z1r {z}: caja intacta y periferia reflejada',
          z1r.shape == (E.H, E.W, 3) and np.array_equal(z1r[top:top + ch, left:left + cw], box)
          and np.array_equal(z1r[top - 1, left:left + cw], box[1]))
    check(f'Z2 {z}: 512 × 384', z2.shape == (E.H, E.W, 3))
check('Caja a 1,50×: 341 × 256, márgenes 85/86 y 64', E.zoom_box(1.5) == (85, 64, 341, 256))
check('Z0 del repositorio = recorte de la caja + resize bilineal', np.array_equal(
    np.asarray(E.op_existing(img, 'zoom', 1.5)),
    np.asarray(img.crop((85, 64, 426, 320)).resize((512, 384), Image.Resampling.BILINEAR))))

# ---------------------------------------------------------------- 4. rotación
geo = E.rotation_mask_checks()
check('R1 sin píxeles de fill=0 a 10°, 30° y 60°', all(geo[lv]['R1_pixels_touched_by_fill0'] == 0 for lv in E.LEVELS))
for a in E.ANGLES.values():
    aux = E.r2_aux(img, a)
    check(f'R1 y R2 de 512 × 384 a {a}°; R2 frente a R0 con 5 medidas finitas',
          np.asarray(E.op_r1(img, a)).shape == np.asarray(E.op_r2(img, a)).shape == (E.H, E.W, 3)
          and aux.shape == (5,) and np.isfinite(aux).all())
    print(f'  {a}°: R2 frente a R0: MAE R/G/B {aux[:3].round(3)}, máx. {aux[3]:.0f}, % distintos {aux[4]:.2f}; '
          f'R1 por reflexión {geo[[k for k in E.LEVELS if E.ANGLES[k] == a][0]]["R1_fraction_from_reflection_any"]:.4f}')

# ---------------------------------------------------------------- 5. gamma
arr = np.asarray(img)
h = E.channel_hist(arr)
for g in (0.5, 0.8):
    direct = (E.luminance((arr / 255.0) ** g) - E.luminance(arr / 255.0)).mean()
    check(f'ΔY por histograma = cálculo directo (γ = {g})', abs(float(E.delta_y_from_hist(h, g)) - direct) < 1e-12)
g = E.calibrate_gamma(h, 0.05)
check('Bisección: |ΔY(γ) − objetivo| < 1e-5', abs(float(E.delta_y_from_hist(h, g)) - 0.05) < 1e-5)
cal, cal_img = E.gamma_calibration(paths, log, workers=0)
L = cal['levels']
check('0 < γ_alto < γ_medio < γ_bajo < 1', 0 < L['high']['gamma'] < L['medium']['gamma'] < L['low']['gamma'] < 1)
check('ΔY logrado = objetivo en los 3 niveles',
      all(abs(L[lv]['achieved_delta_Y_global'] - L[lv]['target_delta_Y']) < 1e-5 for lv in E.LEVELS))
manual = np.mean([E.luminance(np.asarray(E.op_existing(Image.open(p).convert('RGB'), 'brightness', 1.4)) / 255.0).mean()
                  - E.luminance(np.asarray(Image.open(p).convert('RGB')) / 255.0).mean() for p in paths])
check('Objetivo ΔY (1,4) = cálculo directo sobre el brillo del repositorio', abs(manual - L['medium']['target_delta_Y']) < 1e-12)
check('Saturación nueva por histograma = 0 en los 3 niveles',
      all(L[lv]['gamma_newly_saturated_component_fraction'] == 0 for lv in E.LEVELS))
x01 = (arr.transpose(2, 0, 1) / np.float32(255)).astype(np.float32)
ga = E.gamma_aux(x01, x01 ** np.float32(L['high']['gamma']))
check('Medidas de gamma por imagen: 6 valores; saturación nueva (píxel y componente) = 0',
      ga.shape == (6,) and ga[4] == 0 and ga[5] == 0 and ga[1] > ga[0])
fake_x = np.full((3, 2, 2), 0.5, np.float32)
fake_y = fake_x.copy()
fake_y[0, 0, 0] = 1.0
gb = E.gamma_aux(fake_x, fake_y)
check('La proporción de saturación nueva se calcula (1 de 4 píxeles, 1 de 12 componentes)',
      gb[4] == 0.25 and abs(gb[5] - 1 / 12) < 1e-12)

# ---------------------------------------------------------------- 6. umbrales de casi cero
vals = rng.lognormal(0, 2, size=(3, 200_000))
vals[2] = 0
counts = np.zeros((3, E.NBINS), np.int64)
for c in range(3):
    v = vals[c][vals[c] > 0]
    b = np.clip(np.floor((np.log10(v) - E.LOG_LO) / (E.LOG_HI - E.LOG_LO) * E.NBINS), 0, E.NBINS - 1).astype(int)
    counts[c] = np.bincount(b, minlength=E.NBINS)
med, thr, live = E.thresholds_from_counts(counts)
check('Mediana por histograma ≈ exacta (error relativo < 0,3 %)', abs(med[0] / np.median(vals[0]) - 1) < 3e-3)
check('Canal sin valores no nulos: muerto, umbral NaN', (not live[2]) and np.isnan(thr[2]) and live[0])
check('Umbral = 1 % de la mediana', np.allclose(thr[:2], 0.01 * med[:2]))
thr5 = {b: np.linspace(0.01, 1, E.DIMS[b]) for b in E.BLOCKS}
live5 = {b: np.ones(E.DIMS[b], bool) for b in E.BLOCKS}
d1 = E.thresholds_digest(thr5, live5)
thr5b = {b: v.copy() for b, v in thr5.items()}
thr5b['Block3'][7] *= 1.0000001
check('La huella de umbrales cambia con cualquier umbral', d1 != E.thresholds_digest(thr5b, live5))

# ---------------------------------------------------------------- 7. control de embeddings
labels = pd.read_csv(args.repo / 'data/manifests/original_image_index.csv')['class'].to_numpy()
emb = {b: np.abs(rng.normal(0, 1, (E.N_EXPECTED, E.DIMS[b]))).astype(np.float32) for b in E.BLOCKS}
tr = {b: (emb[b] * 0.9 + np.abs(rng.normal(0, 0.3, emb[b].shape))).astype(np.float32) for b in E.BLOCKS}
old = {'original_uint8': emb, 'brightness_low': tr}
_, ok = E.embedding_gate({'original_uint8': {b: emb[b].copy() for b in E.BLOCKS},
                          'brightness_low': {b: tr[b].copy() for b in E.BLOCKS}}, old, labels)
check('Control de embeddings: idénticos aprueban', ok)
_, ok = E.embedding_gate({'original_uint8': {b: emb[b] * np.float32(1 + 2e-5) for b in E.BLOCKS},
                          'brightness_low': tr}, old, labels)
check('Control de embeddings: norma +2e-5 falla', not ok)
_, ok = E.embedding_gate({'original_uint8': emb,
                          'brightness_low': {b: tr[b] + np.float32(1e-2) * rng.normal(0, 1, tr[b].shape).astype(np.float32)
                                             for b in E.BLOCKS}}, old, labels)
check('Control de embeddings: transformada perturbada falla', not ok)
sel = pd.Series(labels).groupby(labels, sort=False).head(3).index.to_numpy()
_, ok = E.embedding_gate({'original_uint8': {b: emb[b][sel] for b in E.BLOCKS}}, old, labels, sel)
check('Control de embeddings con subconjunto (--smoke): filas correctas aprueban', ok)
_, ok = E.embedding_gate({'original_uint8': {b: emb[b][sel[::-1]] for b in E.BLOCKS}}, old, labels, sel)
check('Control de embeddings con subconjunto: filas desordenadas fallan', not ok)
if args.npz_originales:
    with np.load(args.npz_originales) as z:
        orig = {b: z[b] for b in E.BLOCKS}
    gate, _ = E.embedding_gate({'original_uint8': orig}, {'original_uint8': orig}, labels)
    Bs = gate.B_reference.round(4).tolist()
    print('  B(b) con el NPZ verificado:', Bs)
    check('B(b) = 0,8445 / 0,8701 / 0,7909 / 0,6434 / 0,4083', Bs == [0.8445, 0.8701, 0.7909, 0.6434, 0.4083])
else:
    print('Aviso: sin --npz-originales; se omite la comprobación de B(b).')

# ---------------------------------------------------------------- 8. control de píxeles
png_root = tmp / 'png'
for i, p in enumerate(paths):
    for t, levels in E.EXISTING.items():
        for lv, v in levels.items():
            d = png_root / f'{t}_{lv}' / index['class'][i]
            d.mkdir(parents=True, exist_ok=True)
            E.op_existing(Image.open(p).convert('RGB'), t, v).save(d / f'{Path(p).stem}.png')
pix = E.pixel_gate(paths, index, png_root, log, workers=0, smoke=False)
check('Control de píxeles completo: 144 comparaciones idénticas', pix['mode'] == 'completo' and pix['passed']
      and pix['n_compared'] == 144)
check('En modo --smoke el control se rotula «prueba_reducida»',
      E.pixel_gate(paths, index, png_root, log, workers=0, smoke=True)['mode'] == 'prueba_reducida')
victim = png_root / 'zoom_high' / 'glass' / 'glass1.png'
a = np.asarray(Image.open(victim).convert('RGB')).copy()
a[10, 10, 0] = (int(a[10, 10, 0]) + 1) % 256
Image.fromarray(a).save(victim)
pix = E.pixel_gate(paths, index, png_root, log, workers=0, smoke=False)
check('Un solo valor distinto en un PNG hace fallar el control y se identifica',
      (not pix['passed']) and len(pix['failures']) == 1 and pix['failures'][0]['file'] == 'glass1.jpg'
      and pix['failures'][0]['transformation'] == 'zoom')
victim.unlink()
check('Con PNG incompletos el control queda como diagnóstico',
      E.pixel_gate(paths, index, png_root, log, workers=0, smoke=False)['mode'] == 'diagnostico')

# ---------------------------------------------------------------- 9. archivos de conjunto y reutilización
n = 5
rows_sel = np.arange(n)
gammas = {'low': 0.7, 'medium': 0.5, 'high': 0.4}
spec = by['gamma_high']


def fake_set(spec, fid='F1', tdig='T1', gam=gammas, rows=rows_sel, drop=None):
    o = {f'gap_{b}': np.ones((n, E.DIMS[b]), np.float32) for b in E.BLOCKS}
    o.update({f'czero_{b}': np.zeros((n, E.DIMS[b]), np.float32) for b in E.BLOCKS})
    o.update({f's_exact_{b}': np.zeros(n, np.float32) for b in E.BLOCKS})
    o.update({f's_near_{b}': np.zeros(n, np.float32) for b in E.BLOCKS})
    o['logits'] = np.zeros((n, 1000), np.float32)
    kind = E.aux_kind(spec)
    if kind:
        o.update({k: np.zeros(n, np.float32) for k in E.AUX_FIELDS[kind]})
    o.update(E.set_metadata(spec, rows, gam, fid, tdig))
    if drop:
        o.pop(drop)
    return o


sd = tmp / 'sets'
sd.mkdir()
E.save_set(sd / 'g.npz', fake_set(spec))
check('Conjunto que coincide en todo: se reutiliza', E.valid_set_file(sd / 'g.npz', spec, rows_sel, gammas, 'F1', 'T1')[0])
check('No quedan temporales tras guardar', not list(sd.glob('*.tmp.npz')))
cases = {
    'otra huella': ('F2', 'T1', gammas, rows_sel, spec),
    'otros umbrales': ('F1', 'T2', gammas, rows_sel, spec),
    'otro gamma': ('F1', 'T1', dict(gammas, high=0.41), rows_sel, spec),
    'otras imágenes': ('F1', 'T1', gammas, rows_sel + 1, spec),
    'otro conjunto': ('F1', 'T1', gammas, rows_sel, by['gamma_low']),
}
for why, (f_, t_, g_, r_, s_) in cases.items():
    check(f'Conjunto con {why}: no se reutiliza', not E.valid_set_file(sd / 'g.npz', s_, r_, g_, f_, t_)[0])
E.save_set(sd / 'g2.npz', fake_set(spec, drop='gamma_newly_saturated_pixel_fraction'))
check('Conjunto sin una medida del esquema: no se reutiliza',
      not E.valid_set_file(sd / 'g2.npz', spec, rows_sel, gammas, 'F1', 'T1')[0])

# ---------------------------------------------------------------- 10. huella de la carpeta de salida
fp = {'script_sha256': 'a', 'mode': {'smoke_images_per_class': 4}, 'environment': {'gpu': 'Tesla T4'}}
o1 = tmp / 'out1'
o1.mkdir()
fid1, resumed = E.check_out_folder(o1, fp)
check('Carpeta vacía: se crea la huella', (o1 / 'run_fingerprint.json').exists() and not resumed)
fid2, resumed = E.check_out_folder(o1, dict(fp))
check('Misma huella: se permite reanudar', resumed and fid2 == fid1)
for label, fp_other in {'modo completo sobre carpeta de --smoke': dict(fp, mode={'smoke_images_per_class': 0}),
                        'otro script': dict(fp, script_sha256='b'),
                        'otra GPU': dict(fp, environment={'gpu': 'NVIDIA L4'})}.items():
    try:
        E.check_out_folder(o1, fp_other)
        check(f'Huella distinta ({label}): se rechaza', False)
    except SystemExit as e:
        check(f'Huella distinta ({label}): se rechaza con código 4', e.code == 4)
check('La diferencia de huellas nombra el campo', E.fingerprint_diff(fp, dict(fp, mode={'smoke_images_per_class': 0}))
      == ['mode.smoke_images_per_class'])
o2 = tmp / 'out2'
o2.mkdir()
(o2 / 'algo.npz').write_bytes(b'x')
try:
    E.check_out_folder(o2, fp)
    check('Carpeta no vacía y sin huella: se rechaza', False)
except SystemExit as e:
    check('Carpeta no vacía y sin huella: se rechaza con código 4', e.code == 4)

# ---------------------------------------------------------------- 11. resúmenes posteriores (B_float y R2–R0)
n = 30
sd2 = tmp / 'post'
sd2.mkdir()
base = {b: np.abs(rng.normal(0, 1, (n, E.DIMS[b]))).astype(np.float32) for b in E.BLOCKS}


def write(name, emb, aux=None):
    o = {f'gap_{b}': emb[b] for b in E.BLOCKS}
    o.update(aux or {})
    np.savez(sd2 / f'{name}.npz', **o)


write('original_uint8', base)
write('original_float', {b: base[b] * 1.0001 for b in E.BLOCKS})
for lv in E.LEVELS:
    write(f'rotation_{lv}', base)
    write(f'rotR2_{lv}', base, {k: np.full(n, v, np.float32) for k, v in
                                zip(E.AUX_FIELDS['R2'], [0.2, 0.3, 0.4, 3, 1.0])})
post = E.post_summaries(sd2, np.array(['a', 'b', 'c'] * 10), tmp, log)
bf = pd.read_csv(tmp / 'G_B_float_y_rutas.csv')
check('B_float = B_uint8 si la ruta float solo reescala', np.allclose(bf.B_float, bf.B_uint8_recomputed))
check('R2 idéntico en embeddings y MAE ≤ 1: equivalente', post['R2_equivalent_to_R0'])
write('rotR2_high', base, {k: np.full(n, v, np.float32) for k, v in zip(E.AUX_FIELDS['R2'], [0.2, 1.3, 0.4, 3, 1.0])})
check('Un MAE por componente > 1: no equivalente', not E.post_summaries(sd2, np.array(['a', 'b', 'c'] * 10), tmp,
                                                                        log)['R2_equivalent_to_R0'])

# ---------------------------------------------------------------- 12. registro con búfer
lg = E.Logger()
lg('antes')
lg.attach(tmp / 'log2.txt')
lg('después')
check('El registro guarda las líneas previas a validar la carpeta', [ln.split('] ')[1] for ln in
                                                                     (tmp / 'log2.txt').read_text().splitlines()]
      == ['antes', 'después'])

shutil.rmtree(tmp)
print()
for k, v in results.items():
    print(('OK    ' if v else 'FALLO ') + k)
n_ok = sum(results.values())
print(f'\n{n_ok}/{len(results)} pruebas superadas ({backend})')
sys.exit(0 if n_ok == len(results) else 1)
