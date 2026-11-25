# CKA de los controles de zoom · Análisis descriptivo E1

El comentario 4 del Revisor 3 pide recalcular, para los controles de zoom, la sensibilidad angular ajustada y también el CKA. La adenda v1.2 solo fijó el CKA para la rotación, así que este cálculo se añade como análisis **descriptivo y no preespecificado**: no lleva veredicto ni regla de decisión.

El script reutiliza `analyze_adenda_v3.py` de `analysis/revision_2026/analisis_cpu_v3/` y exige su SHA-256 auditado (`6e797590…2a7e`). Usa la misma extracción, el mismo bootstrap congelado y la misma implementación del CKA. La versión 1.1 exige que las 15 celdas del CKA de Z0 concuerden con la tabla publicada y detiene la ejecución con código 8 si el error absoluto supera 10⁻⁶ o la referencia está incompleta.

## Celdas de Colab en CPU

Ejecuta la primera celda y selecciona `cka_zoom_v1_1.zip` cuando aparezca el botón de carga. Esta celda crea una carpeta nueva para el cálculo y verifica el script antes de ejecutar las pruebas.

```python
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from google.colab import drive, files

drive.mount('/content/gdrive', readonly=True)
base = Path('/content/gdrive/MyDrive/2026-0')
extraction = base / 'Revision_Adenda1_extraccion_v2'
package_v2 = base / 'revision_robustez_v2.zip'
if not (extraction / 'H_checksums.sha256').is_file():
    raise FileNotFoundError(extraction / 'H_checksums.sha256')
if not package_v2.is_file():
    raise FileNotFoundError(package_v2)

work = Path(tempfile.mkdtemp(prefix='cka_zoom_', dir='/content'))
repo_url = 'https://github.com/jorge-necochea/vgg16-trashnet-representational-sensitivity.git'
repo_main = work / 'repo_main'
repo_base = work / 'repo_base'
subprocess.run(['git', 'clone', '-q', repo_url, str(repo_main)], check=True)
subprocess.run(['git', '-C', str(repo_main), 'checkout', '--detach',
                '3c10f56c1666d2a43fbb773d8fe039d8ee7d0573'], check=True)
subprocess.run(['git', '-C', str(repo_main), 'worktree', 'add', '--detach',
                str(repo_base), '6672c2c7e4b84a5b0381de0d70476a97438223e5'], check=True)
analysis_dir = repo_main / 'analysis/revision_2026/analisis_cpu_v3'
with zipfile.ZipFile(package_v2) as archive:
    archive.extractall(work / 'v2')
run_v2 = work / 'v2/revision_robustez_v2/run_v2'

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError('Selecciona únicamente cka_zoom_v1_1.zip.')
code_dir = work / 'cka_zoom'
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    archive.extractall(code_dir)
script = code_dir / 'cka_zoom_controles.py'
expected = 'eb847a941a074344a7e2e481515724b339a11232461c5c17fbbdbccd5bd2ee77'
if hashlib.sha256(script.read_bytes()).hexdigest() != expected:
    raise ValueError('El script no coincide con el SHA-256 de la versión 1.1.')
subprocess.run(['python', str(code_dir / 'tests/test_cka_zoom.py'),
                '--analysis-dir', str(analysis_dir)], check=True)
```

Cuando aparezca `12/12 pruebas superadas`, ejecuta la segunda celda. El ZIP se genera después de que el cálculo termina y se verifican los checksums de todas las tablas y del registro.

```python
out = work / 'analisis_cka_zoom_v1'
subprocess.run([
    'python', str(script),
    '--analysis-dir', str(analysis_dir),
    '--extraction', str(extraction),
    '--v2-package', str(run_v2),
    '--repo', str(repo_base),
    '--out', str(out),
], check=True)
subprocess.run(['sha256sum', '-c', 'checksums_outputs.sha256'], cwd=out, check=True)
result = shutil.make_archive(str(out), 'zip', root_dir=work, base_dir=out.name)
files.download(result)
```

## Salidas

| Tabla | Contenido |
|---|---|
| `20_cka_zoom_por_bloque.csv` | CKA lineal centrado por bloque, con IC 2,5–97,5 de 2000 réplicas, para el zoom original (Z0) y para Z1, Z1r y Z2 en los tres factores. |
| `21_cka_zoom_descenso_B1_B5.csv` | D = CKA(B1) − CKA(B5) con IC, el bloque de CKA mínimo y los tramos en que el CKA sube. |
| `22_cka_zoom_diferencia_Z0.csv` | CKA(control) − CKA(Z0) por bloque, con IC calculado sobre las mismas réplicas. |

El análisis informa diferencias en la escala de CKA. En la tabla publicada, D_Z0 es 0,013758 a 1,10× y 1,25×, y −0,001263 a 1,50×. Una razón con esos denominadores sería inestable.

## Pruebas

```bash
python tests/test_cka_zoom.py --analysis-dir RUTA/analisis_cpu_v3 \
    [--emb-root RUTA/embeddings --v2-package RUTA/run_v2 --repo RUTA/repo_6672c2c]
```

Las 12 pruebas sintéticas cubren las tres tablas, el IC de la diferencia pareada, los checksums y los códigos de parada 6, 7 y 8.

Las dos pruebas con los NPZ históricos verifican que el CKA de Z0 reproduce la tabla publicada (error máximo 4,8·10⁻¹¹) y los IC de la tabla 07 del paquete v2. En total pasaron 14 de 14 pruebas.
