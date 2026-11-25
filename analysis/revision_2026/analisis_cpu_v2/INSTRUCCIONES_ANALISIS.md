# Análisis en CPU de la Adenda 1 v1.2: `analyze_adenda_v2.py`

La versión 2 corrige tres puntos de la auditoría de `analisis_cpu_v1.zip`. La adenda no cambia.

1. Vincula criptográficamente la extracción auditada y los archivos del repositorio que usa. Además, el error de los pares de referencia detiene el análisis si supera 1e-5.
2. Completa los diagnósticos de gamma: ΔSD(Y) de brillo y de gamma, y saturación nueva media y máxima, por píxel y por componente. La frase «sin saturación nueva» depende de esos valores.
3. Guarda la región pico de cada una de las 20 semillas de F4 (tabla 06b).

## Contenido del paquete

- `analyze_adenda_v2.py`: el análisis. Es una sola implementación y no necesita GPU.
- `tests/test_analisis.py`: pruebas con datos sintéticos. No necesitan ningún dato real.
- `tests/regresion_v2.py`: regresión con datos reales. Comprueba que el núcleo del análisis reproduce el paquete v2 aprobado.
- `INSTRUCCIONES_ANALISIS.md`: este archivo.

Requisitos: Python 3.10 o posterior, NumPy, pandas y SciPy. Colab ya los trae. Memoria: unos 3,6 GB de RAM (medido con datos de tamaño real). Duración: entre 6 y 10 minutos en un Colab de CPU (6 minutos en el entorno de Claude, con 2 núcleos).

## Entradas, todas de solo lectura

| Argumento | Qué es |
|---|---|
| `--extraction` | La carpeta `Revision_Adenda1_extraccion_v2` completa: los 47 archivos más `H_checksums.sha256`. |
| `--v2-package` | La carpeta `run_v2` de `revision_robustez_v2.zip`. Usa `bootstrap_indices.npy` y `10_extracciones_balanceadas_ids.csv`. |
| `--repo` | El repositorio en el commit `6672c2c7e4b84a5b0381de0d70476a97438223e5`. |
| `--out` | Una carpeta nueva y vacía. |

## Qué se valida antes de analizar

Si cualquiera de estas comprobaciones falla, el script se detiene sin calcular nada.

1. **Checksums de la extracción.**
   - `H_checksums.sha256` debe ser exactamente el auditado (`409ff94360f15f08e7ad79468863bfb740be04ac57bb020be74cd97a24b770e4`).
   - Se verifican los 47 archivos que lista y se comprueba que no haya ninguno de más ni ausente.

   Si falla: código 2.
2. **Huella y controles de la extracción.**
   - `run_fingerprint.json` debe reproducir su propio `fingerprint_id`, que además tiene que coincidir con el del manifiesto.
   - La extracción debe ser completa (sin `--smoke`), con 2527 imágenes y hecha con el script de extracción v2 (`c103a44e…ec42`).
   - Los hashes de la adenda, los criterios y el commit deben coincidir.
   - El coseno de los 100 000 pares de referencia, recalculado con los originales nuevos, no puede diferir del publicado en más de 1e-5.
   - El control de píxeles tiene que ser completo y estar aprobado, y el de embeddings también aprobado.

   Si algo de esto falla: código 3.
3. **Metadatos de cada NPZ.** En cada uno de los 35 se comprueban el nombre, la huella, las filas 0–2526, el hash de los umbrales de casi cero y el gamma calibrado. Si falla: código 4.
4. **Bootstrap y extracciones congeladas.** Se verifican por SHA-256 frente al paquete v2:
   - `bootstrap_indices.npy` (`c2f882d2…5730`), comprobando además que cada réplica esté estratificada por clase;
   - `10_extracciones_balanceadas_ids.csv` (`1b1ea9ee…85730`), con 200 extracciones de 137 imágenes por clase.

   Si falla: código 5.

5. **Archivos del repositorio.** Se verifican por SHA-256 los tres que usa el análisis. Si falla: código 5.
   - índice: `796f11e8…257f`
   - pares de referencia: `084dd90e…f482`
   - tabla CKA: `819619d0…e367`

Si la carpeta de salida no está vacía, el script se detiene con el código 6.

## Ejecución en Colab (CPU)

```python
from google.colab import drive
drive.mount('/content/gdrive')
BASE = '/content/gdrive/MyDrive/2026-0'
!git clone https://github.com/jorge-necochea/vgg16-trashnet-representational-sensitivity /content/repo
!git -C /content/repo checkout 6672c2c7e4b84a5b0381de0d70476a97438223e5
!unzip -q {BASE}/revision_robustez_v2.zip -d /content/v2          # o la ruta donde lo tengas
!ls /content/v2/revision_robustez_v2/run_v2/bootstrap_indices.npy
!unzip -q {BASE}/analisis_cpu_v2.zip -d /content/analisis
!sha256sum /content/analisis/analyze_adenda_v2.py
!cd /content/analisis && python tests/test_analisis.py --repo-real /content/repo
!python /content/analisis/analyze_adenda_v2.py \
    --extraction {BASE}/Revision_Adenda1_extraccion_v2 \
    --v2-package /content/v2/revision_robustez_v2/run_v2 --repo /content/repo --out /content/analisis_adenda_v2
!cd /content && zip -qr analisis_adenda_v2.zip analisis_adenda_v2
```

Al descomprimir `revision_robustez_v2.zip`, la carpeta queda en `/content/v2/revision_robustez_v2/run_v2`, como confirmó la auditoría. Si en tu caso queda en otra ruta, usa en `--v2-package` la carpeta que contenga `bootstrap_indices.npy`.

## Pruebas

```bash
python tests/test_analisis.py [--repo-real RUTA/repo]
python tests/regresion_v2.py --emb-root RUTA/Revision_R1_Baseline_CKA_VGG16/embeddings \
    --v2-package RUTA/run_v2 --repo RUTA/repo
```

`test_analisis.py` tarda unos 30 segundos y no necesita ningún dato real; son 50 pruebas, 49 sin `--repo-real`. Construye en una carpeta temporal una extracción sintética completa, un paquete v2 mínimo y un repositorio mínimo, y comprueba:

- los códigos de parada 2–6, incluidos el `H_checksums.sha256` distinto del auditado, los archivos del repositorio distintos de los fijados y los pares de referencia fuera de tolerancia;
- la región de las 20 semillas de F4;
- los diagnósticos de gamma, y que la redacción deja de decir «sin saturación nueva» si hay saturación;
- la ejecución completa y sus checksums;
- las familias de Holm;
- los veredictos esperados en perfiles diseñados;
- el hallazgo de duplicados;
- las funciones aisladas;
- todas las ramas de redacción.

La opción `--test-fixture` existe solo para estas pruebas. Con ella, la identidad de la extracción se informa pero no detiene, y las salidas quedan marcadas como no válidas para el manuscrito.

`regresion_v2.py` tarda unos 2 minutos. Aplica las funciones nuevas a las 12 condiciones verificadas: el brillo con las reglas de gamma, la rotación con las de R1 y el zoom con las de Z1. Compara el resultado con las tablas 04, 06, 07, 08, 09 y 11 de la v2. En el entorno de Claude superó las 154 comprobaciones:

- regiones, estabilidad, semillas de F4, F3 no informativa, relevancia, conservación y veredictos: idénticos;
- r_rb, medianas, IC, valores p y CKA: diferencias relativas ≤ 2,1·10⁻⁶. Es el redondeo a 6 cifras con que se guardaron las tablas v2.

## Salidas

| Tabla | Contenido |
|---|---|
| 01 | Hallazgo post-extracción: pares de imágenes idénticas. Para cada par: si el JPEG y el GAP son idénticos, si es entre clases, en cuántas extracciones balanceadas aparece y si está entre los 100 000 pares de referencia. |
| 02–04 | Diagnósticos de la extracción: calibración de gamma, con mediana e IQR de ΔSD(Y) para brillo y para gamma y saturación nueva media y máxima por píxel y por componente; máscaras de rotación; B_float(b), equivalencia de las rutas uint8 y float, y equivalencia R2–R0 recalculada desde los conjuntos. |
| 05–09 | Gamma, R1 (y R2 si no fuera equivalente), Z1, Z1r y Z2: perfiles F1–F4 con IC, regiones y estabilidad, contrastes con p de Holm por familia (gamma 12, R1 12, zoom 36), criterios por condición y composición de clases. |
| 06b | Región pico y medias por bloque de cada una de las 20 semillas de F4, en cada condición. |
| 10 | Gamma con B_float(b) frente al B(b) del manuscrito: región, si cumple y veredicto. |
| 11–12 | CKA por bloque bajo R0 y R1; C4 y la razón q = D_R1 / D_R0 con IC y `n_boot_valid`. |
| 13–15 | Sparsity: descriptivos, s frente a la proporción de canales con GAP cero, y asociación con A_adj (informatividad, relevancia y bandas). |
| 16–18 | Cambio en la salida de la cabeza ImageNet en 30 condiciones: métricas e IC, Spearman por bloque y Spearman exploratorio entre condiciones. |
| 19 | Redacción resultante de cada interpretación, en `.csv` y `.md`. |

Además se generan:

- `manifest.json`, con las entradas, las comprobaciones, las interpretaciones I1–I7 y el hallazgo H1;
- `run_log.txt`;
- `checksums_outputs.sha256`, calculado después de cerrar el registro.

## Imágenes duplicadas entre clases

Hay tres pares de imágenes idénticas en clases distintas:

- glass115 y metal91;
- glass176 y plastic152;
- glass389 y plastic332.

Se registran como hallazgo posterior a la extracción, no preespecificado (H1 en el manifiesto). No se excluyen del análisis principal. El script los detecta de forma independiente, por SHA-256 del JPEG y por igualdad exacta del GAP de los originales. Si encontrara otros pares entre clases, lo avisaría en el registro.

## Qué devolver

`analisis_adenda_v2.zip`, es decir, la carpeta de salida completa. Solo contiene tablas, así que ocupa poco.
