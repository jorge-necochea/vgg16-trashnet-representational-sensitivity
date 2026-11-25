# Extracción nueva en Colab (Adenda 1 v1.2), script v2

Contenido del paquete:

- `extract_colab_v2.py`
- `INSTRUCCIONES_COLAB.md`
- `tests/test_local.py`
- `tests/stubs/`: sustitutos mínimos de torch, torchvision y tqdm, usados solo si no están instalados.

## Pruebas locales (sin GPU ni datos)

```bash
git clone https://github.com/jorge-necochea/vgg16-trashnet-representational-sensitivity repo
git -C repo checkout 6672c2c7e4b84a5b0381de0d70476a97438223e5
python tests/test_local.py --repo repo --npz-originales embeddings_originales_VGG16_R1.npz
```

`--npz-originales` es opcional y solo añade la comprobación de B(b). Las pruebas generan imágenes sintéticas en una carpeta temporal y la borran al terminar. Usan torch y torchvision reales si están instalados.

Las pruebas no pasan imágenes por la red. Eso lo cubre la prueba reducida en Colab.

## Antes de empezar en Colab

- Usa **Entorno de ejecución → Cambiar tipo → GPU T4**. El notebook 01 se ejecutó en una T4, y la T4 no tiene TF32. Aun así, la garantía de reproducibilidad es el control de embeddings, no la GPU: se desconoce qué GPU usó el notebook 02.
- Sube `extract_colab_v2.py` a `MyDrive/2026-0/`.

## Celda 1: Drive, repositorio fijado y copia local de las imágenes

```python
from google.colab import drive
drive.mount('/content/gdrive')
!git clone https://github.com/jorge-necochea/vgg16-trashnet-representational-sensitivity /content/repo
!git -C /content/repo checkout 6672c2c7e4b84a5b0381de0d70476a97438223e5
!git -C /content/repo rev-parse HEAD
!cp -r /content/gdrive/MyDrive/2026-0/DatasetOriginalCompleto /content/DatasetOriginalCompleto
!cp /content/gdrive/MyDrive/2026-0/extract_colab_v2.py /content/
!sha256sum /content/extract_colab_v2.py
!nvidia-smi --query-gpu=name --format=csv
```

El SHA-256 del script debe coincidir con el registrado. Si el repositorio no está en ese commit, o tiene cambios en archivos versionados, el script se detiene con el código 5.

## Celda 2: variables

```python
BASE = '/content/gdrive/MyDrive/2026-0'
ARGS = (f'--dataset-root /content/DatasetOriginalCompleto '
        f'--png-root {BASE}/TrashNet_transformaciones_completo_png '
        f'--npz-root {BASE}/Revision_R1_Baseline_CKA_VGG16/embeddings '
        f'--repo /content/repo')
```

## Celda 3: prueba reducida (unos 3–5 minutos)

La prueba usa 4 imágenes por clase y recorre todas las etapas y los 35 conjuntos. Sus salidas no sirven para el análisis. Además, su huella impide reutilizarlas en la ejecución completa.

```python
!python /content/extract_colab_v2.py {ARGS} --smoke 4 --out /content/prueba_smoke_v2
!tail -n 30 /content/prueba_smoke_v2/registro_extraccion.txt
```

Comprueba dos cosas:

- El registro termina con «H: registro cerrado».
- En la línea F, el control de embeddings dice APROBADO.

Si dice FALLIDO, para aquí y envíame `registro_extraccion.txt` y `F_control_embeddings.csv`.

## Celda 4: ejecución completa (alrededor de 1 hora)

```python
!python /content/extract_colab_v2.py {ARGS} --out {BASE}/Revision_Adenda1_extraccion_v2
```

Usa una carpeta nueva. El script rechaza, con el código 4:

- una carpeta que no esté vacía y no tenga `run_fingerprint.json`;
- una carpeta cuya huella no coincida con la ejecución actual.

La huella cubre el script, el commit, las imágenes, los NPZ de referencia, el modo, los parámetros, los pesos y el entorno (versiones y GPU).

Si se desconecta el entorno, vuelve a ejecutar las celdas 1, 2 y 4 con la misma carpeta. Solo se reutiliza lo que tenga exactamente la misma huella; cada conjunto comprueba además gamma, los umbrales de casi cero y las filas de imágenes. Si al reconectar tocara otra GPU, o Colab hubiera cambiado la versión de torch, la huella no coincidirá. En ese caso hay que empezar en otra carpeta.

Otras salidas con código:

- **Código 2:** falló el control de píxeles.
- **Código 3:** falló el control de embeddings.

En ninguno de los dos casos se analiza nada. Envíame el registro.

## Qué devolverme

Todo el contenido de `Revision_Adenda1_extraccion_v2` salvo la subcarpeta `conjuntos/`, es decir:

- `run_fingerprint.json`
- `registro_extraccion.txt`
- `H_manifiesto.json`, que incluye el esquema de cada NPZ
- `H_checksums.sha256`
- los archivos de las etapas A a G

Los 35 NPZ de `conjuntos/` ocupan en torno a 1 GB y quedan en Drive para el análisis.
