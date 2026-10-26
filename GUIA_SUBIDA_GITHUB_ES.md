> **Documento histórico (archivado).** Estas instrucciones describen la primera carga de un repositorio vacío. La publicación ya se completó; la rama vigente es `main` y el PR #1 se fusionó el 20 de octubre de 2026. Para reproducibilidad y resultados actuales, consulte [README.md](README.md) y [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

# Guía de subida a GitHub

## Antes de subir

Este paquete está ordenado y validado. El notebook original de las
transformaciones JPG permitió confirmar el encoder y la calidad: Pillow
`Image.save()` con calidad JPEG predeterminada 75. La versión histórica exacta
de Pillow/libjpeg no fue registrada y se declara como tal.

## Subida desde la pantalla vacía del repositorio

1. Descomprime el ZIP definitivo en tu computadora.
2. En GitHub, abre el repositorio vacío.
3. Pulsa **uploading an existing file**.
4. Abre la carpeta descomprimida y arrastra **todo su contenido**, no el ZIP.
   Si GitHub rechaza `.gitignore` por ser un archivo oculto, continúa sin él;
   se puede crear después desde **Add file > Create new file**.
5. Espera a que termine la lista de archivos.
6. En “Commit changes”, escribe: `Initial reproducibility release`.
7. Pulsa **Commit changes**.

Los CSV grandes están comprimidos como `.csv.gz` para respetar los límites de
la carga web. La compresión es sin pérdida y pandas los abre directamente.

