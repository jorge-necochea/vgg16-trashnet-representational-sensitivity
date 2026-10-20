# Actualización controlada del repositorio — 18 octubre 2026

Este paquete fue construido sobre el ZIP descargado del repositorio original. Conserva todos los archivos previos excepto que sustituye el `.csv.gz` incompleto de R1 y actualiza cuatro archivos documentales/técnicos (`README.md`, `docs/R6_CHECKLIST.md`, `scripts/validate_repository.py`, `checksums.sha256`). Además incorpora `analysis/robustness822/`, `notebooks/originals/` y este documento.

## Qué puede verificarse
- Integridad gzip de todos los archivos comprimidos.
- Cobertura SHA-256 de todos los archivos de esta revisión, excluyendo el propio manifiesto.
- Correspondencia exacta de las primeras 13 columnas entre las métricas principales y las métricas R1 completas (151 620 filas).
- Fórmula de sensibilidad ajustada y presencia de 30 324 rutas en el manifiesto de transformaciones.
- Número esperado de resultados tabulados de robustez para 822 imágenes.

## Qué NO queda demostrado
- Ejecución integral del pipeline VGG-16 desde un entorno nuevo.
- Reproducción independiente de CKA de 822 desde las matrices de embeddings completas; estas no se distribuyen.
- Compatibilidad de instalación de todos los paquetes en un runtime limpio.
- Exactitud retrospectiva de versiones del codificador JPG original.

## Uso
1. Guarda una copia del estado actual de GitHub.
2. Sobre una copia local del repositorio, copia los archivos del ZIP de actualización conservando exactamente las rutas.
3. Ejecuta `python scripts/validate_repository.py`.
4. Realiza commit y push, luego descarga el repositorio publicado y repite la verificación.
5. Solo entonces cambia a completado el estado R6 en la carta al editor.

El archivo `supplementary_materials/Supplementary_Materials_VGG16_TrashNet_Revised.docx` no se ha reemplazado automáticamente: primero debe finalizarse y revisarse su versión editorial integrada.
