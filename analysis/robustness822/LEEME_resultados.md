# Auditoría V2 de robustez (equilibrado822)
822 imágenes, 49320 observaciones por estrategia.
Se incluyen: métricas originales recalculadas, controles B/C, referencias geométricas por estrategia y CKA.
No interpretar diferencias de CKA o bloques máximos del piloto como prueba suficiente de robustez del conjunto completo.
Comparar dentro del mismo conjunto de imágenes; no aplicar el baseline publicado A a estrategias B/C.
Separar efectos del relleno del efecto de deformación 4:3 a 1:1.
Si la reproducción A difiere sustancialmente (>1e-4), detener interpretación y revisar el código.

## Cambios medios máximos frente a los resultados publicados (descriptivos)
- stretch_original: max |delta media cos| = 0.0000001; rel = 0.0000003
- letterbox: max |delta media cos| = 0.0966018; rel = 0.1746500
- no_padding: max |delta media cos| = 0.0155866; rel = 0.0314917