# Resultados del análisis de robustez preespecificado — jimaging-4632369 (paquete v2, tras auditoría)

- Criterios: versión 1.3, SHA-256 del .docx `492a06251c6995dd505ab1624eeaa559ca2cdd486eb7a15b8be0f4f093eddf57` (registrado fuera del .docx).
- Ejecución: 23 nov 2026, `run_robustness_v2.py`, 2000 réplicas bootstrap, 200 extracciones balanceadas. SHA-256 del script, de las entradas y de las salidas: `run_v2/manifest.json` y `run_v2/checksums_outputs.sha256` (28/28 verificados).
- La v2 solo corrige el empaquetado y añade columnas de auditoría (A1–A3). Todas las tablas excepto la 17 y la 21, y los índices bootstrap, son idénticas byte a byte a la v1.
- Este resumen es una lectura de las tablas; la auditoría debe hacerse sobre los CSV.

## Validaciones internas

| Comprobación | Resultado |
|---|---|
| Pares de referencia (semilla 2026) | reproducidos exactamente en los 5 bloques |
| F1 frente al A_adj publicado (151 620 valores) | error máximo 3,4 × 10⁻⁶ |
| CKA frente a la tabla publicada (60 celdas) | error máximo 5,0 × 10⁻¹¹ |
| F4 frente a F1 en el bloque 1 (proyección ortogonal) | error máximo 1,3 × 10⁻⁸ |

## Veredictos por afirmación (tabla 12)

| Afirmación | Veredicto | Detalle |
|---|---|---|
| C1 brillo, región superficial | Robusta | Las tres intensidades son robustas. A 1,2 hay meseta: B2 frente a B1 da r_rb = −0,03, así que la afirmación se formula por región, no por bloque 2 |
| C2 contraste, región superficial | Robusta | F3 no informativa a 1,2 |
| C3 rotación, región intermedia | Parcialmente robusta | 30° y 60°: robustas. 10°: «región estable, efecto no relevante»; B3 frente a B4 da una diferencia pareada mediana de −0,0075 (IC −0,0107 a −0,0027) y r_rb = 0,04, aunque la diferencia de medias es +0,018 |
| C4 CKA bajo rotación decrece con la profundidad | Sostenida | CKA(B1) − CKA(B5): 0,061 [0,058; 0,064], 0,129 [0,121; 0,135] y 0,154 [0,144; 0,161]; monótono en las tres intensidades |
| C5 zoom, región profunda | Parcialmente robusta | 1,10×: robusta (F3 no informativa). 1,25× y 1,50×: F1, F2 y F4 concuerdan; F3 discrepa |
| C6 (C1 o C2) y C5 | Sostenida | La evidencia del zoom depende en parte de la normalización (F3 discrepa a 1,25× y 1,50×); hay que mencionarlo |

Estabilidad bootstrap de F1 para la región informada: 1,000 en las 12 condiciones. La estabilidad tampoco cambia (1,000) al re-estimar B(b) en cada réplica: B5 = 0,4085 [0,4040; 0,4136] frente a 0,4083 informado. Friedman sobre A_adj: W de Kendall entre 0,29 (zoom 1,50×) y 0,83 (brillo 1,6), con todas las p de Holm < 0,001.

### Por qué discrepa F3 en el zoom (para la auditoría)

A 1,50×, la proporción de imágenes por debajo del percentil 2,5 de las distancias entre pares aleatorios fue del 74,7 %, 94,6 %, 98,7 %, 99,2 % y 99,8 % en los bloques 1 a 5, respectivamente. La media de P es 0,0248 en B1 y 0,0004 en B5. Así, el percentil se concentra cada vez más cerca de su piso en los bloques profundos. Como la distribución de pares del bloque 1 es muy estrecha, desplazamientos pequeños ocupan ahí percentiles algo mayores. La regla de no informatividad (más del 50 % de las imágenes en el mínimo en al menos 3 bloques) no se activó a 1,25× ni a 1,50×. Por tanto, según los criterios congelados, F3 cuenta como discrepante. No se reclasifica.

## Sensibilidades

- **Composición de clases:** la región pico se conserva en el 100 % de las 200 extracciones en las 12 condiciones, tanto con B(b) fijo como recalculado. El subconjunto de las imágenes 1–137 da las mismas regiones.
- **Variante sin canales mayormente nulos** (46 canales retirados en B5 y 1 en B4): mismas regiones en las 12 condiciones.
- **F4:** 20 de 20 semillas en la región informada, salvo rotación 10° (17 de 20).

## Clipping (R3-2, exploratorio)

- Spearman entre la proporción de píxeles recién saturados y A_adj: ρ entre 0,41 y 0,79 en todos los bloques e intensidades. Las 30 p de Holm son < 0,001.
- Desviación D1: el ajuste literal por residuos centrados no puede cambiar las medias por bloque (diferencia 0 por construcción). Como complemento no preespecificado se informan perfiles por tercil de clipping (tablas 23–24). En el tercil de menor clipping, el pico pasa de B2 a B1 a 1,2 y 1,4, dentro de la región superficial, y se mantiene en B2 a 1,6.
- Lectura: la región superficial resiste la estratificación por clipping; el pico exacto en el bloque 2, no.

## Sparsity (exploratorio)

- No hay canales muertos en ningún bloque. En B5, 507 de 512 canales son nulos en alguna imagen y 46 lo son en más del 50 % de las imágenes.
- Dimensionalidad efectiva (participation ratio): 4,5; 3,7; 6,1; 12,0; 31,8.
- Asociación entre A_adj y |z_t − z_0| (A2, regla de auditoría no preespecificada: no informativa si ≥ 90 % de las imágenes tienen |z_t − z_0| = 0 o hay < 95 % de réplicas válidas). Los bloques 1–3 son no informativos en las 12 condiciones. Los bloques 4–5 son informativos en las 12, todas con 2000 réplicas válidas: llega a ρ = 0,46 (rotación 10°, B4); con brillo, ρ va de 0,18 a 0,42; con contraste y zoom suele ser ≤ 0,12.

## Clases, bloque 5 (exploratorio)

- **K1, sostenida solo para zoom bajo y medio.** Cardboard tiene la mediana más alta bajo zoom 1,10× en el 100 % de las extracciones y bajo 1,25× en el 95,5 % con el denominador aleatorio; en el 100 % de ambas con el denominador same-class. A 1,50× baja al 66,5 % (aleatorio) y al 94 % (same-class).
- **K2.** Zoom 1,50×, denominador aleatorio, Holm sobre los 5 contrastes prefijados: frente a paper, r_rb = 0,06 y p de Holm = 0,105; frente a trash, r_rb = 0,15 y p de Holm = 0,017. El manuscrito, con Holm sobre 15 pares, informaba que trash no era significativo. Con el denominador same-class, todas las diferencias son significativas (r_rb de 0,16 a 0,46). Conclusión de auditoría: no afirmar que cardboard sea indistinguible de trash bajo zoom alto, porque el resultado depende del denominador y de la familia de corrección.
- **η²_H:** de 0,027 a 0,090 con el denominador aleatorio; de 0,052 a 0,121 con el same-class.

## K3 y control de calidad de máscaras (exploratorio)

- **Con las 2527 imágenes K3 se sostiene** en los tres niveles: cardboard frente a cada otra clase, r_rb de 0,39 a 0,78, todas las p de Holm < 0,001.
- **Control de calidad:** el bounding box toca un borde de la imagen en el 75–91 % de las imágenes de cada clase (cardboard: 89,6 %). En cardboard, el 42,9 % de los bounding boxes ocupa ≥ 0,95 de la imagen.
- **Sensibilidad que excluye esas imágenes:** quedan 437 (cardboard 42, glass 91, metal 100, paper 55, plastic 121, trash 28). K3 no se sostiene en ningún nivel, y en los niveles medio y alto tampoco se conservan los tamaños de efecto (por ejemplo, a 1,50×: r_rb −0,07 frente a glass, 0,00 frente a paper y 0,07 frente a plastic).
- **Decisión de auditoría:** se retira del resumen y de las conclusiones. En resultados: «Cardboard mostró la mayor pérdida ajustada en el análisis completo; sin embargo, el resultado no se mantuvo al excluir máscaras que tocaban los bordes o cubrían al menos el 95 % de la imagen, por lo que se considera exploratorio e inconcluso».

## Desviaciones e interpretaciones registradas (manifest.json)

- D1: clipping, explicado arriba.
- I1: la «diferencia pareada» de los criterios de dirección se implementó como la mediana de las diferencias por imagen, igual que en la regla de tamaño de efecto.
- I2: la región y la estabilidad de F2–F4 se juzgan respecto de la región informada en el manuscrito enviado.
- Ajustes de auditoría de la v2: A1, checksums calculados después de cerrar el registro; A2, n_boot_valid y regla de no informatividad en la tabla 21; A3, imágenes retenidas por clase en la tabla 17. Los perfiles por tercil de clipping (D1) siguen rotulados como exploratorios.

## Archivos

Script: `run_robustness_v2.py` (incluido también `run_robustness_v1.py` para trazabilidad). Tablas: `run_v2/01`–`25`. Registro de ejecución: `run_v2/run_log.txt`. Índices bootstrap: `run_v2/bootstrap_indices.npy`. Identificadores de las extracciones: `run_v2/10_extracciones_balanceadas_ids.csv`. Script de verificación de entradas: `verify_inputs.py`.
