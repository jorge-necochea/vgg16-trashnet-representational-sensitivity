# Redacción resultante (Adenda 1 v1.2)

## C1 (gamma): se conserva

El brillo actúa en la región superficial, no solo por el clipping: la respuesta superficial persiste con gamma, sin saturación nueva, en los tres niveles.

Detalle: Bloque pico con gamma: low: Block1; medium: Block2; high: Block2. Saturación nueva (media/máxima por imagen; píxel | componente): low: 0/0 | 0/0; medium: 0/0 | 0/0; high: 0/0 | 0/0. Transversal: low: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado); meseta (se formula solo por región) | medium: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado) | high: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado)

## C1 (gamma): referencia: no cambia las conclusiones

La elección de referencia (B_float frente a B del manuscrito) no cambia las conclusiones: misma región pico y mismo veredicto en los tres niveles.

Detalle: low: región superficial/superficial, cumple True/True; medium: región superficial/superficial, cumple True/True; high: región superficial/superficial, cumple True/True

## C3 (R1): se conserva

La rotación actúa en la región intermedia; el efecto es compatible con la rotación sin bordes negros, no exclusivo de la orientación.

Detalle: 10°: cumple False (informativo); 30°: True; 60°: True. Transversal: low: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado) | medium: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado) | high: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado)

## C4 (R1): se conserva

Bajo rotación, el CKA es menor en B5 que en B1 en los tres ángulos, también sin bordes negros; el descenso se formula entre los bloques extremos.

Detalle: 10°: D=0.041 [0.039, 0.043], q=0.67 [0.65, 0.69], réplicas válidas 2000; 30°: D=0.111 [0.104, 0.117], q=0.86 [0.84, 0.88], réplicas válidas 2000; el CKA puntual sube en Block3→Block4 (0.936 → 0.943); 60°: D=0.142 [0.134, 0.149], q=0.92 [0.90, 0.95], réplicas válidas 2000

## Zoom: mecanismo: se elimina el lenguaje mecanicista

El mecanismo específico del efecto profundo del zoom no queda identificado.

Detalle: low: Z1 False (profunda), Z1r profunda, Z2 False; medium: Z1 False (profunda), Z1r profunda, Z2 False; high: Z1 False (intermedia), Z1r profunda, Z2 False. Z1 y Z1r coinciden en la región: False. Transversal Z1: low: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado); meseta (se formula solo por región) | medium: discordancia aislada; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado); meseta (se formula solo por región) | high: sin discordancia de región; estable ante el balance de clases (conservación de la región observada 1.00 con B fijo, 1.00 con B recalculado) respecto de la región observada (intermedia), no de la esperada (profunda); meseta (se formula solo por región)

## Bloque 5 y cabeza ImageNet: descriptivo (sin veredicto)

La sensibilidad representacional no equivale a vulnerabilidad de la tarea, y la cabeza ImageNet no mide rendimiento en TrashNet; se habla de «cambio en la salida de la cabeza ImageNet», nunca de accuracy. 

Detalle: brightness_low: ρ=0.61 (grande), r_rb=0.29 (pequeña); brightness_medium: ρ=0.59 (grande), r_rb=0.27 (pequeña); brightness_high: ρ=0.61 (grande), r_rb=0.33 (moderada); contrast_low: ρ=0.60 (grande), r_rb=0.30 (moderada); contrast_medium: ρ=0.62 (grande), r_rb=0.28 (pequeña); contrast_high: ρ=0.63 (grande), r_rb=0.30 (moderada); rotation_low: ρ=0.54 (grande), r_rb=0.27 (pequeña); rotation_medium: ρ=0.54 (grande), r_rb=0.30 (pequeña); rotation_high: ρ=0.52 (grande), r_rb=0.35 (moderada); zoom_low: ρ=0.57 (grande), r_rb=0.24 (pequeña); zoom_medium: ρ=0.54 (grande), r_rb=0.24 (pequeña); zoom_high: ρ=0.50 (grande), r_rb=0.20 (pequeña); gamma_low: ρ=0.43 (moderada), r_rb=0.13 (pequeña); gamma_medium: ρ=0.43 (moderada), r_rb=0.17 (pequeña); gamma_high: ρ=0.41 (moderada), r_rb=0.18 (pequeña); rotR1_low: ρ=0.60 (grande), r_rb=0.23 (pequeña); rotR1_medium: ρ=0.58 (grande), r_rb=0.26 (pequeña); rotR1_high: ρ=0.58 (grande), r_rb=0.29 (pequeña); rotR3_low: ρ=0.53 (grande), r_rb=0.30 (pequeña); rotR3_medium: ρ=0.48 (moderada), r_rb=0.39 (moderada); rotR3_high: ρ=0.45 (moderada), r_rb=0.38 (moderada); zoomZ1_low: ρ=0.34 (moderada), r_rb=0.18 (pequeña); zoomZ1_medium: ρ=0.40 (moderada), r_rb=0.22 (pequeña); zoomZ1_high: ρ=0.49 (moderada), r_rb=0.29 (pequeña); zoomZ1r_low: ρ=0.84 (grande), r_rb=0.44 (moderada); zoomZ1r_medium: ρ=0.79 (grande), r_rb=0.43 (moderada); zoomZ1r_high: ρ=0.64 (grande), r_rb=0.40 (moderada); zoomZ2_low: ρ=0.59 (grande), r_rb=0.18 (pequeña); zoomZ2_medium: ρ=0.59 (grande), r_rb=0.17 (pequeña); zoomZ2_high: ρ=0.59 (grande), r_rb=0.18 (pequeña)

## Sparsity antes del GAP: asociación en algunos bloques

Donde la asociación es relevante, el cambio de sparsity se asocia con la sensibilidad en ese bloque; no se formula como causa.

Detalle: brightness_low Block1: ρ=0.80 [0.79, 0.82]; brightness_low Block3: ρ=0.69 [0.67, 0.71]; brightness_low Block4: ρ=0.45 [0.42, 0.49]; brightness_low Block5: ρ=0.36 [0.33, 0.40]; brightness_medium Block1: ρ=0.76 [0.74, 0.78]; brightness_medium Block2: ρ=0.65 [0.63, 0.68]; brightness_medium Block3: ρ=0.86 [0.85, 0.87]; brightness_medium Block4: ρ=0.65 [0.63, 0.68]; brightness_medium Block5: ρ=0.46 [0.43, 0.49]; brightness_high Block1: ρ=0.74 [0.72, 0.76]; brightness_high Block2: ρ=0.69 [0.67, 0.72]; brightness_high Block3: ρ=0.88 [0.87, 0.89]; brightness_high Block4: ρ=0.71 [0.69, 0.73]; brightness_high Block5: ρ=0.52 [0.48, 0.55]; contrast_low Block1: ρ=0.53 [0.50, 0.56]; contrast_low Block4: ρ=0.33 [0.29, 0.36]; contrast_medium Block1: ρ=0.51 [0.48, 0.54]; contrast_medium Block4: ρ=0.33 [0.29, 0.36]; contrast_high Block1: ρ=0.45 [0.41, 0.48]; contrast_high Block4: ρ=0.33 [0.29, 0.36]; rotation_low Block1: ρ=0.40 [0.37, 0.43]; rotation_low Block4: ρ=0.36 [0.33, 0.40]; rotation_medium Block1: ρ=0.45 [0.42, 0.47]; rotation_high Block1: ρ=0.38 [0.35, 0.41]; zoom_low Block2: ρ=0.32 [0.28, 0.35]; zoom_medium Block1: ρ=0.31 [0.28, 0.35]; zoom_medium Block2: ρ=0.40 [0.37, 0.44]; zoom_high Block1: ρ=0.31 [0.27, 0.34]; zoom_high Block2: ρ=0.38 [0.35, 0.41]; gamma_low Block1: ρ=0.62 [0.59, 0.64]; gamma_low Block2: ρ=0.78 [0.76, 0.79]; gamma_low Block3: ρ=0.68 [0.66, 0.70]; gamma_low Block4: ρ=0.43 [0.39, 0.46]; gamma_medium Block1: ρ=0.76 [0.74, 0.78]; gamma_medium Block2: ρ=0.77 [0.75, 0.78]; gamma_medium Block3: ρ=0.76 [0.75, 0.78]; gamma_medium Block4: ρ=0.57 [0.54, 0.59]; gamma_high Block1: ρ=0.79 [0.77, 0.80]; gamma_high Block2: ρ=0.75 [0.73, 0.77]; gamma_high Block3: ρ=0.81 [0.80, 0.82]; gamma_high Block4: ρ=0.62 [0.60, 0.65]; rotR2_low Block1: ρ=0.40 [0.37, 0.43]; rotR2_low Block4: ρ=0.36 [0.33, 0.40]; rotR2_medium Block1: ρ=0.45 [0.42, 0.47]; rotR2_high Block1: ρ=0.38 [0.35, 0.41]; rotR3_low Block1: ρ=0.47 [0.44, 0.50]; rotR3_medium Block1: ρ=0.53 [0.51, 0.56]; rotR3_high Block1: ρ=0.53 [0.50, 0.55]; zoomZ1_low Block2: ρ=0.46 [0.43, 0.49]; zoomZ1_high Block2: ρ=0.32 [0.28, 0.35]; zoomZ1r_low Block5: ρ=0.37 [0.34, 0.41]; zoomZ1r_medium Block1: ρ=0.34 [0.30, 0.37]; zoomZ1r_medium Block2: ρ=0.38 [0.35, 0.41]; zoomZ1r_medium Block4: ρ=0.39 [0.36, 0.42]; zoomZ1r_high Block1: ρ=0.33 [0.29, 0.37]; zoomZ1r_high Block2: ρ=0.58 [0.55, 0.60]; zoomZ1r_high Block4: ρ=0.43 [0.40, 0.47]

## Hallazgo post-extracción: imágenes duplicadas entre clases

3 pares de imágenes idénticas en clases distintas (tabla 01). Se registran como hallazgo posterior a la extracción y no se excluyen del análisis principal.
