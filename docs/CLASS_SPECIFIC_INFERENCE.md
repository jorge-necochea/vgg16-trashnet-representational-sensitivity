# Complete class-specific inference (Block1–Block5)

The primary 2,527-image PNG analysis reports the main-text class-specific inferential summaries for **Block5**. This directory makes the corresponding statistical tests for **all five VGG-16 blocks** publicly inspectable.

## Results

- `data/summary_tables/main/10_kruskal_wallis_clases.csv`: 120 omnibus tests (4 transformations × 3 intensity levels × 5 blocks × 2 metrics), with H, raw p, Holm-adjusted p and the corrected **eta-squared based on H** (`eta_squared_h`). This estimator is `max(0, (H-k+1)/(n-k))`, not epsilon-squared.
- `data/summary_tables/main/11_mann_whitney_clases_holm.csv`: 1,800 two-sided pairwise Mann–Whitney U tests (15 class pairs for each of 120 conditions), with within-family Holm-adjusted p and rank-biserial correlation (`rank_biserial_a_vs_b`). Pairwise tests were computed for every condition as in the historical analysis notebook.

## Provenance and recomputation

Input: `data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz` (151,620 image–condition–block observations).

From the repository root:

```bash
python scripts/recompute_class_tests.py
python scripts/validate_repository.py
```

The regeneration script applies the original `notebooks/originals/analisis_estadistico_definitivo_original.ipynb` test definitions, with corrected column nomenclature: Holm across 60 Kruskal–Wallis tests separately per metric; Holm within each group of 15 post hoc class comparisons. It makes no changes to model embeddings or image preprocessing. The regenerated 120/1,800 outputs were checked against the historical October 2026 statistical tables (numerical values match), and the 12 Block5 **cosine similarity** Kruskal–Wallis H and `eta_squared_h` values match manuscript Table 8 after its published rounding.

**Scope:** This is an independent rerun of the downstream class-specific inference using already-published per-image metrics. It is **not** a rerun of VGG-16 feature extraction or a clean end-to-end replay of all notebooks. Source images and full embedding matrices are not redistributed.
