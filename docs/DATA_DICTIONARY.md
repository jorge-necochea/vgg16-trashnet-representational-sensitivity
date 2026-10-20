# Data dictionary

## Main image-level metrics

File: `data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz`

| Column | Meaning |
|---|---|
| `image_id` | Filename stem used as the image identifier. |
| `original_filename` | Original TrashNet JPEG filename. |
| `in_balanced_822` | Whether the image belongs to the balanced 137-per-class subset. |
| `original_format` | Storage format of the source image. |
| `transformed_format` | Storage format of the controlled transformed image. |
| `class` | TrashNet class. |
| `transformation` | Brightness, contrast, rotation, or zoom. |
| `level` | Transformation-specific low, medium, or high setting. |
| `value` | Numeric factor or angle for the condition. |
| `block` | VGG-16 representational stage. |
| `euclidean_distance` | Euclidean distance between paired embeddings. |
| `cosine_similarity` | Cosine similarity between paired embeddings. |
| `relative_euclidean_distance` | Euclidean distance divided by original embedding norm plus epsilon. |

## R1 adjusted sensitivity

File: `data/derived_metrics/r1/image_level_adjusted_sensitivity.csv.gz`

The baseline-adjusted angular sensitivity is
`(1 - paired cosine) / (1 - median random-pair cosine for the block)`.
Centered linear CKA summaries are in `data/summary_tables/r1/`.

The complete transformation manifest is stored as
`data/manifests/transformation_manifest_complete.csv.gz`. Gzip compression is
lossless, and pandas reads each of these files directly with `pd.read_csv()`.

## R3 clipping

Image-level brightness saturation and newly clipped-pixel proportions are in
`data/derived_metrics/r3/`. A pixel is saturated when at least one RGB channel
equals 255.

## R4 intensity ordering

The R4 tables report medians, Kendall's W, strict low–medium–high ordering,
and image-level inversions, with angular sensitivity defined as `1 - cosine`.

## R5 peripheral gradient and object extent

The R5 image-level files contain automatic object-extent proxies, percentage
of peripheral gradient energy removed under center-crop zoom, and the merged
Block5 sensitivity measures. The foreground mask is an image-based proxy, not
a semantic segmentation ground truth.

## C7 storage-format aggregate fields

`data/summary_tables/storage_format_aggregate_summary.csv` contains one row per
**primary metric**. Its two `direct_jpg_png_embedding_cosine_*` columns are populated
**only for the cosine-similarity row**, because they describe a single independent
JPG–PNG embedding comparison, not a separate statistic computed for each primary
metric. The pooled mean covers **822 images × 12 conditions × 5 blocks = 49,320**
matched image–condition–block pairs. The `minimum_of_60_cell_means` is the
smallest of **60 condition-by-block means**, **not** the minimum individual-pair
cosine similarity. The stored precision is rounded to four decimal places.

## C8 Kruskal–Wallis effect size

`data/summary_tables/r5/07_gradient_class_omnibus_tests.csv` reports
`eta_squared_h = max(0, (H - k + 1)/(n - k))`, with `k=6` TrashNet classes
and `n=2527` per zoom condition. Earlier output incorrectly labelled this
value `epsilon_squared`; only the column name has changed. Values are identical.
The alternative epsilon-squared statistic `H/(n-1)` was **not calculated here**.
The original historical statistical notebook remains unchanged as provenance;
the maintained R5 notebook uses the corrected `eta_squared_h` identifier.
