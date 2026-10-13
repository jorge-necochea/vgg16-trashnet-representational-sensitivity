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
