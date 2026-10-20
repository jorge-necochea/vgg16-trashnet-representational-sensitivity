# Reproducibility notes

## Determinism

The principal extraction uses seed 2026, deterministic cuDNN settings, frozen
ImageNet VGG-16 weights, and no stochastic image augmentation. R1 uses the same
seed to select 100,000 random, same-class, and different-class pairs per block.
R5 bootstrap analyses also use seed 2026.

## Image processing order

1. Load the 512×384 original JPEG and convert it to RGB.
2. Apply the controlled transformation at the original resolution.
3. Save the transformed image as a 512×384 PNG.
4. Directly resize both original and transformed images to 224×224.
5. Convert to a tensor and apply ImageNet normalization.

Brightness and contrast preserve dimensions. Rotation uses bilinear
interpolation, a fixed canvas (`expand=False`), and zero-valued fill.
Center-crop zoom removes a centered region and bilinearly resizes it back to
512×384. The direct 512×384-to-224×224 resize changes the aspect ratio from
4:3 to 1:1 for both members of every pair.

## Released versus omitted files

The repository releases the complete scalar metrics and the derived tables
needed to inspect the reported results. The 13 full NPZ embedding matrices are
not duplicated because they can be regenerated from the public TrashNet data
and notebooks 01–02. The released R1 baseline-pair sample contains the exact
100,000 cosine values per pair type and block used for the reported baseline
summaries.

The original TrashNet images and transformed image collections are not
redistributed.

## JPG storage settings in the balanced 822-image analysis

The original transformation notebook is released as
`notebooks/00_generate_transformations_822_jpg_original.ipynb`. It imports
`Image` from Pillow and saves every transformed image with
`transformed.save(output_path)`. Because `output_path` retains the original
`.jpg` filename extension, Pillow selects JPEG automatically. No `quality`
argument was supplied, so Pillow's documented default quality of 75 was used.
No explicit subsampling, optimization, or progressive options were supplied.
The encoder was Pillow's JPEG writer backed by the JPEG library available in
the Google Colab runtime. The exact historical Pillow and
libjpeg/libjpeg-turbo version was not recorded in the notebook.

## Editorial harmonization (20 October 2026)

The active supplement was updated to Tables S1–S13 without recomputing model
embeddings. The C7 aggregate storage-format CSV was clarified: its 0.9985 pooled
direct JPG–PNG embedding cosine covers 49,320 matched pairs and 0.9949 is the
minimum of the 60 condition-by-block **mean** cosines. The C8 R5 effect-size
variable was renamed from the incorrect `epsilon_squared` to `eta_squared_h`:
`max(0, (H-k+1)/(n-k))`. These are documentation/naming corrections only.

The full end-to-end pipeline has not been reproduced independently on a clean
runtime, and neither the published original source JPEG compression settings
nor the historical JPG encoder versions have been recovered.
