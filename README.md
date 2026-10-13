# VGG-16 TrashNet representational sensitivity

Code, derived data, and figures for the manuscript **“Representational
Sensitivity of VGG-16 to Controlled Image Transformations: A Layer-Wise and
Class-Specific Case Study.”**

The study evaluates 2,527 TrashNet images under brightness, contrast,
rotation, and center-crop zoom at three intensity levels. Representations from
five post-ReLU convolutional stages of ImageNet-pretrained VGG-16 are compared
using cosine similarity, relative Euclidean distance, baseline-adjusted angular
sensitivity, and centered linear CKA.

## Repository status

The primary PNG analysis and editorial revision analyses R1, R3, R4, and R5
are included. R2 is a manuscript scope revision and does not require a new
computational analysis.

The original notebook used to generate the transformed JPG files for the
balanced 822-image analysis is included as notebook 00. It uses Pillow
`Image.save()` without an explicit `quality` argument. Because the original
filenames retain their `.jpg` extension, Pillow selects JPEG automatically and
uses its default JPEG quality of 75. No explicit subsampling, optimization, or
progressive option was supplied. The exact historical Pillow and
libjpeg/libjpeg-turbo version used by the Colab runtime was not recorded by
that notebook; this limitation is stated explicitly rather than guessed.

## Directory guide

- `notebooks/`: clean analysis notebooks in execution order;
- `data/manifests/`: image identifiers and relative paths, without original images;
- `data/derived_metrics/`: image-level derived results;
- `data/summary_tables/`: compact statistical outputs;
- `figures/`: publication and supplementary figures;
- `environment/`: software versions, seeds, and validation records;
- `docs/`: data dictionary, reproduction instructions, and R6 checklist;
- `scripts/`: transformation and repository-validation utilities.

Large tabular files are stored as gzip-compressed CSV files (`.csv.gz`) so
they can be uploaded through the GitHub web interface. They contain the full
original tables without sampling or loss of numeric precision. Pandas reads
them directly with `pd.read_csv(path)` because compression is inferred from
the filename.

## Original data

TrashNet is publicly available at:
https://github.com/garythung/trashnet

The original TrashNet files used in this study are JPEG images. The transformed
images stored as PNG are lossless *storage derivatives* of these already lossy
JPEG sources; conversion to PNG does not restore information lost in the
original JPEG encoding. Original images are not redistributed here.

## Core settings

- Model: torchvision VGG-16 with `VGG16_Weights.IMAGENET1K_V1`;
- Stages: Block1–Block5 at feature indices 3, 8, 15, 22, and 29;
- Pooling: global average pooling over the spatial dimensions;
- Input preprocessing: direct bilinear resize from 512×384 to 224×224,
  tensor conversion, and ImageNet normalization; aspect ratio was not preserved;
- Seed: 2026;
- Brightness factors: 1.2, 1.4, 1.6;
- Contrast factors: 1.2, 1.4, 1.6;
- Rotation angles: 10°, 30°, 60°;
- Center-crop zoom factors: 1.10×, 1.25×, 1.50×.

The labels low, medium, and high are transformation-specific experimental
settings and are not a common physical severity scale across transformations.

## Reproduction order

1. Download TrashNet and arrange the six class folders under
   `DatasetOriginalCompleto/`.
2. Notebook 00 documents the original 822-image JPG transformation workflow.
3. Generate the controlled PNG transformations with
   `scripts/generate_transformations_png.py`.
4. Run notebook 01 to extract the complete image-level metrics.
5. Run notebook 02 for R1 baselines, adjusted sensitivity, and CKA.
6. Run notebooks 03–05 for the R3–R5 analyses.
7. Run `python scripts/validate_repository.py` to verify released files.

The notebooks were developed for Google Colab. Their configuration cells use
`/content/gdrive/MyDrive/2026-0` as the default project root; edit only that
root if the project is stored elsewhere.

## Data availability statement

Suggested manuscript wording is provided in `docs/DATA_AVAILABILITY.md`.

## License

Code is released under the MIT License. Derived data and original figures are
released under CC BY 4.0; see `DATA_LICENSE.md`. The original TrashNet dataset
is not covered by these repository licenses.
