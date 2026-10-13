from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
errors = []

main = pd.read_csv(ROOT / 'data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz')
if len(main) != 151620:
    errors.append(f'Main metrics rows: {len(main)} instead of 151620')
if main['image_id'].nunique() != 2527:
    errors.append(f'Unique images: {main["image_id"].nunique()} instead of 2527')
if main.duplicated(['image_id', 'transformation', 'level', 'block']).any():
    errors.append('Duplicate main-metric keys detected')

manifest = pd.read_csv(ROOT / 'data/manifests/transformation_manifest_complete.csv.gz')
if len(manifest) != 2527 * 12:
    errors.append(f'Manifest rows: {len(manifest)} instead of {2527 * 12}')

required_notebooks = [
    '00_generate_transformations_822_jpg_original.ipynb',
    '01_extract_vgg16_metrics_complete_dataset.ipynb',
    '02_r1_baselines_adjusted_sensitivity_cka.ipynb',
    '03_r3_brightness_clipping_preprocessing_audit.ipynb',
    '04_r4_intensity_ordering_exceptions.ipynb',
    '05_r5_peripheral_gradient_object_extent.ipynb',
    '06_jpeg_encoding_audit.ipynb',
]
for name in required_notebooks:
    path = ROOT / 'notebooks' / name
    if not path.exists():
        errors.append(f'Missing notebook: {name}')
    else:
        json.loads(path.read_text(encoding='utf-8'))

if errors:
    raise SystemExit('\n'.join(errors))

print('Repository validation passed.')
print('Main metric rows:', len(main))
print('Images:', main['image_id'].nunique())
print('Manifest rows:', len(manifest))
