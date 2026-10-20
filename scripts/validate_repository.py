"""Integrity and numerical-consistency checks for the public VGG-16 dataset.

Run from repository root: python scripts/validate_repository.py
Checks existing derived data and published checksums; DOES NOT rerun the model.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors = []

def check(condition: bool, message: str) -> None:
    if not bool(condition): errors.append(message)

def load_csv(rel: str) -> pd.DataFrame:
    try:
        return pd.read_csv(ROOT / rel)
    except Exception as exc:
        errors.append(f"Could not read {rel}: {exc}")
        return pd.DataFrame()

# Verify every gzip archive to the stream terminator (detects truncated archives).
for path in sorted(ROOT.rglob('*.gz')):
    try:
        with gzip.open(path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''): pass
    except Exception as exc:
        errors.append(f"Invalid gzip: {path.relative_to(ROOT)}: {exc}")

# Verify every SHA-256 line; the manifest intentionally does not hash itself.
manifest_file = ROOT / 'checksums.sha256'
try:
    listed = set()
    for line in manifest_file.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        checksum, rel = line.split('  ', 1)
        listed.add(rel)
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"Missing checksummed file: {rel}")
            continue
        sha = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                sha.update(chunk)
        check(sha.hexdigest() == checksum, f"Checksum mismatch: {rel}")
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
              if p.is_file() and p != manifest_file
              and '.git' not in p.relative_to(ROOT).parts}
    check(actual == listed, f"Checksum manifest coverage mismatch: missing={sorted(actual-listed)}, stale={sorted(listed-actual)}")
except Exception as exc:
    errors.append(f"Could not verify checksums: {exc}")

main = load_csv('data/derived_metrics/main/embeddings_metrics_vgg16_trashnet_complete.csv.gz')
r1 = load_csv('data/derived_metrics/r1/image_level_adjusted_sensitivity.csv.gz')
if not main.empty and not r1.empty:
    check(len(main) == 151620, f"Main metrics: {len(main)} rows; expected 151620")
    check(len(r1) == 151620, f"R1 metrics: {len(r1)} rows; expected 151620")
    expected_cols = ['image_id','original_filename','in_balanced_822','original_format','transformed_format','class','transformation','level','value','block','euclidean_distance','cosine_similarity','relative_euclidean_distance']
    check(all(c in main.columns for c in expected_cols), 'Missing main-metric columns')
    check(all(c in r1.columns for c in expected_cols + ['random_pair_baseline_median','baseline_adjusted_sensitivity']), 'Missing R1 columns')
    if all(c in main.columns and c in r1.columns for c in expected_cols):
        for col in expected_cols:
            if pd.api.types.is_numeric_dtype(main[col]) and col != 'in_balanced_822':
                check(np.allclose(main[col], r1[col], atol=1e-12, rtol=0, equal_nan=False), f"Main/R1 values differ: {col}")
            else:
                check(main[col].equals(r1[col]), f"Main/R1 rows differ: {col}")
        check(main['image_id'].nunique() == 2527, 'Expected 2527 original images')
        check(main['block'].nunique() == 5, 'Expected 5 blocks')
        check(len(main[['transformation','level']].drop_duplicates()) == 12, 'Expected 12 conditions')
        check(not main.duplicated(['image_id','transformation','level','block']).any(), 'Duplicate main rows')
        check(main[expected_cols].isna().sum().sum() == 0, 'Null values in main metrics')
    if all(c in r1.columns for c in ['cosine_similarity','random_pair_baseline_median','baseline_adjusted_sensitivity']):
        formula=(1-r1['cosine_similarity'])/(1-r1['random_pair_baseline_median'])
        check(np.allclose(formula, r1['baseline_adjusted_sensitivity'], atol=2e-8, rtol=0), 'R1 formula mismatch')
        check(not r1.isna().any().any(), 'Null values in R1 metrics')

manifest = load_csv('data/manifests/transformation_manifest_complete.csv.gz')
if not manifest.empty:
    check(len(manifest) == 30324, f'Transformation manifest: {len(manifest)} rows; expected 30324')
    check(not manifest.duplicated(['class','original_filename','transformation','level']).any(), 'Duplicate transformation manifest keys')
storage = load_csv('data/summary_tables/storage_format_matched_822_complete.csv')
if not storage.empty:
    check(len(storage) == 180, f'Storage table: {len(storage)} rows; expected 180')

robust = ROOT / 'analysis/robustness822'
for filename, expected in {'imagenes_seleccionadas.csv':822,'CKA_por_estrategia.csv':180,'baselines_por_estrategia.csv':45,'bloques_pico_sensibilidad_ajustada.csv':36,'sensibilidad_ajustada_por_estrategia.csv':180}.items():
    df=load_csv(f'analysis/robustness822/{filename}')
    if not df.empty: check(len(df) == expected, f'Robustness {filename}: expected {expected}, got {len(df)}')
try:
    metadata=json.loads((robust/'metadatos_robustez.json').read_text(encoding='utf-8'))
    check(metadata.get('images') == 822 and metadata.get('rows_per_strategy') == 49320, 'Robustness metadata count mismatch')
    check(set(metadata.get('strategies',[])) == {'stretch_original','no_padding','letterbox'}, 'Unexpected preprocessing strategy set')
except Exception as exc:
    errors.append(f'Invalid robustness metadata: {exc}')

notebooks = sorted(ROOT.rglob('*.ipynb'))
for path in notebooks:
    try: json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc: errors.append(f'Invalid notebook {path.relative_to(ROOT)}: {exc}')


# C7 / C8 editorial result-label and formula validation
c7 = load_csv('data/summary_tables/storage_format_aggregate_summary.csv')
if not c7.empty:
    check(len(c7) == 2, 'C7: expected exactly 2 primary-metric rows')
    v_mean='direct_jpg_png_embedding_cosine_overall_mean_49320_pairs'
    v_min='direct_jpg_png_embedding_cosine_minimum_of_60_cell_means'
    check(all(x in c7.columns for x in (v_mean,v_min)), 'C7: missing explicit direct cosine labels')
    if all(x in c7.columns for x in ('metric',v_mean,v_min)):
        cosine=c7[c7['metric']=='cosine_similarity']
        distance=c7[c7['metric']=='relative_euclidean_distance']
        check(len(cosine)==1 and len(distance)==1, 'C7: incorrect metric rows')
        if len(cosine)==1:
            check(abs(cosine.iloc[0][v_mean]-0.9985)<0.00005, 'C7: pooled mean cosine mismatch')
            check(abs(cosine.iloc[0][v_min]-0.9949)<0.00005, 'C7: minimum cell-mean cosine mismatch')
        if len(distance)==1:
            check(pd.isna(distance.iloc[0][v_mean]) and pd.isna(distance.iloc[0][v_min]), 'C7: duplicate direct JPG–PNG cosine in relative-distance row')
c8 = load_csv('data/summary_tables/r5/07_gradient_class_omnibus_tests.csv')
if not c8.empty:
    check('eta_squared_h' in c8.columns and 'epsilon_squared' not in c8.columns, 'C8: effect-size column mislabelled')
    if all(x in c8.columns for x in ('H','df','eta_squared_h')):
        k=c8['df']+1
        expected=np.maximum(0, (c8['H']-k+1)/(2527-k))
        check(np.allclose(expected,c8['eta_squared_h'],atol=1e-12,rtol=0), 'C8: eta squared H values disagree with Kruskal–Wallis formula')
try:
    from zipfile import ZipFile
    supplement=ROOT/'supplementary_materials/Supplementary_Materials_VGG16_TrashNet_Revised.docx'
    with ZipFile(supplement) as doc:
        source=doc.read('word/document.xml').decode('utf-8')
        for label in ('S11','S12','S13'):
            check(label in source, f'Missing consolidated supplemental table {label}')
except Exception as exc:
    errors.append(f'Could not inspect consolidated supplement: {exc}')

if errors:
    print('VALIDATION FAILED:')
    for issue in errors: print(' -',issue)
    raise SystemExit(1)
print('PASS — integrity checks on all compressed files and published checksums')
print('PASS — 151620 aligned main and R1 records; adjusted sensitivity formula')
print('PASS — 30324 transformation manifest entries and 822-image robustness summaries')
print('NOTE — this does not establish clean end-to-end notebook execution or regenerate embedding matrices')
