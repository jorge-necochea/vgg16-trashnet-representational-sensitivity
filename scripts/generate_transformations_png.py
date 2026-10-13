"""Generate the controlled PNG transformation set from original TrashNet JPEGs.

Usage:
    python scripts/generate_transformations_png.py \
        --input /path/to/DatasetOriginalCompleto \
        --output /path/to/TrashNet_transformaciones_completo_png

This script implements the transformation order documented in the manuscript.
It does not create the earlier secondary JPG set because its exact encoder and
quality settings still require verification from the original experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CONDITIONS = {
    'brightness': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'contrast': {'low': 1.2, 'medium': 1.4, 'high': 1.6},
    'rotation': {'low': 10, 'medium': 30, 'high': 60},
    'zoom': {'low': 1.10, 'medium': 1.25, 'high': 1.50},
}


def center_crop_zoom(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(width / factor))
    crop_height = max(1, int(height / factor))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), resample=Image.Resampling.BILINEAR)


def transform(image: Image.Image, name: str, value: float) -> Image.Image:
    if name == 'brightness':
        return F.adjust_brightness(image, brightness_factor=value)
    if name == 'contrast':
        return F.adjust_contrast(image, contrast_factor=value)
    if name == 'rotation':
        return F.rotate(
            image,
            angle=value,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0,
        )
    if name == 'zoom':
        return center_crop_zoom(image, value)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    originals = []
    for class_name in CLASSES:
        originals.extend((class_name, path) for path in sorted((args.input / class_name).glob('*.jpg')))

    for transformation, levels in CONDITIONS.items():
        for level, value in levels.items():
            for class_name, path in tqdm(originals, desc=f'{transformation}_{level}'):
                output_dir = args.output / f'{transformation}_{level}' / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(path) as source:
                    rgb = source.convert('RGB')
                    result = transform(rgb, transformation, value)
                    result.save(output_dir / f'{path.stem}.png', format='PNG')


if __name__ == '__main__':
    main()
