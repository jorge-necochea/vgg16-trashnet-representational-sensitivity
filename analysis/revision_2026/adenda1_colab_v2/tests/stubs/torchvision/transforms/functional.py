# Equivalentes PIL de torchvision.transforms.functional (rutas PIL de torchvision 0.26).
import numpy as np
from PIL import Image, ImageEnhance
def adjust_brightness(im, brightness_factor): return ImageEnhance.Brightness(im).enhance(brightness_factor)
def adjust_contrast(im, contrast_factor): return ImageEnhance.Contrast(im).enhance(contrast_factor)
def rotate(im, angle, interpolation, expand=False, fill=0):
    fc = tuple([fill] * len(im.getbands())) if len(im.getbands()) > 1 else fill
    return im.rotate(angle, resample=Image.BILINEAR, expand=expand, fillcolor=fc)
def pad(im, padding, padding_mode='constant'):
    l, t, r, b = padding
    a = np.asarray(im)
    pw = ((t, b), (l, r)) + (((0, 0),) if a.ndim == 3 else ())
    return Image.fromarray(np.pad(a, pw, mode=padding_mode))
def center_crop(im, output_size):
    ch, cw = output_size; w, h = im.size
    top, left = int(round((h - ch) / 2.0)), int(round((w - cw) / 2.0))
    return im.crop((left, top, left + cw, top + ch))
