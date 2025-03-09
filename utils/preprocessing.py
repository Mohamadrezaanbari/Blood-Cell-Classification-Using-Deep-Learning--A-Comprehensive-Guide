import numpy as np


def normalize_images(images):
    """
    نرمال‌سازی تصاویر به بازه [0, 1].

    پارامترها:
        images (list of np.array): لیست تصاویر.

    بازگشت:
        np.array: آرایه NumPy حاوی تصاویر نرمال‌شده.
    """
    images = np.array(images, dtype=np.float32)
    return images / 255.0


def normalize_masks(masks):
    """
    نرمال‌سازی ماسک‌ها به بازه [0, 1].

    پارامترها:
        masks (list of np.array): لیست ماسک‌ها.

    بازگشت:
        np.array: آرایه NumPy حاوی ماسک‌های نرمال‌شده.
    """
    masks = np.array(masks, dtype=np.float32)
    return masks / 255.0