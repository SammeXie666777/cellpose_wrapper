# -*- coding: utf-8 -*-

### 

import numpy as np
import os
from tqdm import trange
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

def save_mask_overlay_tiff(raw_path, mask_path, out_path=None, alpha=0.35):
    """
    Overlay a multi-slice mask on top of a grayscale raw image and save as an ImageJ-viewable TIFF.
    alpha : float
        Transparency of the colored mask overlay (0-1, default 0.35).
    """
    raw_path, mask_path = Path(raw_path), Path(mask_path)
    if out_path is None:
        out_path = raw_path.with_name(raw_path.stem + "_overlay.tif")

    print(f"Loading image: {raw_path.name}")
    raw_stack = tiff.imread(raw_path)
    print(f"Loading mask: {mask_path.name}")
    mask_stack = tiff.imread(mask_path)

    assert raw_stack.shape == mask_stack.shape, \
        f"Shape mismatch: raw {raw_stack.shape} vs mask {mask_stack.shape}"

    # find all unique cell IDs across the stack (for consistent coloring)
    unique_ids = np.unique(mask_stack[mask_stack > 0])
    print(f"Found {len(unique_ids)} labeled regions.")

    # assign random colors consistently across slices
    rng = np.random.default_rng(42)
    colors = rng.uniform(0, 1, size=(unique_ids.max() + 1, 3))  # color map (ID -> RGB)

    # build composite RGB stack
    composite_stack = []
    for z in range(raw_stack.shape[0]):
        raw_slice = raw_stack[z].astype(np.float32)
        raw_slice /= raw_slice.max() + 1e-8  # normalize 0-1
        rgb_slice = np.stack([raw_slice]*3, axis=-1)  # grayscale -> RGB

        mask_slice = mask_stack[z]
        color_mask = np.zeros_like(rgb_slice)
        nonzero = mask_slice > 0
        color_mask[nonzero] = colors[mask_slice[nonzero]]

        overlay = (1 - alpha) * rgb_slice + alpha * color_mask
        composite_stack.append((overlay * 255).astype(np.uint8))
    composite_stack = np.stack(composite_stack, axis=0)

    # Save as ImageJ-readable RGB stack
    tiff.imwrite(
        out_path,
        composite_stack,
        photometric='rgb',
        imagej=True
    )

    print(f"Overlay saved as: {out_path}")
    print(f"Shape: {composite_stack.shape}  (Z, Y, X, 3)")
    return out_path




# # Plot example overlay tiff stacks
# plt.tight_layout()
# plt.show()

# raw_path = "/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/Sparrow_test/New_Model/test/Sparrow_2_4-16_2_1_405_NotOn3D_stitch_thres_0.2/Sparrow_2_4-16_2_1_405.tif"
# mask_path = "/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/Sparrow_test/New_Model/test/Sparrow_2_4-16_2_1_405_NotOn3D_stitch_thres_0.2/Sparrow_2_4-16_2_1_405_cp_masks.tif"
# save_mask_overlay_tiff(raw_path, mask_path)


