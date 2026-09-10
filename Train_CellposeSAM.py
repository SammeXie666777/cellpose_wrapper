# -*- coding: utf-8 -*-
"""
Run Cellpose (Cellpose-SAM compatible call) on a folder of images from a local Anaconda environment.
"""

# ------------------------- Change things here-------------------------
root_dir   = "/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/Sparrow_test/New_Model/"
masks_ext  = "_masks"   # e.g. "_masks", "_seg.npy", "_masks.tif"
model_name = "Marmoset_NewModel"
#testdir = "/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/Sparrow_test/New_Model/test/"
# ---------------------------------------------------------------

# importing stuff
#from natsort import natsorted
#import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.env import require_package
require_package('cellpose') # maybe modify this function into a env/dependency setup
from tqdm import trange
from cellpose import models, core, io, plot, metrics, train, utils, transforms, dynamics

# ------------------------- helpers -------------------------
def make_train_paired_auto(
    dir_path,
    crop_size=512,
    nimg_per_tif=10,
    anisotropy=1.0,
    z_axis = 0,
    channel_axis = -1,
    sharpen_radius=0.0,
    tile_norm=0,
    visualize=True,
):
    """
    Automatically find paired raw and mask volumes in a directory,
    and generate perfectly aligned 2D training crops (YX, ZY, ZX).
    """

    dir_path = Path(dir_path)
    out_dir = dir_path / "train"
    out_dir.mkdir(exist_ok=True)

    # --- Find raw and mask pairs ---
    all_files = list(dir_path.glob("*.tif"))
    mask_files = [f for f in all_files if "_masks" in f.stem]
    raw_files = [
        f for f in all_files if not any(x in f.stem for x in ["_masks", "_seg", "_flows"])
    ]

    pairs = []
    for raw in raw_files:
        base = raw.stem
        mask_match = next((m for m in mask_files if base in m.stem), None)
        if mask_match:
            pairs.append((raw, mask_match))

    if not pairs:
        raise FileNotFoundError("No matching raw/mask file pairs found.")

    print(f"Found {len(pairs)} image/mask pairs in {dir_path}")

    # --- Process each pair ---
    pm = [(0, 1, 2, 3), (2, 0, 1, 3), (1, 0, 2, 3)]  # YX, ZY, ZX
    npm = ["YX", "ZY", "ZX"]

    for raw_path, mask_path in pairs:
        print(f"\nProcessing: {raw_path.name}")

        img = io.imread_3D(raw_path)
        mask = io.imread_3D(mask_path)
        assert img.shape == mask.shape, f"Shape mismatch: {img.shape} vs {mask.shape}"

        img = transforms.convert_image(img, channel_axis=channel_axis, z_axis=z_axis, do_3D=True)
        mask = transforms.convert_image(mask, channel_axis=channel_axis, z_axis=z_axis, do_3D=True)

        np.random.seed(0)
        for p, perm in enumerate(pm):
            img_p = img.transpose(perm).copy()
            mask_p = mask.transpose(perm).copy()
            Ly, Lx = img_p.shape[1:3]

            slices = np.random.permutation(img_p.shape[0])[:nimg_per_tif]
            if anisotropy > 1.0 and p > 0:
                img_p = transforms.resize_image(img_p, Ly=int(anisotropy * Ly), Lx=Lx)
                mask_p = transforms.resize_image(mask_p, Ly=int(anisotropy * Ly), Lx=Lx)
                Ly = int(anisotropy * Ly)

            for k, idx in enumerate(slices):
                img_slice = img_p[idx]
                mask_slice = mask_p[idx]

                ly = 0 if Ly - crop_size <= 0 else np.random.randint(0, Ly - crop_size)
                lx = 0 if Lx - crop_size <= 0 else np.random.randint(0, Lx - crop_size)

                crop_img = img_slice[ly:ly + crop_size, lx:lx + crop_size]
                crop_mask = mask_slice[ly:ly + crop_size, lx:lx + crop_size]

                io.imsave(out_dir / f"{raw_path.stem}_{npm[p]}_{k}.tif", crop_img)
                io.imsave(out_dir / f"{raw_path.stem}_{npm[p]}_{k}_masks.tif", crop_mask)

        print(f"Saved aligned crops for {raw_path.stem} -> {out_dir}")

    # --- Optional visualization ---
    if visualize:
        print("\nShowing example crops from last processed volume...")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ori in enumerate(npm):
            img_file = next(out_dir.glob(f"*_{ori}_*_masks.tif"))
            base = str(img_file).replace("_masks.tif", ".tif")
            raw_ex = io.imread(base)
            mask_ex = io.imread(img_file)
            axes[i].imshow(raw_ex, cmap="gray")
            axes[i].imshow(mask_ex, alpha=0.3)
            axes[i].set_title(ori)
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()
        
    return out_dir # return training directory


def collapse_nonzero_channel(img_list):
    """
    Given a list of 3D images (H, W, C), collapse to 2D by keeping
    the single nonzero channel (channels with all zeros are dropped).
    Returns a new list of 2D arrays.
    """
    collapsed = []
    for i, img in enumerate(img_list):
        if img.ndim == 3:  # has channels
            # Compute sum of absolute values per channel
            sums = [np.sum(np.abs(img[..., c])) for c in range(img.shape[-1])]
            keep_c = np.argmax(sums)  # keep the channel with the most signal
            if sums[keep_c] == 0:
                print(f"Warning: all-zero image at index {i}")
                collapsed.append(img[..., 0])
            else:
                collapsed.append(img[..., keep_c])
        else:
            collapsed.append(img)
    return collapsed


# ------------------------- setup -------------------------
io.logger_setup()  # enable Cellpose logging
useGPU = core.use_gpu()
if not useGPU:
    print("No GPU access detected; using CPU.")

# initialize model: default model cpsam; training from scratch not allowed
model = models.CellposeModel(gpu=useGPU)

# slice images
train_dir = make_train_paired_auto(root_dir)
train_dir = str(train_dir)
print(f"Train dir: {train_dir}")

# ------------------------- Training -------------------------
n_epochs = 100
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 8

# get files
# Train_Dir should contain 2D slices 
output = io.load_train_test_data(train_dir, None, mask_filter=masks_ext, look_one_level_down=False)
train_data, train_labels, _, _, _, _ = output
train_labels = collapse_nonzero_channel(train_labels)

nimg = len(train_data)
diam_train = np.zeros(nimg)
nmasks = np.zeros(nimg)
for k in trange(nimg):
    tl = train_labels[k]
    diam_train[k], dall = utils.diameters(tl)
    nmasks[k] = len(dall)
nmasks

model_path = train_dir
print("Starting training ...")
new_model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=train_data,
    train_labels=train_labels,
    batch_size=batch_size,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    nimg_per_epoch=max(2, len(train_data)), # can change
    model_name=model_name,
    compute_flows=True,
    save_path = model_path
)
print(f"Model saved to: {new_model_path}")


# ------------------------- Evaluate -------------------------
test_dir = Path(root_dir) / "test"
test_tiff =  io.get_image_files(str(test_dir), mask_filter=masks_ext)
test_data= io.imread(test_tiff[0])
model_eval = models.CellposeModel(gpu=useGPU, pretrained_model=new_model_path)
pred_masks = model_eval.eval(test_data, batch_size=32, do_3D=True,channel_axis=-1,z_axis=0)[0]
name = os.path.splitext(str(test_tiff[0]))[0]
io.imsave( f"{name}_masks.tif", pred_masks)

step = 5
z_indices = np.arange(0, test_data.shape[0], step)  # sample every 20th plane
plt.figure(figsize=(12, 6), dpi=150)
for i, z in enumerate(z_indices):
    img = test_data[z].copy()

    plt.subplot(2, len(z_indices)//2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.imshow(pred_masks[z], cmap='tab20', alpha=0.35)  # overlay with transparency
    plt.axis('off')
    plt.title(f'Slice {z}')
plt.tight_layout()
plt.show()


# have_test = bool(test_dir and test_data and test_labels)
# if have_test:
#     print("Evaluating on test set ...")
#     model_eval = models.CellposeModel(gpu=useGPU, pretrained_model=new_model_path)

#     # model.eval returns [masks, flows, styles, diams]
#     pred_masks = model_eval.eval(test_data, batch_size=32, do_3D=True)[0]

#     # AP can be returned in different shapes; handle 1D or 2D
#     ap_out = metrics.average_precision(test_labels, pred_masks)
#     ap = ap_out[0] if isinstance(ap_out, (list, tuple)) and len(ap_out) > 0 else ap_out
#     ap = np.asarray(ap)
#     ap50 = float(ap[:, 0].mean()) if ap.ndim == 2 and ap.shape[1] >= 1 else float(ap.mean())

#     print(f"\n>>> average precision at IoU=0.5 H {ap50:.3f}")

#     # ------------------------- plot a panel -------------------------
#     kmax = min(6, len(test_data))  # cap for readability
#     plt.figure(figsize=(12, 2 + 2*kmax), dpi=150)
#     for k in range(kmax):
#         im_disp = to_display(np.array(test_data[k]))

#         plt.subplot(3, kmax, k + 1)
#         plt.imshow(im_disp)
#         plt.axis('off')
#         if k == 0:
#             plt.title('image')

#         plt.subplot(3, kmax, kmax + k + 1)
#         plt.imshow(pred_masks[k])
#         plt.axis('off')
#         if k == 0:
#             plt.title('predicted')

#         plt.subplot(3, kmax, 2*kmax + k + 1)
#         plt.imshow(test_labels[k])
#         plt.axis('off')
#         if k == 0:
#             plt.title('ground truth')
#     plt.tight_layout()
# else:
#     print("No test set to evaluate; skipping evaluation/plots.")