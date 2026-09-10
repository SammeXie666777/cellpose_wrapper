#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:05:48 2025

@author: sammxie
# Read mat files and get file names for processing
"""

from scipy.io import loadmat, savemat
import numpy as np
from pathlib import Path
import socket
from utils.env import require_package
require_package('cellpose') # maybe modify this function into a env/dependency setup
from cellpose import models, core, io, plot, metrics, train, utils, transforms, dynamics
from tqdm import trange

def convert_mat_paths(animal, og_base="X:/Projects/ARG/Animals", 
                      scc_base="/net/claustrum2/mnt/data/Projects/ARG/Animals"):
    """
    Convert Windows (X:\\...) paths in mat_405 variables of MATLAB .mat files
    into SCC-style (/net/claustrum2/...) paths.
    """
    hostname = socket.gethostname()

    # --- Detect current environment
    if "scc" in hostname.lower():
        base_path = Path(scc_base)
        system_label = "SCC"
    else:
        base_path = Path(og_base)
        system_label = "local"

    print(f"Running on {system_label} system.")
    print(f"Using base path: {base_path}\n")

    mat_file = base_path / animal / f"{animal}_cellpose_input_tiff.mat"

    # --- Fast existence check
    if not mat_file.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_file}")

    print(f"Found file: {mat_file}\n")

    # --- Load .mat file
    data = loadmat(mat_file, simplify_cells=True)

    if "mat_DAPI" not in data:
        raise KeyError(f"'mat_405' variable not found in {mat_file}")

    mat_DAPI = np.atleast_1d(data["mat_DAPI"])

    # --- Replace old base with current base
    if system_label == "SCC":
        old_base = og_base.replace("\\", "/")
        new_base = scc_base
    else:
        old_base = scc_base
        new_base = og_base.replace("\\", "/")

    updated_paths = []
    for path in mat_DAPI:
        p_str = str(path).replace("\\", "/")
        p_str = p_str.replace(old_base, new_base)
        updated_paths.append(p_str)

    print(f"Updated {len(updated_paths)} paths for {animal}.")
    print("Obtained all files for processing, ready to run trained model.\n")

    return updated_paths

def run_trained_model(path2files, new_model_path, do_3D=False, 
                      cellprob_threshold = 0.0, stitch_threshold = 0.2, tile_norm_blocksize = 0.0):
    # Evalute raw data by inputting raw file path
    # save masks into a mat file for further processing
 
    io.logger_setup()  # enable Cellpose logging
    useGPU = core.use_gpu()
    print(f"GPU available: {useGPU}")

    model_eval = models.CellposeModel(gpu=useGPU, pretrained_model=new_model_path)

    # --- handle input paths
    path2files = Path(path2files)
    imgfiles = io.get_image_files(str(path2files), mask_filter="_masks")

    print(f"Found {len(imgfiles)} image files to process.")

    for i in trange(len(imgfiles)):
        f = Path(imgfiles[i])
        fname = f.stem
        mask_name = f.parent / f"{fname}_masks.tif"
        flow_name = f.parent / f"{fname}_seg.npy"
        mat_name  = f.parent / f"{fname}_mask.mat"   # MATLAB output file

        # Skip if already processed
        if mask_name.exists() and flow_name.exists() and mat_name.exists():
            print(f"   Skipping {f.name} - results already exist.")
            continue

        print(f"Processing: {f.name}")

        # --- Load image
        data = io.imread(f)

        # --- Run Cellpose model
        masks, flows, styles = model_eval.eval(
            data, batch_size=32, channel_axis=-1, z_axis=0,
            stitch_threshold=stitch_threshold, do_3D=do_3D,
            normalize={"tile_norm_blocksize": tile_norm_blocksize},
            cellprob_threshold=cellprob_threshold
        )

        # --- Save Cellpose outputs into folder as the file path
        io.imsave(str(mask_name), masks)
        io.masks_flows_to_seg(data, masks, flows, str(f))

        # Include both masks and metadata if needed
        savemat(str(mat_name), {
            "mask": masks.astype(np.uint16),  # ensure integer type for MATLAB
            "filename": str(f),
            "model_used": str(new_model_path),
            "do_3D": do_3D,
            "cellprob_threshold": cellprob_threshold,
            "stitch_threshold": stitch_threshold
        })

        print(f"Saved results to {mat_name}")

    #print("All files processed successfully.")

def load_save_npy(fpath):
    filepath = Path(fpath)
    mask = np.load(filepath, allow_pickle=True).item()  # usually a dict containing 'masks', 'flows', etc.
    
    if isinstance(mask, dict):
        if "masks" in mask:
            seg = mask["masks"]
        else:
            raise KeyError(f"'masks' key not found in {filepath.name}. Keys = {list(mask.keys())}")
    else:
        seg = mask  # if file just contains an array
    
    # --- Save to .mat file
    mat_path = filepath.with_name(filepath.stem + "_mask.mat")
    savemat(mat_path, {"result": seg.astype(np.uint16)})
    
    print(f"Saved mask to: {mat_path}")
    print(f"Shape: {seg.shape}, dtype: {seg.dtype}")