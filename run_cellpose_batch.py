#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 00:09:14 2025

@author: sammxie
"""

import argparse
#import sys
#sys.path.append("/net/128.197.168.185/mnt/data/Dropbox/Chen Lab Dropbox/Chen Lab Team Folder/Projects/Starotator/Cellpose_SAM/Scripts/")
import math
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import savemat
from tqdm import trange
from cellpose import io, models, core
from Run_Cellpose import convert_mat_paths  # keep your existing function
import time

def list_images_needing_processing(fpath):
    """
    Given a list of directories, return a list of image file paths that
    still need processing (i.e., missing any of: *_masks.tif, *_seg.npy, *_mask.mat).
    """
    pending = []
    for f in fpath:
        f = Path(f)
        #imgfiles = io.get_image_files(str(d), mask_filter="_masks")
        #for f in imgfiles:
        stem = f.stem
        flow_name = f.parent / f"{stem}_seg.npy"
        mat_name  = f.parent / f"{stem}_mask.mat"

        if not (flow_name.exists() and mat_name.exists()):
            pending.append(str(f))
    return pending


def chunk_list(seq, n_chunks):
    if n_chunks <= 1:
        return [seq]
    size = math.ceil(len(seq) / n_chunks)
    return [seq[i:i+size] for i in range(0, len(seq), size)]


def write_manifest(manifest_path, file_list):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        for p in file_list:
            fh.write(p + "\n")
    return str(manifest_path)


def write_qsub_script(script_path, manifest_path, args, job_idx):
    script_path = Path(script_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.job_base) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"cellpose_job_{args.animal}_{job_idx:03d}"
    py = Path(__file__).resolve()
    do3d = "--do_3D" if args.do_3D else ""

    sge = f"""#!/bin/bash -l
#$ -P jchenlab
#$ -N {job_name}
#$ -pe omp 16
#$ -l avx2
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd
#$ -o {log_dir}/{job_name}.$JOB_ID.out
#$ -e {log_dir}/{job_name}.$JOB_ID.err

set -euo pipefail

# Ensure log dir exists at runtime too (paranoia on exec host)
mkdir -p "{log_dir}"

START_TIME=$(date +%s)
echo "[${{HOSTNAME}}] $(date) START {job_name}"
echo "Manifest: {manifest_path}"

module load cellpose/4.0.4

# Activate venv only if SCC_CELLPOSE_VENV is set and exists
if [ -n "${{SCC_CELLPOSE_VENV:-}}" ] && [ -f "$SCC_CELLPOSE_VENV/bin/activate" ]; then
  echo "Activating venv at $SCC_CELLPOSE_VENV"
  # shellcheck disable=SC1090
  source "$SCC_CELLPOSE_VENV/bin/activate"
else
  echo "Warning: SCC_CELLPOSE_VENV is not set or invalid; continuing without venv."
fi

python "{py}" \\
  --worker \\
  --file_list "{manifest_path}" \\
  --model_path "{args.model_path}" \\
  --cellprob_threshold {args.cellprob_threshold} \\
  --stitch_threshold {args.stitch_threshold} \\
  --tile_norm_blocksize {args.tile_norm_blocksize} \\
  {do3d}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[${{HOSTNAME}}] $(date) DONE {job_name} in $ELAPSED s (~$((ELAPSED/60)) min)"
"""

    script_path.write_text(sge)
    return str(script_path)


def submit_qsub(script_path):
    """Submit a PBS script and return the job ID (or stdout)."""
    res = subprocess.run(["qsub", script_path], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"qsub failed for {script_path}:\n{res.stderr}")
    return res.stdout.strip()


def run_trained_model(
    path2files,
    new_model_path,
    do_3D=False,
    cellprob_threshold=0.0,
    stitch_threshold=0.2,
    tile_norm_blocksize=0.0,
    batch_size=32,
):
    """
    Evaluate raw data by inputting a directory OR a newline-separated manifest file of image paths.
    Saves masks/flows and a small .mat metadata file for MATLAB.

    path2files can be:
      - a directory path (string)
      - a text file ending with .txt containing newline-separated absolute paths to images
    """
    io.logger_setup()  # enable Cellpose logging
    useGPU = core.use_gpu()
    print(f"GPU available: {useGPU}")

    model_eval = models.CellposeModel(gpu=useGPU, pretrained_model=new_model_path)

    path2files = Path(path2files)
    if path2files.is_file() and path2files.suffix.lower() == ".txt":
        # Manifest mode
        with path2files.open("r") as fh:
            imgfiles = [line.strip() for line in fh if line.strip()]
    else:
        # Directory mode
        imgfiles = io.get_image_files(str(path2files), mask_filter="_masks")

    print(f"Found {len(imgfiles)} image files to process in this chunk.")

    for i in trange(len(imgfiles)):
        
        f = Path(imgfiles[i])
        fname = f.stem
        flow_name = f.parent / f"{fname}_seg.npy"
        mat_name  = f.parent / f"{fname}_mask.mat"

        # Skip if already processed
        if flow_name.exists() and mat_name.exists():
            print(f"Skipping {f.name} - results already exist.")
            continue
        start_time = time.time() 
        print(f"Processing: {f.name}")

        # --- Load image
        data = io.imread(f)

        # --- Run Cellpose model
        masks, flows, styles = model_eval.eval(
            data,
            batch_size=batch_size,
            channel_axis=-1,
            z_axis=0,
            stitch_threshold=stitch_threshold,
            do_3D=do_3D,
            normalize={"tile_norm_blocksize": tile_norm_blocksize},
            cellprob_threshold=cellprob_threshold,
        )

        # --- Save Cellpose outputs
        #o.imsave(str(mask_name), masks)
        io.masks_flows_to_seg(data, masks, flows, str(f))

        # --- Save lightweight MATLAB companion file
        savemat(str(mat_name), {
            "mask": masks.astype(np.uint16),
            "filename": str(f),
            "model_used": str(new_model_path),
            "do_3D": do_3D,
            "cellprob_threshold": cellprob_threshold,
            "stitch_threshold": stitch_threshold,
        })
        elapsed = time.time() - start_time  # <-- stop timer
        print(f"Saved results to {mat_name}")
        print(f"Elapsed time for {f.name}: {elapsed:.2f} seconds (~{elapsed/60:.2f} minutes)\n")


def main():
    parser = argparse.ArgumentParser(description="Launch or run chunked Cellpose jobs.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true", help="Discover pending files, split into jobs, write & submit PBS scripts.")
    mode.add_argument("--worker", action="store_true", help="Process a provided --file_list (called by PBS scripts).")

    # Common args
    parser.add_argument("--model_path", required=True, help="Path to trained Cellpose model (.npy or .pth).")
    parser.add_argument("--do_3D", action="store_true", help="Enable 3D mode.")
    parser.add_argument("--cellprob_threshold", type=float, default=0.0, help="Cell probability threshold.")
    parser.add_argument("--stitch_threshold", type=float, default=0.2, help="Stitch threshold.")
    parser.add_argument("--tile_norm_blocksize", type=float, default=0.0, help="Tile normalization block size.")

    # Launch mode args
    parser.add_argument("--animal", help="Animal name (folder under the project).")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of PBS jobs/chunks to submit.")
    parser.add_argument("--job_base", default="/tmp/cellpose_jobs", help="Where to write qsub scripts, manifests, and logs.")
    parser.add_argument("--og_base", default="X:/Projects/ARG/Animals", help="Local base path (for convert_mat_paths).")
    parser.add_argument("--scc_base", default="/net/claustrum2/mnt/data/Projects/ARG/Animals", help="SCC base path (for convert_mat_paths).")

    # Worker mode args
    parser.add_argument("--file_list", help="Path to a newline-separated manifest of image files for this worker.")
    parser.add_argument("--path2files", help="(Alternative) A single directory to process in worker mode.")

    args = parser.parse_args()

    if args.launch:
        if not args.animal:
            raise ValueError("--animal is required in --launch mode")

        # 1) Convert .mat paths to SCC dirs via your existing helper
        fpath = convert_mat_paths(
            animal=args.animal,
            og_base=args.og_base,
            scc_base=args.scc_base
        )
        if isinstance(fpath, (list, tuple)):
            print(f"\nFound {len(fpath)} converted paths.")
            preview = fpath[:2] if len(fpath) >= 2 else fpath
            for i, p in enumerate(preview, start=1):
                print(f"  [{i}] {p}")
        else:
            print(f"convert_mat_paths() returned a single path: {fpath}")

        # 2) Find only images that still need processing
        pending = list_images_needing_processing(fpath)
        if len(pending) == 0:
            print("No pending images found. Nothing to do.")
            return

        # 3) Chunk into n_jobs
        chunks = chunk_list(pending, args.n_jobs)

        # 4) Write manifests & PBS scripts, then submit
        manifests_dir = Path(args.job_base) / "manifests" / args.animal
        scripts_dir   = Path(args.job_base) / "qsubs" / args.animal
        manifests_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        print(f"Discovered {len(pending)} images across {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks, start=1):
            manifest_path = manifests_dir / f"{args.animal}_chunk_{i:03d}.txt"
            script_path   = scripts_dir   / f"{args.animal}_chunk_{i:03d}.pbs"

            write_manifest(manifest_path, chunk)
            write_qsub_script(script_path, manifest_path, args, i)
            job_id = submit_qsub(str(script_path))
            print(f"Submitted {script_path.name} -> {job_id}")

    elif args.worker:
        # Worker accepts either a manifest or a single directory
        if args.file_list:
            run_trained_model(
                path2files=args.file_list,
                new_model_path=args.model_path,
                do_3D=args.do_3D,
                cellprob_threshold=args.cellprob_threshold,
                stitch_threshold=args.stitch_threshold,
                tile_norm_blocksize=args.tile_norm_blocksize
            )
        elif args.path2files:
            run_trained_model(
                path2files=args.path2files,
                new_model_path=args.model_path,
                do_3D=args.do_3D,
                cellprob_threshold=args.cellprob_threshold,
                stitch_threshold=args.stitch_threshold,
                tile_norm_blocksize=args.tile_norm_blocksize
            )
        else:
            raise ValueError("Worker mode needs --file_list (preferred) or --path2files.")
    else:
        raise RuntimeError("Invalid mode selection.")


if __name__ == "__main__":
    main()
