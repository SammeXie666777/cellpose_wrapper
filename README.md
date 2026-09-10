# cellpose_wrapper

Wrapper around [Cellpose-SAM](https://github.com/MouseLand/cellpose) for 3D nuclei segmentation
of light-sheet / confocal volumes, with MATLAB-friendly outputs and SGE cluster batching.

Fine-tune the `cpsam` model on a handful of annotated volumes, then run the trained model over a
whole animal's image set — either on one machine or split across GPU jobs on an SGE cluster.
Every segmented volume gets a `.mat` companion file so downstream analysis can stay in MATLAB.

## Pipeline

```
Train_CellposeSAM.py     fine-tune cpsam on paired raw/mask volumes -> trained model
        |
        v
Run_Cellpose.py          single-machine inference; also resolves image paths from a .mat manifest
        |
        v
run_cellpose_batch.py    chunk pending images and submit them as parallel SGE GPU jobs
submit_cellpose_job.sh   launcher wrapper (edit the USER SETTINGS block, then qsub)
        |
        v
Visualization.py         overlay masks on raw volumes as an ImageJ-readable RGB TIFF for QC
```

## Files

| File | Role |
|---|---|
| `Train_CellposeSAM.py` | Slices 3D volumes into aligned YX/ZY/ZX crops and fine-tunes `cpsam`. |
| `Run_Cellpose.py` | `convert_mat_paths()` maps Windows/cluster paths; `run_trained_model()` segments a directory. |
| `run_cellpose_batch.py` | Two modes: `--launch` (chunk + submit jobs) and `--worker` (process one manifest). |
| `submit_cellpose_job.sh` | SGE launcher; sets animal, model path, job count and thresholds. |
| `Visualization.py` | Writes a colored mask overlay TIFF for visual QC. |
| `utils/env.py` | `require_package()` — imports a package, pip-installing it if missing. |

## Outputs

For each input image, alongside the source file:

- `<name>_seg.npy` — Cellpose masks + flows
- `<name>_mask.mat` — masks as `uint16` plus the parameters used, for MATLAB
- `<name>_overlay.tif` — optional QC overlay

Already-processed images are skipped, so runs are resumable.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
# edit root_dir / model_name at the top of the file first
python Train_CellposeSAM.py
```

Segment on one machine:

```python
from Run_Cellpose import run_trained_model
run_trained_model("/path/to/images", "/path/to/model", do_3D=True)
```

Segment on an SGE cluster:

```bash
# edit the USER SETTINGS block, then
qsub submit_cellpose_job.sh
```

## Notes

- Paths are environment-aware: `convert_mat_paths()` detects whether it is running locally or on
  the cluster and rewrites the drive-letter base accordingly.
- Cluster settings (`-P jchenlab`, `module load cellpose/4.0.4`, the `/net/...` mounts) are
  specific to the BU SCC and will need editing elsewhere.
- `require_package()` pip-installs missing packages at import time; on a cluster, prefer a
  pre-built venv via `SCC_CELLPOSE_VENV`.
