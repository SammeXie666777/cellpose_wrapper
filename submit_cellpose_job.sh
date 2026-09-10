#!/bin/bash -l
# ---------------- SGE PARAMETERS (launcher) ----------------
#$ -P jchenlab                     # project
#$ -N cellpose_launch              # job name
#$ -pe omp 1                       # 1 core is enough for the launcher
#$ -l avx2                         # match your cluster norm
#$ -l h_rt=01:00:00                # short runtime; just submits children
#$ -j y                            # merge stderr into stdout
#$ -m n                            # no emails
#$ -cwd                            # run from submit dir
#$ -o /net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/job_logs/launcher.$JOB_ID.out
#$ -e /net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/job_logs/launcher.$JOB_ID.err

# ---------------- USER SETTINGS ----------------
ANIMAL="Sparrow"                          # animal folder name
MODEL_PATH="/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/Sparrow_test/Model_100725/train/models/Marmoset_NewModel"
N_JOBS=2                                 # how many chunks/jobs to submit
DO_3D_FLAG="--do_3D"                        # set to "" to disable 3D
CELLPROB=0.0
STITCH=0.2
TILEBLOCK=0.0

# (Override bases used by convert_mat_paths
OG_BASE='Y:\Projects\Marmoset\Animals\'
SCC_BASE="/net/128.197.168.185/mnt/data1/Projects/Marmoset/Animals/"

# Where to save generated qsub scripts, manifests, and logs
JOB_BASE="/net/claustrum4/mnt/storage/data/Projects/Marmoset/cellpose/job_logs"
mkdir -p "${JOB_BASE}"

# ---------------- ENVIRONMENT SETUP ----------------
module load cellpose/4.0.4
source "$SCC_CELLPOSE_VENV/bin/activate"

SCRIPT_DIR="/usr3/bustaff/sammxie/CellPose/Script"
cd "$SCRIPT_DIR"

echo "Launching chunked Cellpose jobs for ANIMAL=${ANIMAL}, N_JOBS=${N_JOBS}"

set -euo pipefail
START_TIME=$(date +%s)
echo "[${HOSTNAME}] $(date) Launching chunked Cellpose jobs for ANIMAL=${ANIMAL}, N_JOBS=${N_JOBS}"

python run_cellpose_batch.py \
    --launch \
    --animal "${ANIMAL}" \
    --model_path "${MODEL_PATH}" \
    --n_jobs ${N_JOBS} \
    --cellprob_threshold ${CELLPROB} \
    --stitch_threshold ${STITCH} \
    --tile_norm_blocksize ${TILEBLOCK} \
    --job_base "${JOB_BASE}" \
    --og_base "${OG_BASE}" \
    --scc_base "${SCC_BASE}" \
    ${DO_3D_FLAG}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[${HOSTNAME}] $(date) Done submitting chunked jobs."
echo "Total launcher runtime: ${ELAPSED} seconds (~$((ELAPSED/60)) minutes)"

