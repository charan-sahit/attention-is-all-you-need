#!/bin/bash
#SBATCH -A research
#SBATCH -p short
#SBATCH --nodelist=gnode073
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=4G
#SBATCH --time=05:00:00
#SBATCH --job-name=transformer-iwslt
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL

# ---- environment -------------------------------------------------------
# Adjust CUDA module name to whatever `module avail` shows on Ada.
# The wiki examples use u18/cuda/10.1; for a modern PyTorch you'll want a
# newer one (11.x or 12.x). Check `module avail cuda` before submitting.
module load u18/cuda/11.8 2>/dev/null || module load u18/cuda/10.1
module load u18/cudnn/8-cuda-11.8 2>/dev/null || true

# Activate your Python environment (edit path if using venv instead).
# Replace 'attention' with the actual env name you set up.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate attention

# ---- project paths -----------------------------------------------------
PROJECT_DIR="$HOME/Documents/ml-papers/attention"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs checkpoints

# ---- runtime info ------------------------------------------------------
echo "=========================================================="
echo "Job:       $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Started:   $(date)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- run -------------------------------------------------------------
# torchrun spawns one process per GPU. --standalone picks a free
# rendezvous port automatically (single-node DDP).
torchrun \
    --standalone \
    --nproc_per_node=3 \
    -m train.train

echo "Finished: $(date)"
