#!/bin/bash
#SBATCH --job-name=xtts_synth
#SBATCH --output=/data/user_data/xoy/slurm_logs/xtts_synth_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/xtts_synth_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
# The xtts-accent venv's torch 2.6 (cu124) has no kernels for Blackwell GPUs; preempt's 18
# RTX_PRO_6000 nodes fail every CUDA launch ("no kernel image is available", job 10350518).
# Regenerate with: sinfo -p preempt -h -N -o "%N %G" | grep RTX_PRO | awk '{print $1}' | sort -u
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# Synthesize one shard of one test set with an accented XTTS-v2 checkpoint.
#   sbatch eval/run_synth.sbatch <dataset> <shard_index> <num_shards> [prompt_mode] [checkpoint_name]
# e.g. sbatch eval/run_synth.sbatch libritts_test_clean 0 4
# preempt per the standing partition preference (evals are interruption-
# tolerant): synthesize_testset.py skips wavs that already exist, so a
# requeued job resumes where it was.
set -euo pipefail
DATASET=${1:?dataset}
SHARD=${2:-0}
NSHARDS=${3:-1}
PROMPT_MODE=${4:-self}
CKPT_NAME=${5:-XTTS-v2-accent-finetune-filtered_full}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
VENV=/data/user_data/xoy/venvs/xtts-accent
CKPT_DIR=/data/user_data/xoy/xtts_accent_checkpoints/$CKPT_NAME
OUT_ROOT=/data/user_data/xoy/xtts_accent_eval/$CKPT_NAME
OUT_DIR=$OUT_ROOT/${PROMPT_MODE}/$DATASET
mkdir -p "$OUT_DIR"
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET shard=$SHARD/$NSHARDS prompt=$PROMPT_MODE ckpt=$CKPT_NAME"
"$VENV/bin/python" -u eval/synthesize_testset.py \
  --dataset "$DATASET" --checkpoint_dir "$CKPT_DIR" --out_dir "$OUT_DIR" \
  --prompt_mode "$PROMPT_MODE" --shard_index "$SHARD" --num_shards "$NSHARDS"
echo "SYNTH_DONE $DATASET $SHARD/$NSHARDS"
