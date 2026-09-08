#!/bin/bash
#SBATCH --job-name=xtts_accent_per_utt
#SBATCH --output=/data/user_data/xoy/slurm_logs/xtts_accent_per_utt_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/xtts_accent_per_utt_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
# Per-utterance accent cosine (+ CommonAccent label agreement) for one scored test set, folded
# into its eval_<dataset>.json as by_accent/by_speaker.accent_cosine.
#   sbatch eval/run_accent_per_utt.sbatch <dataset> [prompt_mode] [checkpoint_name]
set -euo pipefail
DATASET=${1:?dataset}
PROMPT_MODE=${2:-self}
CKPT_NAME=${3:-XTTS-v2-accent-finetune-filtered_full}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
ACCENT_VENV=/data/user_data/xoy/venvs/eval-accent
OUT_ROOT=/data/user_data/xoy/xtts_accent_eval/$CKPT_NAME/$PROMPT_MODE
echo "job $SLURM_JOB_ID on $(hostname) dataset=$DATASET"
"$ACCENT_VENV/bin/python" -u eval/score_accent_per_utt.py --dataset "$DATASET" \
  --wav_pairs_dir "$OUT_ROOT/$DATASET/wav_pairs_16k" \
  --results_path "$OUT_ROOT/eval_${DATASET}.json" \
  --per_utt_out "$OUT_ROOT/eval_${DATASET}_accent_per_utt.json"
echo "ACCENT_PER_UTT_DONE $DATASET"
