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
# The eval-genaid venv (GenAID accent metric, torch 2.3 cu121) has no Blackwell kernels: exclude preempt's
# RTX_PRO_6000 nodes (same list as run_synth.sbatch; babel-l9-16 is among them).
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16
# Per-utterance accent cosine (+ CommonAccent label agreement) for one scored test set, folded
# into its eval_<dataset>.json as by_accent/by_speaker.accent_cosine.
#   sbatch eval/run_accent_per_utt.sbatch <dataset> [prompt_mode] [checkpoint_name]
set -euo pipefail
DATASET=${1:?dataset}
PROMPT_MODE=${2:-self}
CKPT_NAME=${3:-XTTS-v2-accent-finetune-filtered_full}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
OUT_ROOT=/data/user_data/xoy/xtts_accent_eval/$CKPT_NAME/$PROMPT_MODE
echo "job $SLURM_JOB_ID on $(hostname) dataset=$DATASET"
"$ACCENT_VENV/bin/python" -u eval/score_accent_per_utt.py --dataset "$DATASET" \
  --wav_pairs_dir "$OUT_ROOT/$DATASET/wav_pairs_16k" \
  --results_path "$OUT_ROOT/eval_${DATASET}.json" \
  --per_utt_out "$OUT_ROOT/eval_${DATASET}_accent_per_utt.json"
echo "ACCENT_PER_UTT_DONE $DATASET"
