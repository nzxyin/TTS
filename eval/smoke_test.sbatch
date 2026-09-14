#!/bin/bash
#SBATCH --job-name=xtts_smoke
#SBATCH --output=/data/user_data/xoy/slurm_logs/xtts_smoke_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/xtts_smoke_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# The xtts-accent venv's torch 2.6 (cu124) has no kernels for Blackwell GPUs; preempt's 18
# RTX_PRO_6000 nodes fail every CUDA launch ("no kernel image is available", job 10350518).
# Regenerate with: sinfo -p preempt -h -N -o "%N %G" | grep RTX_PRO | awk '{print $1}' | sort -u
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# End-to-end smoke test on 3 utterances per test set: synthesis (xtts-accent
# venv) -> scoring (eval-articulatory-tts venv) -> accent side metric
# (eval-accent venv). Writes under .../xtts_accent_eval/<ckpt>/smoke/ so it
# never collides with the real runs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
VENV=/data/user_data/xoy/venvs/xtts-accent
EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
CKPT_NAME=XTTS-v2-accent-finetune-filtered_full
CKPT_DIR=/data/user_data/xoy/xtts_accent_checkpoints/$CKPT_NAME
OUT_ROOT=/data/user_data/xoy/xtts_accent_eval/$CKPT_NAME/smoke
LIMIT=${LIMIT:-3}
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for ds in ljspeech vctk libritts_test_clean libritts_test_other; do
  echo "=== synth $ds ==="
  rm -f "$OUT_ROOT/$ds"/manifest.shard*of*.jsonl "$OUT_ROOT/$ds"/manifest.shard*of*.jsonl.done "$OUT_ROOT/$ds/split_info.json"
  "$VENV/bin/python" -u eval/synthesize_testset.py --dataset $ds --checkpoint_dir "$CKPT_DIR" \
     --out_dir "$OUT_ROOT/$ds" --limit "$LIMIT" --overwrite
done
for ds in ljspeech vctk libritts_test_clean libritts_test_other; do
  echo "=== score $ds ==="
  "$EVAL_VENV/bin/python" -u eval/score_synthesized.py --dataset $ds --out_dir "$OUT_ROOT/$ds" --limit "$LIMIT" \
     --checkpoint "$CKPT_NAME" --results_path "$OUT_ROOT/eval_${ds}.json" \
     --per_utt_out "$OUT_ROOT/eval_${ds}_per_utt.json" --wav_pairs_dir "$OUT_ROOT/$ds/wav_pairs_16k"
done
echo "=== accent side metric (vctk) ==="
"$ACCENT_VENV/bin/python" -u /home/xoy/articulatory-tts/score_side_metric.py --metric accent \
   --wav_pairs_dir "$OUT_ROOT/vctk/wav_pairs_16k" --results_path "$OUT_ROOT/eval_vctk_accent.json"
"$EVAL_VENV/bin/python" -u /home/xoy/articulatory-tts/merge_eval_results.py --base "$OUT_ROOT/eval_vctk.json" --side "$OUT_ROOT/eval_vctk_accent.json"
echo "SMOKE_DONE"
