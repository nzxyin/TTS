#!/bin/bash
#SBATCH --job-name=xtts_env_build
#SBATCH --output=/data/user_data/xoy/slurm_logs/xtts_env_build_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/xtts_env_build_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# One-shot setup for the accented-XTTS evaluation: builds the Python 3.10
# inference venv on /data (login node can't -- no /data mount, 1h CPU rlimit)
# and, in parallel, downloads the "filtered" checkpoint from the HF Hub.
# CPU-only (preempt_cpu_qos) -- nothing here needs a GPU. Safe to re-run:
# uv/hf both skip work that's already done.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/data/user_data/xoy/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/user_data/xoy/.local/share/uv/python
export PIP_CACHE_DIR=/data/user_data/xoy/.cache/pip
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

VENV=/data/user_data/xoy/venvs/xtts-accent
CKPT_ROOT=/data/user_data/xoy/xtts_accent_checkpoints
mkdir -p "$CKPT_ROOT" /data/user_data/xoy/slurm_logs

# --- 1) checkpoint download, in the background ------------------------------
DL_LOG=/data/user_data/xoy/slurm_logs/xtts_ckpt_download_${SLURM_JOB_ID}.log
(
  set +e
  hf download HSTE/XTTS-v2-accent-finetune-filtered_full \
     --local-dir "$CKPT_ROOT/XTTS-v2-accent-finetune-filtered_full"
  echo "hf download exit code: $?"
) > "$DL_LOG" 2>&1 &
DL_PID=$!

# --- 2) inference venv -------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.10 "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r eval/requirements-xtts-accent.txt
# TTS itself: --no-deps because accent_release's requirements.txt is a conda
# export that setup.py's install_requires parser can't read; deps are already
# satisfied by the pinned file above.
uv pip install --python "$VENV/bin/python" --no-deps -e .
# uv venvs ship without pip; the trainer package shells out to `pip` at import
# in some code paths, so make sure it exists.
uv pip install --python "$VENV/bin/python" pip

echo "=== import smoke test ==="
"$VENV/bin/python" - <<'PY'
import torch, transformers, numpy, TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "numpy", numpy.__version__)
print("TTS import OK")
PY

# repo-local .venv symlink for tools that expect it there (convention: bulk lives on /data)
if [ ! -e .venv ]; then ln -s "$VENV" .venv; fi

# --- 3) wait for the download -----------------------------------------------
wait "$DL_PID" || true
echo "=== download log ==="; cat "$DL_LOG"
grep -q "hf download exit code: 0" "$DL_LOG" || { echo "checkpoint download FAILED (see log above)"; exit 1; }
for f in model.pth dvae.pth mel_stats.pth vocab.json config.json; do
  [ -s "$CKPT_ROOT/XTTS-v2-accent-finetune-filtered_full/$f" ] || { echo "missing checkpoint file $f"; exit 1; }
done
echo "=== checkpoint dir ==="; ls -la "$CKPT_ROOT/XTTS-v2-accent-finetune-filtered_full"
echo "BUILD_DONE"
