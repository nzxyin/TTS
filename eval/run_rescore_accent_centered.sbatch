#!/bin/bash
#SBATCH --job-name=xtts_rescore_accent_centered
#SBATCH --output=/data/user_data/xoy/slurm_logs/xtts_rescore_accent_centered_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/xtts_rescore_accent_centered_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
# The eval-genaid venv (torch 2.3 cu121) has no Blackwell kernels: exclude preempt's RTX_PRO_6000 nodes
# (same list as run_rescore_accent_genaid.sbatch / run_accent_per_utt.sbatch / run_synth.sbatch).
#SBATCH --exclude=babel-l5-16,babel-l5-20,babel-l5-28,babel-l5-32,babel-l9-16,babel-l9-20,babel-l9-28,babel-l9-32,babel-m9-16,babel-m9-20,babel-m9-28,babel-m9-32,babel-n9-20,babel-z5-16,babel-z5-20,babel-z5-28,babel-z5-32,babel-z9-16

# 2026-09-16 -- accent_cosine raw GenAID -> centroid-centered GenAID rescore (branch eval/accent-centering).
# Tracked in nzxyin/articulatory-tts issue #51 ("Decide: accent-cosine post-processing after the GenAID vs
# CommonAccent diagnostic") -- this fork has issues disabled, per eval/README.md's "Known issues" section.
#
# VCTK only: it's the only set with a per-accent breakdown and a meaningful accent axis (the other three
# sets carry accent_cosine only as a side metric with no by_accent view -- see run_rescore_accent_genaid.sbatch
# for the 4-dataset raw-GenAID rescore this supersedes for accent_cosine's headline value on VCTK). The 16 kHz
# wav pairs from that rescore are still on disk ($OUT_ROOT/vctk/wav_pairs_16k), so this only re-runs the accent
# side metric (articulatory-tts score_side_metric.py --metric accent, now with --center_vector) on them, merges
# with --keep_old_as genaid_raw (old raw-GenAID accent_cosine -> accent_cosine_genaid_raw; *_commonaccent keys
# from the 2026-09-14 rescore are left untouched throughout), and redoes the VCTK per-accent/per-speaker
# breakdown with eval/score_accent_per_utt.py (also now --center_vector-aware).
#
# Assumes articulatory-tts main has merged the centering API (genaid_accent.DEFAULT_CENTER_VECTOR /
# accent_cosines() / centered_model_tag(), score_side_metric.py --center_vector/--no_center) from branch
# feature/accent-centering -- fails loudly below if it hasn't.
#   sbatch eval/run_rescore_accent_centered.sbatch
set -euo pipefail
DATASET=vctk
PROMPT_MODE=${PROMPT_MODE:-self}
CKPT_NAME=${CKPT_NAME:-XTTS-v2-accent-finetune-filtered_full}
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export PYTHONUNBUFFERED=1
EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
ACCENT_VENV=/data/user_data/xoy/venvs/eval-genaid
ART_REPO=${ART_REPO:-/home/xoy/articulatory-tts}   # override to a worktree while feature/accent-centering is unmerged
OUT_ROOT=/data/user_data/xoy/xtts_accent_eval/$CKPT_NAME/$PROMPT_MODE
RESULTS=$OUT_ROOT/eval_${DATASET}.json
SIDE=$OUT_ROOT/eval_${DATASET}_accent_genaid_centered.json
WAV_PAIRS=$OUT_ROOT/$DATASET/wav_pairs_16k
echo "job $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dataset=$DATASET prompt=$PROMPT_MODE"
[ -s "$RESULTS" ] || { echo "no results JSON at $RESULTS"; exit 2; }
[ -d "$WAV_PAIRS" ] || { echo "no wav_pairs_16k dir at $WAV_PAIRS"; exit 2; }
echo "pairs on disk: $(ls "$WAV_PAIRS"/*_pred.wav | wc -l)"

# Resolve the centering vector once, explicitly, rather than relying on score_side_metric.py's own default --
# fails loudly here if articulatory-tts main hasn't merged the centering API yet, instead of mid-job.
CENTER_VECTOR=${CENTER_VECTOR:-$("$ACCENT_VENV/bin/python" -c "
import sys; sys.path.insert(0, '$ART_REPO')
import genaid_accent
print(genaid_accent.DEFAULT_CENTER_VECTOR)
")}
[ -s "$CENTER_VECTOR" ] || { echo "ERROR: center vector not found at '$CENTER_VECTOR' -- has feature/accent-centering merged into $ART_REPO main yet?"; exit 2; }
echo "center_vector: $CENTER_VECTOR"

is_centered_done() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics",{}).get("accent_cosine",{}); sys.exit(0 if isinstance(m,dict) and "centroid-centered" in str(m.get("model","")) else 1)' "$1" 2>/dev/null; }
# Guard against ruler-mixing: refuse to merge --keep_old_as genaid_raw onto a base that isn't raw-GenAID-scored
# yet (e.g. still CommonAccent-era) -- that would file a non-GenAID number under accent_cosine_genaid_raw.
is_raw_genaid() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("metrics",{}).get("accent_cosine",{}); model=str(m.get("model","")) if isinstance(m,dict) else ""; sys.exit(0 if ("GenAID" in model and "centroid-centered" not in model) else 1)' "$1" 2>/dev/null; }

if is_centered_done "$RESULTS"; then echo "aggregate already centroid-centered-scored"; else
  is_raw_genaid "$RESULTS" || { echo "ERROR: $RESULTS metrics.accent_cosine.model is not raw GenAID (expected the substring 'GenAID' present and 'centroid-centered' absent) -- refusing to merge onto a CommonAccent-era or already-centered base"; exit 1; }
  "$ACCENT_VENV/bin/python" -u "$ART_REPO/score_side_metric.py" --metric accent --center_vector "$CENTER_VECTOR" \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$SIDE"
  # provenance first (the merge flips the skip condition), then merge keeping the raw GenAID value
  python3 - "$RESULTS" "$(git -C "$ART_REPO" rev-parse HEAD)" "$(git rev-parse HEAD)" <<'PY'
import json, os, sys, datetime
p, art_sha, own_sha = sys.argv[1:4]
d = json.load(open(p))
d["accent_side_metric_source_genaid_raw"] = d.get("accent_side_metric_source")
d["accent_side_metric_source"] = f"articulatory-tts@{art_sha} score_side_metric.py (GenAID_v6, centroid-centered; genaid_accent.py; rescored {datetime.date.today().isoformat()} on the original wav pairs)"
d["eval_code_commit"] = own_sha
# per-group raw-GenAID accent_cosine survives under accent_cosine_genaid_raw before the per-utt pass
# overwrites it with the centered value; *_commonaccent keys (from the 2026-09-14 rescore) and
# accent_label_agreement (unaffected by centering) are left untouched.
for grp in ("by_accent", "by_speaker"):
    for g, e in d.get(grp, {}).items():
        if "accent_cosine" in e and "accent_cosine_genaid_raw" not in e:
            e["accent_cosine_genaid_raw"] = e.pop("accent_cosine")
tmp = p + ".tmp"; json.dump(d, open(tmp, "w"), indent=2); os.replace(tmp, p)
PY
  "$EVAL_VENV/bin/python" -u "$ART_REPO/merge_eval_results.py" --base "$RESULTS" --side "$SIDE" --keep_old_as genaid_raw
fi

OLD=$OUT_ROOT/eval_${DATASET}_accent_per_utt.json
# The per-utt file is a bare {uid: record} map with no top-level metadata (unlike $RESULTS), so it can never
# itself carry a "center_vector" key -- gate on accent_cosine_genaid_raw instead, which only a CENTERED run of
# score_accent_per_utt.py writes per-record (an uncentered/raw run never writes that key at all).
if [ -s "$OLD" ] && ! grep -q '"accent_cosine_genaid_raw"' "$OLD"; then mv "$OLD" "${OLD%.json}_genaid_raw.json"; fi
"$ACCENT_VENV/bin/python" -u eval/score_accent_per_utt.py --dataset "$DATASET" --center_vector "$CENTER_VECTOR" \
  --wav_pairs_dir "$WAV_PAIRS" --results_path "$RESULTS" --per_utt_out "$OLD"

python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["metrics"]; print("accent_cosine centered", m["accent_cosine"]["mean"], "n", m["accent_cosine"]["n"], "| raw GenAID", m.get("accent_cosine_genaid_raw",{}).get("mean"), "| CommonAccent", m.get("accent_cosine_commonaccent",{}).get("mean"))' "$RESULTS"
echo "RESCORE_DONE $DATASET centered"
