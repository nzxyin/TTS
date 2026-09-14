# Changelog

## 2026-09-14 -- accent similarity: CommonAccent -> GenAID (branch `accent_eval`)

- The accent side metric now uses GenAID (https://github.com/jzmzhong/GenAID,
  GenAID_v6; speaker-adversarial XLSR-53 accent ID, 64-dim embedding), via the
  reference repo's `genaid_accent.py` and its new `eval-genaid` venv; the
  articulatory-tts `score_side_metric.py` that `eval/run_score.sbatch` calls
  made the same switch. `eval/score_accent_per_utt.py` embeds with the same
  code and records GenAID's 13-way top label for prediction and ground truth
  (`pred_label`/`gt_label`, replacing the CommonAccent 16-way fields).
  `run_score.sbatch`, `run_accent_per_utt.sbatch`, `smoke_test.sbatch` point
  at the new venv and exclude the Blackwell nodes it cannot run on.
- `eval/run_rescore_accent_genaid.sbatch` (job 10441102, 4 tasks) rescores
  the four result JSONs on their kept 16 kHz wav pairs; CommonAccent values
  stay as `metrics.accent_cosine_commonaccent` (and `*_commonaccent` keys in
  `by_accent`/`by_speaker`, `eval_vctk_accent_per_utt_commonaccent.json`).
  Tracked in the articulatory-tts issue "Track: accent_cosine CommonAccent ->
  GenAID rescore" (this fork has issues disabled).
- `eval/README.md`: the SPARC and articulatory reference rows of the VCTK
  table now carry their GenAID accent cosines (0.966 / 0.857 / 0.919) instead
  of the CommonAccent values they still showed next to XTTS's GenAID cell.

## 2026-09-08 -- per-accent accent similarity (branch `accent_eval`)

- Added `eval/score_accent_per_utt.py` + `eval/run_accent_per_utt.sbatch`:
  recomputes the CommonAccent embedding cosine per utterance (same model and
  `encode_batch` path as articulatory-tts `score_side_metric.py`, asserts the
  overall mean reproduces the merged `accent_cosine`), folds per-accent and
  per-speaker `accent_cosine` into `eval_<dataset>.json`, and records the
  classifier's top label for prediction and ground truth as an
  accent-classification side view. Run for VCTK (job 10354552); results
  table added to `eval/README.md`.
- Aligned `wer_whisper_normalized` with articulatory-tts's GH #32 rule
  (commit b040aa1): pairs whose normalized reference is empty are skipped
  (was: kept if the hypothesis was non-empty). Re-derived from the saved
  per-utterance texts for all four result JSONs (test-clean 2.5947% ->
  2.5892%, test-other 2.9566% -> 2.9540%; LJSpeech/VCTK unchanged) and
  recorded the rule in each JSON. Normalized WER is now the headline WER in
  `eval/README.md`, matching the sibling CosyVoice3/EmoSphere++ evals.
- `eval/run_score.sbatch` now excludes `babel-l9-16` (onnxruntime import
  failure there silently disables torchmetrics DNSMOS; reported by a
  sibling session).

## 2026-09-07 -- accented-XTTS evaluation on the articulatory-tts test sets (branch `accent_eval`)

- Added `eval/` pipeline: `synthesize_testset.py` (sharded, resumable,
  per-utterance-seeded synthesis of LJSpeech / LibriTTS-R test-clean /
  test-other / VCTK held-out splits with self or cross speaker prompts and
  per-dataset accent tags), `score_synthesized.py` (port of
  articulatory-tts `eval_full_testset.py`'s metrics: Whisper-large-v3 corpus
  WER + normalized variant, UTMOSv2, DNSMOS, ECAPA speaker cosine; writes
  16 kHz wav pairs for the accent side metric), SLURM wrappers
  (`run_synth.sbatch`, `run_score.sbatch` with completeness gate and
  provenance, `smoke_test.sbatch`, `build_env.sbatch`), pinned
  `requirements-xtts-accent.txt`, and `eval/README.md` (protocol, results,
  known issues).
- Restored `requirements.txt` to upstream's pip-style list; the conda export
  that `accent_release` shipped under that name is now
  `requirements.conda-export.txt` (setuptools rejected it at
  `pip install -e .`).
- Built `/data/user_data/xoy/venvs/xtts-accent` (Python 3.10, torch 2.6.0,
  transformers 4.51.2) and downloaded `HSTE/XTTS-v2-accent-finetune-filtered_full`
  to `/data/user_data/xoy/xtts_accent_checkpoints/`.
- Evaluated the filtered checkpoint (self prompt) on all four test sets;
  summary JSONs in `eval/results/XTTS-v2-accent-finetune-filtered_full/self/`.
- Moved the scorers' `~/.cache/utmosv2` and `~/.cache/huggingface/hub` to
  `/data/user_data/xoy/.cache/` with symlinks left behind.
- Added `CLAUDE.md` and this changelog.
