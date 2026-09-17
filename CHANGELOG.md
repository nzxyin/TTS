# Changelog

## 2026-09-17 -- checked-in LJSpeech / test-clean / test-other JSONs refreshed; centering provenance note (branch `eval/refresh-result-jsons`)

- `eval/results/XTTS-v2-accent-finetune-filtered_full/self/eval_{ljspeech,libritts_test_clean,libritts_test_other}.json`
  copied from their `/data` counterparts: the checked-in copies still carried the CommonAccent-era
  `metrics.accent_cosine` (0.717 / 0.785 / 0.639, no `model` tag) although README/CLAUDE.md already quoted
  the 2026-09-14 raw GenAID rescore (0.950 / 0.963 / 0.906). They now hold the GenAID value with
  `accent_cosine_commonaccent` alongside, like `eval_vctk.json`; no other metric changed. Found by the
  2026-09-17 post-merge audit of the accent-centering change.
- CLAUDE.md: provenance note that `eval_vctk.json` records the centering vector by the removed
  `/home/xoy/wt-center` worktree path (byte-identical file on articulatory-tts main; tracked as
  nzxyin/articulatory-tts issue #57).

## 2026-09-17 -- VCTK accent-centering rescore complete (job 10473692); docs and results JSON refreshed (branch `eval/accent-centering`)

- `eval/run_rescore_accent_centered.sbatch` (job 10473692, VCTK only) completed and is verified: the merged
  `eval_vctk.json`'s `metrics.accent_cosine.model` contains `"centroid-centered"` and its mean is 0.6253
  (n=2596, ci95 0.013). Corpus accent cosine: **0.625 +/- 0.013 (95% CI)**, centered GenAID -- raw GenAID
  0.930 (`accent_cosine_genaid_raw`) and CommonAccent 0.663 (`accent_cosine_commonaccent`) are kept as
  superseded context, not deleted.
- Per-accent (centered, +/- 95% CI): American 0.784 +/- 0.019, Canadian 0.813 +/- 0.015, English 0.746 +/-
  0.016, Irish 0.247 +/- 0.041, Northern Irish 0.425 +/- 0.030, Scottish 0.711 +/- 0.019 -- same ranking as
  the raw GenAID sextet (0.950 / 0.962 / 0.939 / 0.879 / 0.919 / 0.928 respectively), but the spread widens
  from 0.083 raw to 0.566 centered; the label-agreement view (pred/GT labelled with the target class) is
  unchanged by centering.
- `eval/README.md` and `CLAUDE.md` updated throughout: every place reporting the VCTK accent cosine (headline
  table, per-accent table, "Current state" notes) now shows the centered value with the job id, with raw
  GenAID and CommonAccent kept alongside as clearly-marked superseded context. The LJSpeech / test-clean /
  test-other tables, which this rescore did not touch, are now explicitly labelled "accent cos (raw)" so they
  are not read against VCTK's centered column.
- Refreshed the checked-in
  `eval/results/XTTS-v2-accent-finetune-filtered_full/self/eval_vctk.json` from
  `/data/user_data/xoy/xtts_accent_eval/XTTS-v2-accent-finetune-filtered_full/self/eval_vctk.json` -- the
  repo copy had been stale since before the 2026-09-14 GenAID rescore (it still showed the CommonAccent-era
  `accent_cosine` of 0.663 as the headline metric with no `accent_cosine_genaid_raw` /
  `accent_cosine_commonaccent` split). Only this one file was refreshed; the other three `eval_*.json`
  copies under `eval/results/` were out of scope for this change and are untouched.

## 2026-09-16 -- accent similarity: raw GenAID -> centroid-centered GenAID (branch `eval/accent-centering`)

- The accent side metric now reports the CENTERED GenAID cosine as the headline `accent_cosine` -- both the
  prediction and ground-truth embeddings have `genaid_accent.DEFAULT_CENTER_VECTOR` (the mean of the six
  speaker-balanced VCTK-training-speaker accent centroids) subtracted before the cosine, per the
  articulatory-tts diagnostic (that repo's `CLAUDE.md` "Accent-metric diagnostic" section, decision
  2026-09-16; tracked in `nzxyin/articulatory-tts` issue #51 -- this fork has issues disabled, see
  `eval/README.md` "Known issues"). The raw (uncentered) GenAID cosine is kept alongside as
  `accent_cosine_genaid_raw`; the `*_commonaccent` keys from the 2026-09-14 CommonAccent -> GenAID rescore
  are untouched throughout.
- `eval/score_accent_per_utt.py`: added `--center_vector`/`--no_center` (same flags as the reference repo's
  `score_side_metric.py`/`score_side_per_utt.py`). Per-utterance and per-group (`by_accent`/`by_speaker`)
  records now carry both `accent_cosine` (centered by default) and `accent_cosine_genaid_raw`; a PREVIOUS
  raw-GenAID per-group `accent_cosine` value is preserved under `accent_cosine_genaid_raw` before being
  overwritten, unless already present (idempotent against re-runs). `results["accent_center_vector"]` records
  the vector path actually used (`null` under `--no_center`); the per-utt provenance field
  `accent_cosine_per_utt_source` now records the centered model tag.
- Added `eval/run_rescore_accent_centered.sbatch` -- VCTK only (the only set with a per-accent breakdown; the
  other three carry `accent_cosine` as an aggregate side metric only). Re-runs
  `articulatory-tts/score_side_metric.py --metric accent --center_vector ...` on the kept 16 kHz wav pairs,
  merges with `--keep_old_as genaid_raw`, updates provenance
  (`accent_side_metric_source`/`accent_side_metric_source_genaid_raw`), and redoes the per-accent/per-speaker
  breakdown with the updated `score_accent_per_utt.py`. Skip condition: `"centroid-centered" in
  metrics.accent_cosine.model`. Ran as job 10473692 -- see the 2026-09-17 entry above for results and the
  doc/JSON refresh.

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
