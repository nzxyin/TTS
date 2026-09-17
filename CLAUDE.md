# TTS (HSTEHSTEHSTE fork of coqui TTS) -- accented XTTS-v2

Fork of coqui TTS 0.22 carrying the "Scalable Controllable Accented TTS" work
(paper: https://arxiv.org/abs/2508.07426). Cluster/SLURM conventions live in
`~/.claude/CLAUDE.md`; this file is the current state of this repo. The
chronological record of changes is in `CHANGELOG.md`.

## Branches

- `dev` -- plain upstream coqui TTS; nothing accent-specific.
- `accent_release` -- the released inference code: `test.py` (bulk accented
  inference), accent-conditioned tokenizer (`TTS/tts/layers/xtts/tokenizer.py`,
  12 CommonVoice accents mapped onto XTTS language tokens) and `Xtts.synthesize(...,
  accents=)`; README links the three checkpoints
  (`HSTE/XTTS-v2-accent-finetune-{filtered,unfiltered,unlabeled}_full`).
- `accent_finetune` -- the fine-tuning recipe (`recipes/ljspeech/xtts_v2/`,
  `TTS/tts/layers/xtts/trainer/`); open PR #1 (2025-04-17). GitHub issues are
  disabled on this fork.
- `accent_eval` -- evaluation work (2026-09-07): everything under `eval/`, plus
  `requirements.txt` restored to upstream's pip-style list (the conda export
  `accent_release` shipped under that name is kept as `requirements.conda-export.txt`).

## Current state (2026-09-17)

- **Filtered checkpoint evaluated on the articulatory-tts test sets** (LJSpeech
  test, LibriTTS-R test-clean/test-other, VCTK held-out speakers; ESD dropped
  by user direction) with the articulatory-tts metric stack (Whisper-large-v3
  corpus WER, UTMOSv2, DNSMOS, ECAPA speaker cosine, GenAID accent
  cosine). Full tables and reading notes: `eval/README.md`; JSONs:
  `eval/results/`. Headline: XTTS matches ground-truth intelligibility within
  ~1 point of normalized WER on every set (VCTK 1.4%, LJSpeech 2.4%,
  test-clean 2.6%, test-other 3.0%), speaker cosine 0.62-0.68, UTMOSv2
  2.98-3.16. Accent cosine on VCTK is **0.625 +/- 0.013 (95% CI), centered
  GenAID (job 10473692, 2026-09-16 rescore)** -- raw GenAID 0.930 and
  CommonAccent 0.663 are kept in the JSON as superseded context
  (`accent_cosine_genaid_raw` / `accent_cosine_commonaccent`). LJSpeech /
  test-clean / test-other were out of scope for that rescore and still
  report raw GenAID: 0.95 / 0.96 / 0.91. Under the reference's
  punctuation-sensitive WER it reads 3.8% / 10.2% / 10.7% / 13.0%.
- **Accent metric switched to GenAID (2026-09-14, all four sets rescored on
  the kept wav pairs, job 10441102).** GenAID (https://github.com/jzmzhong/GenAID,
  speaker-adversarial XLSR-53 accent ID, 64-dim embedding) replaced
  CommonAccent in the reference repo's `score_side_metric.py`; the
  CommonAccent values (LJSpeech 0.717, test-clean 0.785, test-other 0.639,
  VCTK 0.663) are kept in every JSON as `accent_cosine_commonaccent` and are
  not comparable with GenAID's. GenAID cosines sit in a compressed 0.85-0.99
  band across every system, so read differences rather than absolute values.
- **Accent metric re-centered (2026-09-16, branch `eval/accent-centering`; rescored 2026-09-16/17, job
  10473692, VCTK-only -- COMPLETE and verified).** The articulatory-tts accent-metric diagnostic (that
  repo's `CLAUDE.md` "Accent-metric diagnostic" section) found GenAID's compressed 0.85-0.99 band is a
  positive-orthant display artifact, removable by subtracting a fixed centering vector without changing
  effect sizes -- and decided (2026-09-16) that the published `accent_cosine` should be this CENTERED value
  going forward (vector = mean of the six speaker-balanced VCTK-training-speaker accent centroids). Tracked
  in `nzxyin/articulatory-tts` issue #51 (this fork has issues disabled). `eval/score_accent_per_utt.py`
  gained `--center_vector`/`--no_center` (`genaid_accent.DEFAULT_CENTER_VECTOR` by default) and now writes
  the raw GenAID cosine alongside as `accent_cosine_genaid_raw`; `eval/run_rescore_accent_centered.sbatch`
  did the VCTK rescore on the kept wav pairs as job 10473692, confirmed done
  (`metrics.accent_cosine.model` contains `"centroid-centered"`, mean 0.6253). Corpus: **0.625 +/- 0.013
  (95% CI)**; raw GenAID 0.930 and CommonAccent 0.663 kept in the JSON as superseded context. This file and
  `eval/README.md` now report the centered value everywhere the VCTK accent cosine is shown; LJSpeech /
  test-clean / test-other were out of scope for this rescore (no per-accent breakdown there) and still
  report raw GenAID (their checked-in JSONs under `eval/results/` were refreshed from `/data` on
  2026-09-17 -- they had still carried the CommonAccent-era `accent_cosine` -- and now hold the 2026-09-14
  raw GenAID value 0.950 / 0.963 / 0.906 with `accent_cosine_commonaccent` alongside). Provenance caveat:
  the rescore ran with `ART_REPO=/home/xoy/wt-center` (the articulatory-tts PR #54 worktree, since removed),
  so `eval_vctk.json` records `center_vector` / `accent_center_vector` as
  `/home/xoy/wt-center/accent_metric_diag/results/genaid_emb_centroid_mean.npy`; the file is byte-identical
  to `accent_metric_diag/results/genaid_emb_centroid_mean.npy` on articulatory-tts main (sha256
  ce9ca6bc017b...379ef), the model tag's `center_vector=genaid_emb_centroid_mean.npy` is the durable
  identifier (nzxyin/articulatory-tts issue #57).
- **Per-accent accent similarity on VCTK (centered GenAID, `eval/score_accent_per_utt.py`, job 10473692):**
  Canadian 0.813, American 0.784, English 0.746, Scottish 0.711, Northern
  Irish 0.425, Irish 0.247 (raw GenAID, superseded: 0.962 / 0.950 / 0.939 /
  0.928 / 0.919 / 0.879 respectively). Same ranking as raw, but centering
  widens the gap sharply: American/Canadian/English/Scottish now cluster at
  0.71-0.81 while both Irish accents separate out at 0.247 / 0.425 -- a
  split the raw 0.85-0.96 band hid (CommonAccent had a North-American ~0.77
  vs British/Irish 0.58-0.65 split). GenAID's own labels (unaffected by
  centering) recognise the American, English and Scottish ground truth
  (95/76/44%) but not the Canadian or Northern Irish speaker (3/5%); where it
  does, XTTS output is labelled with the target class 84% (American), 75%
  (English) and 69% (Scottish -- more often than the real Scottish
  recordings) of the time. Table and caveats in `eval/README.md`.
- **Protocol** (see `eval/README.md`): self prompt (target utterance as the
  speaker reference, mirroring the articulatory model's use of the target's
  own speaker embedding); explicit accent tag always passed (`US` for
  LJSpeech/LibriTTS, VCTK accents mapped, NorthernIrish->Ireland); XTTS
  sentence splitting on; inference settings from the checkpoint's own
  `config.json`; per-utterance seeds.
- Inference environment: Python 3.10 venv on `/data/user_data/xoy/venvs/xtts-accent`
  (symlinked as `.venv`), pins in `eval/requirements-xtts-accent.txt`.
  Checkpoint at `/data/user_data/xoy/xtts_accent_checkpoints/`. Outputs at
  `/data/user_data/xoy/xtts_accent_eval/<ckpt>/<prompt_mode>/`.

## Standing findings

- `Xtts.synthesize()` overwrites caller-supplied `gpt_cond_len` /
  `gpt_cond_chunk_len` / `max_ref_len` / `sound_norm_refs` with the config
  values (second `settings.update`), so `test.py`'s `gpt_cond_len=3` never
  applied; the release recipe effectively used the config's conditioning
  length. Left unchanged on purpose; see `eval/README.md` "Known issues".
- `preempt` contains 18 Blackwell (`RTX_PRO_6000`) nodes on which torch 2.6
  (cu124) cannot run; `eval/run_synth.sbatch` excludes them.
- Unfiltered / unlabeled checkpoints and the cross-prompt protocol are not
  evaluated yet; the pipeline takes them as arguments
  (`run_synth.sbatch <dataset> <shard> <n> [prompt_mode] [ckpt]`).
