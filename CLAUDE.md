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

## Current state (2026-09-08)

- **Filtered checkpoint evaluated on the articulatory-tts test sets** (LJSpeech
  test, LibriTTS-R test-clean/test-other, VCTK held-out speakers; ESD dropped
  by user direction) with the articulatory-tts metric stack (Whisper-large-v3
  corpus WER, UTMOSv2, DNSMOS, ECAPA speaker cosine, CommonAccent accent
  cosine). Full tables and reading notes: `eval/README.md`; JSONs:
  `eval/results/`. Headline: XTTS matches ground-truth intelligibility within
  ~1 point of normalized WER on every set (VCTK 1.4%, LJSpeech 2.4%,
  test-clean 2.6%, test-other 3.0%), speaker cosine 0.62-0.68, UTMOSv2
  2.98-3.16, accent cosine 0.66 on VCTK. Under the reference's
  punctuation-sensitive WER it reads 3.8% / 10.2% / 10.7% / 13.0%.
- **Per-accent accent similarity on VCTK (2026-09-08, `eval/score_accent_per_utt.py`):**
  accent cosine is high for the two North-American test speakers (American
  0.77, Canadian 0.77) and lower for the British/Irish ones (English 0.58,
  Scottish 0.60, Irish 0.63, Northern Irish 0.64). The CommonAccent
  classifier labels XTTS's American-tagged output as `us` 94% of the time but
  almost never labels the Irish/Scottish-tagged output with the target class;
  it also rarely labels the *ground-truth* VCTK Irish/Scottish recordings
  correctly (4%/4%), so the label view mostly reflects that classifier's
  weakness on VCTK, not a verdict on the accent control. Table and caveats in
  `eval/README.md`.
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
