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
  corpus WER, UTMOSv2, DNSMOS, ECAPA speaker cosine, GenAID accent
  cosine). Full tables and reading notes: `eval/README.md`; JSONs:
  `eval/results/`. Headline: XTTS matches ground-truth intelligibility within
  ~1 point of normalized WER on every set (VCTK 1.4%, LJSpeech 2.4%,
  test-clean 2.6%, test-other 3.0%), speaker cosine 0.62-0.68, UTMOSv2
  2.98-3.16, accent cosine 0.93 on VCTK (GenAID; 0.95 / 0.96 / 0.91 on
  LJSpeech / test-clean / test-other). Under the reference's
  punctuation-sensitive WER it reads 3.8% / 10.2% / 10.7% / 13.0%.
- **Accent metric switched to GenAID (2026-09-14, all four sets rescored on
  the kept wav pairs, job 10441102).** GenAID (https://github.com/jzmzhong/GenAID,
  speaker-adversarial XLSR-53 accent ID, 64-dim embedding) replaced
  CommonAccent in the reference repo's `score_side_metric.py`; the
  CommonAccent values (LJSpeech 0.717, test-clean 0.785, test-other 0.639,
  VCTK 0.663) are kept in every JSON as `accent_cosine_commonaccent` and are
  not comparable with GenAID's. GenAID cosines sit in a compressed 0.85-0.99
  band across every system, so read differences rather than absolute values.
- **Per-accent accent similarity on VCTK (GenAID, `eval/score_accent_per_utt.py`):**
  American 0.950, Canadian 0.962, English 0.939, Scottish 0.928, Northern
  Irish 0.919, Irish 0.879 -- the North-American speakers still lead, Irish is
  now clearly last (CommonAccent had a North-American ~0.77 vs British/Irish
  0.58-0.65 split). GenAID's own labels recognise the American, English and
  Scottish ground truth (95/76/44%) but not the Canadian or Northern Irish
  speaker (3/5%); where it does, XTTS output is labelled with the target class
  84% (American), 75% (English) and 69% (Scottish -- more often than the real
  Scottish recordings) of the time. Table and caveats in `eval/README.md`.
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
