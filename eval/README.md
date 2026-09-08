# Evaluating the accented XTTS-v2 checkpoints on the articulatory-tts test sets

Everything in this directory evaluates one of the released accent-conditioned
XTTS-v2 checkpoints (`HSTE/XTTS-v2-accent-finetune-{filtered,unfiltered,unlabeled}_full`,
see the top-level README on the `accent_release` branch) on the same held-out
test splits, with the same metric implementations, as the
`/home/xoy/articulatory-tts` repo's `eval_full_testset.py`, so the two models'
numbers can be read side by side.

| test set | split file | utterances | speaker prompt (`--prompt_mode self`) | accent tag |
|---|---|---|---|---|
| `ljspeech` | `LJSpeech-1.1/preprocessed/test.json` | 150 | the utterance's own recording | `US` |
| `libritts_test_clean` | `LibriTTS_R/test-clean.json` | 4830 (39 spk) | same | `US` |
| `libritts_test_other` | `LibriTTS_R/test-other.json` | 5106 (33 spk) | same | `US` |
| `vctk` | `vctk_globe_accent_splits/vctk_only/test.tsv` | 2596 (6 held-out spk, one per accent) | same | VCTK accent mapped to the model's 12 labels |

ESD / emotion similarity are deliberately not part of this matrix (accented
model, emotion out of scope -- explicit user direction 2026-09-07).

Metrics: corpus WER with Whisper large-v3 in two flavours -- `wer` (lowercased,
punctuation kept; the convention every articulatory-tts JSON used until
2026-09-08) and `wer_whisper_normalized` (both sides through Whisper's
English text normalizer, pairs with an empty normalized reference skipped;
this is what articulatory-tts's primary `wer` computes since its GH #32 fix,
commit b040aa1, and what the sibling CosyVoice3 / EmoSphere++ evals report,
so **it is the headline intelligibility number**) -- plus UTMOSv2, DNSMOS
(p808/sig/bak/ovr), ECAPA-TDNN speaker cosine (prediction vs. ground truth)
and CommonAccent accent cosine (prediction vs. ground truth; scored by the
reference repo's own `score_side_metric.py`).

## Files

- `requirements-xtts-accent.txt` -- pinned Python 3.10 inference environment
  (pip-installable rendering of the conda export that `accent_release` shipped
  as `requirements.txt`; that export now lives at `requirements.conda-export.txt`
  and `requirements.txt` is upstream's pip-style list again, because setuptools
  rejects the conda format at `pip install -e .` time).
- `build_env.sbatch` -- builds `/data/user_data/xoy/venvs/xtts-accent` and
  downloads the filtered checkpoint to
  `/data/user_data/xoy/xtts_accent_checkpoints/XTTS-v2-accent-finetune-filtered_full/`.
- `synthesize_testset.py` -- test-set definitions, prompt selection, accent
  mapping, sharded + resumable synthesis; writes `wavs/<uid>.wav` (24 kHz) and
  `manifest.shard<i>of<n>.jsonl`.
- `score_synthesized.py` -- the metric port (run in the reference eval venv
  `/data/user_data/xoy/venvs/eval-articulatory-tts`); writes
  `eval_<dataset>.json` (+ `_per_utt.json`, + 16 kHz wav pairs for the accent
  side metric).
- `run_synth.sbatch <dataset> <shard> <nshards> [prompt_mode] [ckpt]`,
  `run_score.sbatch <dataset> [prompt_mode] [ckpt]` -- the SLURM wrappers
  (partition `preempt`, `--requeue`; both stages resume).
- `smoke_test.sbatch` -- 3 utterances per test set through the whole chain.

Outputs land under `/data/user_data/xoy/xtts_accent_eval/<ckpt>/<prompt_mode>/`.

## Typical run

```bash
sbatch eval/build_env.sbatch                       # once
sbatch eval/smoke_test.sbatch                      # sanity check
sbatch eval/run_synth.sbatch ljspeech 0 1
for i in 0 1 2 3; do sbatch eval/run_synth.sbatch libritts_test_clean $i 4; done
for i in 0 1 2 3; do sbatch eval/run_synth.sbatch libritts_test_other $i 4; done
for i in 0 1;     do sbatch eval/run_synth.sbatch vctk $i 2; done
# after all shards report SYNTH_DONE:
for ds in ljspeech libritts_test_clean libritts_test_other vctk; do sbatch eval/run_score.sbatch $ds; done
```

## Protocol notes

- **Self prompt.** The articulatory model drives its SPARC vocoder with the
  target utterance's own speaker embedding, so its `speaker_cosine` is
  "prediction vs. the recording that conditioned it". `--prompt_mode self`
  reproduces that setup for XTTS; `--prompt_mode cross` (another utterance of
  the same speaker) is the stricter zero-shot protocol and is available but
  is not what the reference numbers used.
- **Accent tag for un-labelled corpora.** Every fine-tuning sample carried an
  accent label, so the model has no "no accent" mode. LJSpeech and
  LibriTTS-R (LibriVox, predominantly American readers) get `US`, which is
  also the label the plain `[en]` language token maps to.
- **Text splitting.** XTTS truncates past ~250 characters; 449 LibriTTS-R
  test utterances are longer, so the standard XTTS sentence splitter is
  enabled (only affects those utterances -- shorter texts are a single chunk).
- **Sampling.** Inference settings are the checkpoint's own `config.json`
  (temperature 0.85, top-k 50, top-p 0.85, repetition penalty 2.0, 12 s GPT
  conditioning, 10 s speaker reference), as in the release `test.py`.
  The seed is fixed per utterance (`seed + crc32(uid)`), so a rerun
  reproduces the same audio.

## Cluster gotchas hit while running this

- **Blackwell nodes.** `preempt` includes 18 `RTX_PRO_6000` nodes; the
  xtts-accent venv's torch 2.6 (cu124) has no kernels for them and every CUDA
  op fails with "no kernel image is available for execution on the device".
  `run_synth.sbatch`/`smoke_test.sbatch` carry an `--exclude=` list for them
  (regenerate it with the command in the sbatch comment if the pool changes).
  The scoring venv (torch 2.12, cu129) runs fine on those nodes, so
  `run_score.sbatch` is not constrained.
- **`babel-l9-16`:** `onnxruntime` does not import there, so torchmetrics DNSMOS
  silently reports "not installed" (sibling session's finding, 2026-09-08);
  `run_score.sbatch` excludes that node. If a DNSMOS column ever comes back
  empty, check which node the job ran on first.
- **Preemption.** Synthesis shards get preempted and requeued regularly on
  `preempt`; that's expected -- outputs are written atomically and skipped on
  resume. `synthesize_testset.py` aborts (exit 3) on a CUDA error or 10
  consecutive failures rather than logging thousands of 'failed' records.

## Known issues found along the way (GitHub issues are disabled on this fork, so they live here)

- **`Xtts.synthesize()` ignores cloning kwargs** (`TTS/tts/models/xtts.py`, the
  second `settings.update(...)`): `gpt_cond_len`, `gpt_cond_chunk_len`,
  `max_ref_len` and `sound_norm_refs` passed by the caller are overwritten
  with the config values right before `full_inference()` is called. The
  release `test.py` passes `gpt_cond_len=3` and therefore never actually ran
  with 3 s conditioning -- it used whatever `config.json` said (12 s for the
  fine-tuned checkpoints' own config, 30 s for the base XTTS-v2 config the
  README's example points at). Inherited from upstream coqui TTS. Left as is
  here so the release code's effective behaviour is unchanged; the eval
  scripts rely on `config.json` values only. Fix if wanted: apply the config
  defaults first and the caller's kwargs second.
- **`VoiceBpeTokenizer.encode()` on the accent branches always checks the
  250-character limit against `'en'`** regardless of the language argument
  (harmless for this English-only evaluation).
- **`requirements.txt` on `accent_release` was a conda export**, which
  `pip install -e .` cannot parse; this branch restores upstream's pip-style
  list and keeps the export as `requirements.conda-export.txt`.
- **Editable install builds the `monotonic_align` Cython extension under PEP 517
  build isolation**, i.e. against whatever numpy the isolated build env picks,
  not the pinned runtime numpy 1.22. XTTS never imports that extension, so it
  is inert here; pass `--no-build-isolation` if a VITS/GlowTTS model is ever
  needed from this venv.

## Results -- filtered checkpoint, self prompt (2026-09-07)

Summary JSONs: `eval/results/XTTS-v2-accent-finetune-filtered_full/self/eval_<dataset>.json`
(per-utterance JSONs and audio stay on `/data`). Every split utterance was
synthesized and scored (150 / 4830 / 5106 / 2596; zero failures).

Reference rows come from `/data/user_data/xoy/articulatory-tts/`: ground truth
and SPARC-resynthesis ceilings from `sparc_resynth_baselines/`, the best
articulatory pretrain (`curriculum_softdtw_largedim_25k`, best-step=12900) and
the best VCTK fine-tune (`vctk_only_finetune_softdtw_replay_25k`, step=3500,
known-rate pace) from their `eval_*.json`. WER columns: `WER` is the raw
convention (lowercased, punctuation kept); `WER-n` is the same hypotheses
after Whisper's English normalizer -- the headline number. For GT/resynth
`WER-n` was recomputed from the stored transcripts in
`sparc_resynth_baselines/{gt,resynth}_transcripts_*.json`. The articulatory
model rows were scored before that repo's 2026-09-08 normalizer fix, so they
only have raw `WER`; its `werfix` re-scores (running at the time of writing)
will provide `WER-n` for them.

| LJSpeech (n=150) | WER | WER-n | UTMOSv2 | DNSMOS ovr | p808 | sig | bak | spk cos | accent cos |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 6.97 | 1.60 | 3.951 | 3.292 | 4.026 | 3.638 | 3.959 | -- | -- |
| SPARC resynthesis | 6.78 | 1.83 | 3.196 | 3.357 | 3.909 | 3.614 | 4.121 | 0.611 | -- |
| articulatory, softdtw large_dim 25k | 6.55 | -- | 3.242 | 3.297 | 3.837 | 3.591 | 4.048 | 0.440 | -- |
| **XTTS-v2 accent (filtered)** | **10.22** | **2.44** | **3.158** | **3.402** | **4.047** | **3.661** | **4.136** | **0.682** | **0.717** |

| LibriTTS-R test-clean (n=4830) | WER | WER-n | UTMOSv2 | DNSMOS ovr | p808 | sig | bak | spk cos | accent cos |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 10.60 | 2.27 | 3.253 | 3.136 | 3.708 | 3.467 | 3.927 | -- | -- |
| SPARC resynthesis | 11.25 | 3.10 | 3.034 | 3.240 | 3.769 | 3.543 | 4.024 | 0.826 | -- |
| articulatory, softdtw large_dim 25k | 8.44 | -- | 3.124 | 3.219 | 3.769 | 3.527 | 4.005 | 0.689 | -- |
| **XTTS-v2 accent (filtered)** | **10.74** | **2.59** | **3.122** | **3.316** | **3.916** | **3.592** | **4.079** | **0.680** | **0.785** |

| LibriTTS-R test-other (n=5106) | WER | WER-n | UTMOSv2 | DNSMOS ovr | p808 | sig | bak | spk cos | accent cos |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 13.51 | 4.11 | 3.131 | 3.042 | 3.571 | 3.391 | 3.869 | -- | -- |
| SPARC resynthesis | 15.46 | 6.33 | 2.935 | 3.166 | 3.642 | 3.487 | 3.978 | 0.797 | -- |
| articulatory, softdtw large_dim 25k | 10.95 | -- | 3.092 | 3.165 | 3.677 | 3.488 | 3.971 | 0.606 | -- |
| **XTTS-v2 accent (filtered)** | **12.99** | **2.95** | **2.984** | **3.264** | **3.837** | **3.545** | **4.062** | **0.623** | **0.639** |

| VCTK held-out speakers (n=2596) | WER | WER-n | UTMOSv2 | DNSMOS ovr | p808 | sig | bak | spk cos | accent cos |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 3.70 | 1.21 | 3.593 | 3.202 | 3.640 | 3.517 | 4.003 | -- | -- |
| SPARC resynthesis | 5.77 | 2.73 | 3.145 | 3.189 | 3.545 | 3.492 | 4.031 | 0.646 | 0.783 |
| articulatory, softdtw large_dim 25k (zero-shot, known rate) | 3.42 | -- | 2.651 | 3.132 | 3.620 | 3.479 | 3.927 | 0.424 | 0.446 |
| articulatory, VCTK fine-tune softdtw replay 25k (known rate) | 3.75 | -- | 3.326 | 3.247 | 3.597 | 3.536 | 4.068 | 0.519 | 0.705 |
| **XTTS-v2 accent (filtered)** | **3.80** | **1.42** | **3.124** | **3.205** | **3.737** | **3.512** | **4.039** | **0.643** | **0.663** |

XTTS per VCTK accent (one held-out speaker per accent; from `by_accent` in
`eval_vctk.json`; accent tag passed in parentheses). Accent cosine per
utterance comes from `eval/score_accent_per_utt.py` (same CommonAccent model
and `encode_batch` path as the reference side metric; its overall mean
reproduces the merged 0.663 exactly). The last three columns are the
CommonAccent classifier's 16-way top label: how often the synthesized
utterance is labelled with the target class, how often the *ground-truth*
recording is, and how often the two labels agree.

| accent (tag) | speaker | n | accent cos (ci95) | WER | UTMOSv2 | spk cos | pred labelled target | GT labelled target | pred = GT label |
|---|---|---|---|---|---|---|---|---|---|
| American (US) | p297 | 417 | 0.766 (0.009) | 3.46 | 3.192 | 0.603 | 0.940 | 0.971 | 0.928 |
| Canadian (Canada) | p317 | 423 | 0.771 (0.008) | 5.05 | 2.889 | 0.694 | 0.061 | 0.021 | 0.917 |
| English (England) | p270 | 462 | 0.584 (0.010) | 5.13 | 2.989 | 0.615 | 0.645 | 0.232 | 0.418 |
| Irish (Ireland) | p288 | 412 | 0.629 (0.011) | 2.24 | 3.300 | 0.640 | 0.000 | 0.036 | 0.308 |
| Northern Irish (Ireland) | p304 | 423 | 0.645 (0.010) | 1.92 | 3.496 | 0.615 | 0.000 | 0.175 | 0.326 |
| Scottish (Scotland) | p281 | 459 | 0.596 (0.010) | 4.69 | 2.913 | 0.685 | 0.107 | 0.037 | 0.505 |

Per-accent reading notes:

- Accent cosine splits cleanly into the two North-American speakers (~0.77)
  and the four British/Irish speakers (0.58-0.65); the SPARC-resynthesis
  ceiling for this metric is 0.783 and the articulatory VCTK fine-tune
  averages 0.705. With one speaker per accent, speaker and accent effects
  cannot be separated here.
- The label columns are an accent-classification view, not the metric. The
  classifier calls the real VCTK Irish, Scottish and Canadian recordings
  by their own class only 2-4% of the time (and Northern Irish 18%), so a low
  "pred labelled target" for those accents says more about CommonAccent on
  VCTK than about the accent tag. Where the classifier does work on the
  ground truth (American 97%, English 23%), the synthesized speech is
  labelled the same way at least as often (94%, 65%), and Canadian output
  is labelled `us` -- like the Canadian ground truth -- 92% of the time.
- Per-speaker accent cosine is also in `eval_vctk.json` (`by_speaker`) and
  the per-utterance values with both labels in
  `/data/user_data/xoy/xtts_accent_eval/.../self/eval_vctk_accent_per_utt.json`.

Reading notes:

- **Raw `WER` is punctuation-sensitive.** Whisper writes far less punctuation
  for XTTS audio than for the human recordings (XTTS raw `WER` is above ground
  truth on LJSpeech/test-clean while its normalized `WER-n` is within ~1 point
  of ground truth everywhere). Compare raw `WER` across systems only with that
  in mind; `WER-n` is the intelligibility number and the one all four sibling
  evaluations (articulatory-tts post-b040aa1, CosyVoice3, EmoSphere++, this)
  now compute identically: same Whisper checkpoint/dtype/greedy decoding, same
  normalizer, corpus-level jiwer, empty-normalized-reference pairs skipped
  (one utterance each in test-clean and test-other: `[you thought i had
  forgotten]` and `[see footnote]` -- the normalizer strips bracketed text as
  an annotation, leaving an empty reference).
- **Speaker similarity is not the same protocol as a zero-shot cloning
  benchmark.** Self prompt = the model heard the target recording; this
  mirrors the articulatory setup (its vocoder uses the target's own speaker
  embedding) but inflates absolute values relative to cross-utterance
  prompting. The SPARC-resynthesis row is the articulatory codec's own
  ceiling, so the articulatory model cannot exceed it; XTTS has no such cap.
- **Articulatory VCTK rows are known-rate** (inference told the target's
  speaking rate), the project's standard since 2026-09-07; XTTS gets no
  rate information beyond what is in the prompt audio.
- **Accent cosine is against the ground-truth recording**, so for LJSpeech/
  LibriTTS it mostly measures how much the `US` tag plus the prompt preserve
  the speaker's own accent character; on VCTK it is the in-domain number.
  The reference only reports it for VCTK.
