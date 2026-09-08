"""Synthesize the articulatory-tts held-out test splits with an accented
XTTS-v2 checkpoint (HSTE/XTTS-v2-accent-finetune-{filtered,unfiltered,
unlabeled}_full), one 24 kHz wav per test utterance plus a JSONL manifest
that eval/score_synthesized.py consumes.

Test sets mirror /home/xoy/articulatory-tts/eval_full_testset.py's
DATASET_SPECS exactly (same split files, same raw-audio/transcript
sources), minus ESD (dropped by explicit user direction, 2026-09-07: this
is an accented model, emotion is out of scope):

  ljspeech             LJSpeech-1.1/preprocessed/test.json            (150 utts)
  libritts_test_clean  LibriTTS_R/test-clean.json                     (4830 utts)
  libritts_test_other  LibriTTS_R/test-other.json                     (5106 utts)
  vctk                 vctk_globe_accent_splits/vctk_only/test.tsv    (2596 utts, 6 held-out speakers, one per accent)

Speaker prompt (--prompt_mode):
  self   (default) the target utterance's own ground-truth recording. This
         matches the articulatory-tts protocol, whose SPARC vocoder is
         driven by the target utterance's own speaker embedding
         (spk_emb/{uid}.npy) -- so speaker_cosine there is also "pred vs
         the very utterance that conditioned it". Use this for numbers
         meant to sit next to the articulatory-tts eval JSONs.
  cross  a different utterance from the same speaker within the same test
         split (deterministic: the next uid in that speaker's sorted list,
         cyclic). The stricter zero-shot-TTS protocol; not what the
         reference implementation does, offered for completeness.

Accent tag: the fine-tuned model conditions on one of 12 CommonVoice accent
labels via the tokenizer (see TTS/tts/layers/xtts/tokenizer.py's
accents_to_langcode). VCTK's per-speaker accent is mapped onto that set
(English->England, Scottish->Scotland, Irish/NorthernIrish->Ireland,
American->US, Canadian->Canada). LJSpeech and LibriTTS-R carry no accent
labels; both are read by (predominantly) American speakers and are tagged
"US". "US" is also what the model's plain "[en]" language token maps to,
i.e. the closest thing this model has to "no accent control" -- an explicit
tag is still passed so the prompt is formatted the way training data was
("[en] text", with the space), not the un-accented "[en]text" path.

Inference settings come from the checkpoint's own config.json through
Xtts.synthesize (temperature 0.85, top_k 50, top_p 0.85, repetition_penalty
2.0, gpt_cond_len 12 s, gpt_cond_chunk_len 4 s, max_ref_len 10 s). This is
also what the released test.py on the accent_release branch effectively
runs: test.py passes gpt_cond_len=3, but Xtts.synthesize() re-applies the
config's gpt_cond_len/gpt_cond_chunk_len/max_ref_len/sound_norm_refs AFTER
merging kwargs (TTS/tts/models/xtts.py, second settings.update), so that
override is silently discarded -- an upstream coqui bug this fork inherits;
see eval/README.md "Known issues". This script therefore deliberately does not
try to pass those four knobs as kwargs; change config.json if a different
conditioning length is wanted. enable_text_splitting is on by
default: XTTS's tokenizer warns (and the GPT truncates) beyond 250
characters of English, and 449 LibriTTS-R test utterances exceed that;
splitting only kicks in for those (shorter texts pass through as a single
chunk, identical to the unsplit path).

Restart-safe: an existing non-empty output wav is skipped, so a preempted
job resumes where it stopped. Sharding (--shard_index/--num_shards) splits
the utterance list round-robin after a deterministic sort.

Usage (see eval/run_synth.sbatch):
  python eval/synthesize_testset.py --dataset vctk \
      --checkpoint_dir /data/user_data/xoy/xtts_accent_checkpoints/XTTS-v2-accent-finetune-filtered_full \
      --out_dir /data/user_data/xoy/xtts_accent_eval/filtered_full/vctk
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
import zlib

# numpy/torch are imported inside main(): this module's DATASETS/load_items/assign_prompts
# are also used by eval/run_score.sbatch's completeness gate (system python3, no numpy) and by
# eval/score_synthesized.py (a different venv), so the top level must stay stdlib-only.

# --- test-set definitions (paths identical to articulatory-tts's eval_full_testset.py) ----------
DATASETS = {
    "ljspeech": {
        "split_path": "/data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json",
        "raw_wav_dir": "/data/user_data/xoy/LJSpeech-1.1/wavs",
        "metadata_csv": "/data/user_data/xoy/LJSpeech-1.1/metadata.csv",
        "default_accent": "US",
    },
    "libritts_test_clean": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-clean.json",
        "raw_wav_dir": "/data/user_data/xoy/LibriTTS_R/test-clean",
        "default_accent": "US",
    },
    "libritts_test_other": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-other.json",
        # no local copy of test-other exists; same group_data source the reference eval reads
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/LibriTTS-R/LibriTTS_R/test-other",
        "default_accent": "US",
    },
    "vctk": {
        "split_path": "/data/user_data/xoy/vctk_globe_accent_splits/vctk_only/test.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48",
        "txt_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/txt",
        # TsvCorpusDataset drops split rows with no phn_ids_2 entry (upstream
        # phonemization gap) -- mirror that so the utterance set is identical
        # to what eval_full_testset.py scores. Currently a no-op (2596/2596
        # present), kept so the two stay in lockstep if the split changes.
        "phn_ids_dir": "/data/user_data/xoy/VCTK/VCTK-Corpus/preprocessed/phn_ids_2",
    },
}

# VCTK speaker-info accent group -> the fine-tuned model's 12-way accent vocabulary
# (TTS/tts/layers/xtts/tokenizer.py, VoiceBpeTokenizer.accents_to_langcode).
VCTK_TO_XTTS_ACCENT = {
    "English": "England",
    "Scottish": "Scotland",
    "Irish": "Ireland",
    # No Northern-Irish class exists in the model; Irish English is the
    # nearest available label (closer than England/Scotland). Flagged in the
    # results JSON via the per-utterance accent_label so it can be broken out.
    "NorthernIrish": "Ireland",
    "American": "US",
    "Canadian": "Canada",
    "Australian": "Australia",
    "Indian": "India",
    "Welsh": "Wales",
    "SouthAfrican": "Southern Africa",
    # NewZealand has no counterpart at all; not in the vctk_only test split anyway.
}

XTTS_ACCENTS = ["US", "England", "India", "Germany", "Canada", "Australia",
                "Southern Africa", "Philippines", "Scotland", "Ireland", "Malaysia", "Wales"]


SPLIT_STATS = {"n_split_total": None}  # filled by load_items()


def read_text(path):
    """VCTK/LibriTTS transcripts are overwhelmingly utf-8; the same
    utf-8 -> latin-1 fallback eval_full_testset.py uses for VCTK."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.read().strip()


def load_items(dataset):
    spec = DATASETS[dataset]
    items = []
    if dataset == "ljspeech":
        ids = json.load(open(spec["split_path"]))
        id_set = set(ids)
        texts = {}
        with open(spec["metadata_csv"], encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                if row and row[0] in id_set:
                    # column 3 = normalized transcript (numbers spelled out), same column the reference WER uses
                    texts[row[0]] = row[2] if len(row) > 2 else row[1]
        for uid in ids:
            items.append({
                "uid": uid, "speaker": "LJ", "text": texts.get(uid),
                "gt_wav": os.path.join(spec["raw_wav_dir"], f"{uid}.wav"),
                "accent_label": "unknown", "xtts_accent": spec["default_accent"],
            })
    elif dataset.startswith("libritts"):
        ids = json.load(open(spec["split_path"]))
        for uid in ids:
            spk, chap = uid.split("_")[0], uid.split("_")[1]
            d = os.path.join(spec["raw_wav_dir"], spk, chap)
            txt = os.path.join(d, f"{uid}.normalized.txt")
            items.append({
                "uid": uid, "speaker": spk,
                "text": read_text(txt) if os.path.exists(txt) else None,
                "gt_wav": os.path.join(d, f"{uid}.wav"),
                "accent_label": "unknown", "xtts_accent": spec["default_accent"],
            })
    elif dataset == "vctk":
        with open(spec["split_path"], newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        available = {fn[: -len(".phn.npy")] for fn in os.listdir(spec["phn_ids_dir"]) if fn.endswith(".phn.npy")}
        dropped = [r["stem"] for r in rows if r["stem"] not in available]
        if dropped:
            print(f"WARNING: {len(dropped)}/{len(rows)} VCTK test rows have no phn_ids_2 entry and are "
                  f"excluded (same filter as articulatory-tts's TsvCorpusDataset), e.g. {dropped[:3]}")
        for r in rows:
            if r["stem"] in dropped:
                continue
            spk = r["speaker"]
            txt = os.path.join(spec["txt_dir"], spk, f"{r['stem']}.txt")
            items.append({
                "uid": r["stem"], "speaker": spk,
                "text": read_text(txt) if os.path.exists(txt) else None,
                "gt_wav": os.path.join(spec["raw_wav_dir"], spk, f"{r['stem']}.wav"),
                "accent_label": r["accent"], "xtts_accent": VCTK_TO_XTTS_ACCENT[r["accent"]],
            })
    else:
        raise ValueError(dataset)

    n_all = len(items)
    # XTTS needs the transcript (it is the text to speak) and, in self-prompt mode, the GT recording
    # (it is the speaker prompt), so rows lacking either cannot be synthesized at all. The reference
    # still scores audio metrics for a transcript-less row (its decoder needs no text) -- a real,
    # documented semantic difference that is a no-op on these four splits (0 rows lack either).
    items = [it for it in items if it["text"] and os.path.exists(it["gt_wav"])]
    SPLIT_STATS["n_split_total"] = n_all
    if len(items) != n_all:
        print(f"WARNING: {n_all - len(items)}/{n_all} utterances dropped for missing transcript or ground-truth wav")
    items.sort(key=lambda it: it["uid"])  # deterministic order before sharding / cross-prompt pairing
    return items


def assign_prompts(items, mode):
    if mode == "self":
        for it in items:
            it["prompt_wav"] = it["gt_wav"]
        return
    by_spk = {}
    for it in items:
        by_spk.setdefault(it["speaker"], []).append(it)
    for spk, group in by_spk.items():
        if len(group) == 1:
            print(f"WARNING: speaker {spk} has a single test utterance; cross prompt falls back to self")
        for i, it in enumerate(group):
            it["prompt_wav"] = group[(i + 1) % len(group)]["gt_wav"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--checkpoint_dir", required=True,
                    help="dir with model.pth, dvae.pth, mel_stats.pth, vocab.json (and config.json unless --config is given)")
    ap.add_argument("--config", default=None, help="XTTS config.json; defaults to <checkpoint_dir>/config.json")
    ap.add_argument("--out_dir", required=True, help="wavs go to <out_dir>/wavs/<uid>.wav; manifest JSONL alongside")
    ap.add_argument("--prompt_mode", choices=["self", "cross"], default="self")
    ap.add_argument("--accent_override", default=None, choices=XTTS_ACCENTS + ["none"],
                    help="force one accent tag for every utterance ('none' = pass accents=None, the un-accented "
                         "'[en]text' tokenizer path). Default: per-dataset mapping described in the module docstring.")
    ap.add_argument("--no_text_splitting", action="store_true",
                    help="disable XTTS's >=250-char sentence splitting (long LibriTTS utterances will be truncated)")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N utterances of this shard (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="per-utterance seed = seed + crc32(uid), for reproducible sampling")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    items = load_items(args.dataset)
    assign_prompts(items, args.prompt_mode)
    shard = items[args.shard_index::args.num_shards]
    if args.limit:
        shard = shard[: args.limit]
    print(f"{args.dataset}: {len(items)} utterances total, {len(shard)} in shard "
          f"{args.shard_index}/{args.num_shards}, prompt_mode={args.prompt_mode}")

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, f"manifest.shard{args.shard_index}of{args.num_shards}.jsonl")
    # split-level bookkeeping for the scorer: the reference's n_utterances is the raw split-file
    # count, so record it (and how many rows could not be synthesized) once per output dir.
    info_path = os.path.join(args.out_dir, "split_info.json")
    if not os.path.exists(info_path):
        tmp = info_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"dataset": args.dataset, "split_path": DATASETS[args.dataset]["split_path"],
                       "n_split_total": SPLIT_STATS["n_split_total"], "n_synthesizable": len(items),
                       "n_dropped_missing_text_or_gt_wav": SPLIT_STATS["n_split_total"] - len(items),
                       "prompt_mode": args.prompt_mode}, f, indent=2)
        os.replace(tmp, info_path)

    # everything below needs the model; import late so --help stays cheap
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config_path = args.config or os.path.join(args.checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(config_path)
    # coqpit silently turns a type-mismatched primitive into None and ignores unknown keys, so a
    # config.json from a different coqpit/TTS version could leave an inference knob unset.
    for k in ("temperature", "length_penalty", "repetition_penalty", "top_k", "top_p",
              "gpt_cond_len", "gpt_cond_chunk_len", "max_ref_len"):
        v = getattr(config, k)
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"config.{k} did not load as a number: {v!r}"
    assert isinstance(config.sound_norm_refs, bool), f"config.sound_norm_refs did not load as bool: {config.sound_norm_refs!r}"
    assert isinstance(config.model_args.output_sample_rate, int), "config.model_args.output_sample_rate did not load"
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=args.checkpoint_dir, eval=True)
    model.cuda()
    print(f"loaded {args.checkpoint_dir} with config {config_path}; inference settings: "
          f"temperature={config.temperature} top_k={config.top_k} top_p={config.top_p} "
          f"repetition_penalty={config.repetition_penalty} length_penalty={config.length_penalty} "
          f"gpt_cond_len={config.gpt_cond_len}s gpt_cond_chunk_len={config.gpt_cond_chunk_len}s "
          f"max_ref_len={config.max_ref_len}s sound_norm_refs={config.sound_norm_refs}")
    out_sr = config.model_args.output_sample_rate  # 24000

    n_done = n_skipped = n_failed = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    t_start = time.time()
    gen_time = 0.0
    with open(manifest_path, "a") as mf:
        for k, it in enumerate(shard):
            pred_path = os.path.join(wav_dir, f"{it['uid']}.wav")
            record = {**it, "pred_wav": pred_path, "prompt_mode": args.prompt_mode}
            if not args.overwrite and os.path.exists(pred_path) and os.path.getsize(pred_path) > 1000:
                n_skipped += 1
                record["status"] = "existing"
                mf.write(json.dumps(record) + "\n")
                continue
            accent = it["xtts_accent"]
            if args.accent_override:
                accent = None if args.accent_override == "none" else args.accent_override
            record["xtts_accent"] = accent
            seed = (args.seed + zlib.crc32(it["uid"].encode())) % (2**31)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            t0 = time.time()
            try:
                out = model.synthesize(
                    it["text"], config,
                    speaker_wav=it["prompt_wav"],
                    language="en",
                    accents=accent,
                    enable_text_splitting=not args.no_text_splitting,
                )
                wav = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
                if wav.size == 0:
                    raise RuntimeError("empty waveform")
                tmp = pred_path + ".tmp.wav"
                sf.write(tmp, wav, out_sr)
                os.replace(tmp, pred_path)  # atomic: a preempted job never leaves a truncated final wav
                dt = time.time() - t0
                gen_time += dt
                n_done += 1
                record.update({"status": "ok", "seed": seed, "pred_seconds": round(len(wav) / out_sr, 3),
                               "gen_wall_seconds": round(dt, 2)})
            except Exception as e:
                n_failed += 1
                consecutive_failures += 1
                record.update({"status": "failed", "seed": seed, "error": f"{type(e).__name__}: {e}"})
                print(f"FAILED {it['uid']}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                # A per-utterance failure (odd text, unreadable prompt) is logged and skipped, but a
                # systemic one must abort loudly: a CUDA error (e.g. a GPU architecture this torch
                # build has no kernels for -- hit on a Blackwell node in `preempt`, job 10350518)
                # or a run of consecutive failures means nothing downstream will work either, and
                # continuing just fills the manifest with 1700 'failed' records in 12 seconds.
                if "CUDA" in str(e) or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    mf.write(json.dumps(record) + "\n")
                    mf.flush()
                    print(f"ABORTING: {consecutive_failures} consecutive failures / CUDA error -- "
                          f"see traceback above", file=sys.stderr)
                    sys.exit(3)
            else:
                consecutive_failures = 0
            mf.write(json.dumps(record) + "\n")
            mf.flush()
            if (k + 1) % 25 == 0 or k + 1 == len(shard):
                elapsed = time.time() - t_start
                rate = gen_time / max(n_done, 1)
                remaining = (len(shard) - k - 1) * rate
                print(f"[{k + 1}/{len(shard)}] done={n_done} skipped={n_skipped} failed={n_failed} "
                      f"elapsed={elapsed / 60:.1f}min avg_gen={rate:.2f}s/utt eta={remaining / 60:.1f}min", flush=True)

    print(f"SUMMARY dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
          f"synthesized={n_done} skipped_existing={n_skipped} failed={n_failed} "
          f"gen_time={gen_time / 60:.1f}min manifest={manifest_path}")
    if n_failed and n_done == 0 and n_skipped == 0:
        sys.exit(1)
    # completion marker: only written when the loop ran to the end of the shard (a preempted job
    # leaves a manifest but no marker). run_score.sbatch's gate checks the wavs themselves, this
    # is a cheap secondary signal for humans / other tooling.
    with open(manifest_path + ".done", "w") as f:
        json.dump({"shard": args.shard_index, "num_shards": args.num_shards, "n_shard": len(shard),
                   "synthesized": n_done, "skipped_existing": n_skipped, "failed": n_failed,
                   "limit": args.limit}, f)


if __name__ == "__main__":
    main()
