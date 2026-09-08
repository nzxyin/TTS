"""Score synthesized test-set audio (from eval/synthesize_testset.py) with
the SAME metric implementations articulatory-tts uses, so the numbers sit
next to that repo's eval_*.json files: this is a port of the score()
function in /home/xoy/articulatory-tts/eval_full_testset.py with the
SPARC-decode step replaced by "read the wav XTTS already wrote".

Metrics (identical model choices / preprocessing / aggregation to the reference):
  wer            corpus-level jiwer.wer over lowercased reference vs. Whisper
                 large-v3 (openai/whisper-large-v3, fp32, language=en,
                 task=transcribe) hypothesis of the 16 kHz-resampled prediction.
                 Lowercasing only, punctuation kept -- exactly what the
                 reference does, so keep it that way for comparability.
  wer_whisper_normalized   same hypotheses/references passed through
                 WhisperTokenizer.normalize (the standard Whisper English
                 normalizer: punctuation, casing, number words, British/
                 American spelling), corpus-level, pairs with an empty
                 normalized reference skipped. Since articulatory-tts's GH #32
                 fix (commit b040aa1, 2026-09-08) THIS is what that repo's
                 primary `wer` key computes, so compare this key against
                 articulatory-tts JSONs produced after that commit, and the
                 raw `wer` key against JSONs produced before it.
  utmosv2        UTMOSv2 (utmosv2.create_model(pretrained=True)) on 16 kHz prediction
  dnsmos_*       torchmetrics DNSMOS (p808 / sig / bak / ovr), non-personalized, 16 kHz prediction
  speaker_cosine ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) embedding cosine,
                 prediction vs. ground-truth recording of the same utterance
                 (>= 0.3 s each, same floor as the reference)
  accent_cosine  NOT computed here -- needs the eval-accent venv. Pass
                 --wav_pairs_dir to write {uid}_pred.wav/{uid}_gt.wav (16 kHz)
                 and run /home/xoy/articulatory-tts/score_side_metric.py
                 --metric accent on it, then merge_eval_results.py (see
                 eval/run_score.sbatch), exactly as the reference chain does.

Utterance universe: rebuilt from eval/synthesize_testset.py's own
load_items() (the split file + transcript/GT lookups), NOT from the shard
manifests -- the manifests only contribute per-utterance metadata (seed,
timing, status). An utterance is scored iff its prediction wav exists; every
synthesizable utterance without one is reported as a failure, so the
results are consistent no matter how many times shards were resumed,
resharded or overwritten.

Summary shape per metric is the reference's {mean, ci95, n} (ci95 = 1.96 *
std / sqrt(n); WER's ci95 is None because corpus WER is a single ratio).
Results JSON top-level keys mirror the reference (checkpoint, dataset,
split, n_utterances, n_predicted, metrics) -- n_utterances here is the
number of SYNTHESIZABLE split rows (transcript + GT wav present; equal to the
raw split-file count on all four test sets, which is also recorded as
n_split_total). Per-utterance records (--per_utt_out) keep the raw
reference/hypothesis text so WER can be re-aggregated over any subset later
without re-running Whisper; per-accent and per-speaker breakdowns are also
written ("by_accent", "by_speaker").

Run under the reference eval venv so library versions match the
articulatory-tts numbers: /data/user_data/xoy/venvs/eval-articulatory-tts/bin/python
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize_testset import DATASETS, SPLIT_STATS, assign_prompts, load_items  # noqa: E402


def summarize(values):
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": int(len(values))}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def write_json_atomic(obj, path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def load_manifest_metadata(patterns):
    """uid -> the most informative manifest record ('ok' beats 'existing' beats 'failed';
    among equals the later record wins). Metadata only -- see module docstring."""
    rank = {"ok": 3, "existing": 2, "failed": 1}
    records = {}
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                prev = records.get(r["uid"])
                if prev is None or rank.get(r.get("status"), 0) >= rank.get(prev.get("status"), 0):
                    records[r["uid"]] = r
    print(f"loaded metadata for {len(records)} utterances from {len(files)} manifest file(s)")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir", required=True, help="synthesize_testset.py's --out_dir (wavs/ + manifests live here)")
    ap.add_argument("--prompt_mode", choices=["self", "cross"], default="self",
                    help="must match what synthesize_testset.py was run with (recorded in the results JSON)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint identifier recorded in the results JSON")
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--per_utt_out", default=None)
    ap.add_argument("--wav_pairs_dir", default=None, help="write 16 kHz {uid}_pred.wav/{uid}_gt.wav pairs for score_side_metric.py")
    ap.add_argument("--skip_audio", action="store_true", help="skip UTMOSv2 + DNSMOS")
    ap.add_argument("--skip_wer", action="store_true")
    ap.add_argument("--skip_speaker", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import soundfile as sf
    import librosa
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.dataset)
    assign_prompts(items, args.prompt_mode)
    n_split_total = SPLIT_STATS["n_split_total"]
    wav_dir = os.path.join(args.out_dir, "wavs")
    for it in items:
        it["pred_wav"] = os.path.join(wav_dir, f"{it['uid']}.wav")
    meta = load_manifest_metadata([os.path.join(args.out_dir, "manifest.shard*of*.jsonl")])
    if args.limit:
        items = items[: args.limit]

    def has_wav(it):
        return os.path.exists(it["pred_wav"]) and os.path.getsize(it["pred_wav"]) > 1000

    ok = [it for it in items if has_wav(it)]
    missing = [it for it in items if not has_wav(it)]
    print(f"{args.dataset}: {n_split_total} split rows, {len(items)} synthesizable, "
          f"{len(ok)} with a prediction wav, {len(missing)} missing")
    recorded_modes = {meta[it["uid"]].get("prompt_mode") for it in ok if it["uid"] in meta}
    if recorded_modes and recorded_modes != {args.prompt_mode}:
        raise SystemExit(f"manifests record prompt_mode={recorded_modes} but --prompt_mode={args.prompt_mode}")

    # ---- models (same sources/settings as eval_full_testset.py) ---------------------------------
    dnsmos_fn = utmos_model = whisper_processor = whisper_model = ecapa_model = None
    if not args.skip_audio:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos_fn
        import utmosv2
        utmos_model = utmosv2.create_model(pretrained=True)
    if not args.skip_wer:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import jiwer
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float32
        ).to(device).eval()
        normalize = whisper_processor.tokenizer.normalize
    if not args.skip_speaker:
        from speechbrain.inference.speaker import EncoderClassifier
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.environ.get("HF_HOME", "/tmp"), "spkrec-ecapa-voxceleb"),
            run_opts={"device": str(device)},
        )
    if args.wav_pairs_dir:
        os.makedirs(args.wav_pairs_dir, exist_ok=True)

    def to16k(wav, sr):
        return wav if sr == 16000 else librosa.resample(wav, orig_sr=sr, target_sr=16000)

    def dnsmos_score(wav16):
        p808, sig, bak, ovr = dnsmos_fn(torch.from_numpy(wav16).float().to(device), 16000, personalized=False, num_threads=4)
        return {"p808": float(p808), "sig": float(sig), "bak": float(bak), "ovr": float(ovr)}

    def utmos_score(wav16):
        result = utmos_model.predict(data=wav16.astype(np.float32), sr=16000)
        return float(np.asarray(result).reshape(-1)[0])

    def whisper_transcribe(wav16):
        inputs = whisper_processor(wav16, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        ids_out = whisper_model.generate(inputs, language="en", task="transcribe")
        return whisper_processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()

    MIN_ECAPA_SEC = 0.3

    def ecapa_embed(wav16):
        emb = ecapa_model.encode_batch(torch.from_numpy(wav16).float().unsqueeze(0).to(device))
        return emb.squeeze().detach().cpu().numpy().reshape(-1)

    per_utt = {"dnsmos": [], "utmosv2": [], "speaker_cosine": []}
    wer_refs, wer_hyps = [], []
    per_utt_records = {}
    with torch.no_grad():
        for it in tqdm(ok, desc="scoring"):
            uid = it["uid"]
            m = meta.get(uid, {})
            # same read path as the reference (sf.read -> float32); all four corpora are mono
            pred_wav, pred_sr = sf.read(it["pred_wav"])
            pred_wav = pred_wav.astype(np.float32)
            gt_wav, gt_sr = sf.read(it["gt_wav"])
            gt_wav = gt_wav.astype(np.float32)
            pred16 = to16k(pred_wav, pred_sr)
            rec = per_utt_records.setdefault(uid, {
                "speaker": it["speaker"], "accent_label": it["accent_label"],
                "xtts_accent": m.get("xtts_accent", it["xtts_accent"]), "seed": m.get("seed"),
                "gen_wall_seconds": m.get("gen_wall_seconds"),
                "pred_seconds": round(len(pred_wav) / pred_sr, 3), "gt_seconds": round(len(gt_wav) / gt_sr, 3)})

            if not args.skip_audio:
                d = dnsmos_score(pred16)
                per_utt["dnsmos"].append(d)
                rec.update({f"dnsmos_{k}": v for k, v in d.items()})
                u = utmos_score(pred16)
                per_utt["utmosv2"].append(u)
                rec["utmosv2"] = u

            if not args.skip_wer:
                hyp = whisper_transcribe(pred16)
                ref_lower, hyp_lower = it["text"].lower(), hyp.lower()
                wer_refs.append(ref_lower)
                wer_hyps.append(hyp_lower)
                rec.update({
                    "wer": jiwer.wer(ref_lower, hyp_lower),
                    "wer_reference": ref_lower,
                    "wer_hypothesis": hyp_lower,
                    "wer_reference_normalized": normalize(it["text"]),
                    "wer_hypothesis_normalized": normalize(hyp),
                })

            if not args.skip_speaker:
                pred_sec, gt_sec = len(pred_wav) / pred_sr, len(gt_wav) / gt_sr
                if pred_sec < MIN_ECAPA_SEC or gt_sec < MIN_ECAPA_SEC:
                    tqdm.write(f"WARNING: skipping speaker_cosine for {uid} -- too short for ECAPA-TDNN "
                               f"(pred={pred_sec:.3f}s, gt={gt_sec:.3f}s, min={MIN_ECAPA_SEC}s)")
                else:
                    c = cosine(ecapa_embed(pred16), ecapa_embed(to16k(gt_wav, gt_sr)))
                    per_utt["speaker_cosine"].append(c)
                    rec["speaker_cosine"] = c

            if args.wav_pairs_dir:
                uid_safe = uid.replace("/", "_")
                sf.write(os.path.join(args.wav_pairs_dir, f"{uid_safe}_pred.wav"), pred16.astype(np.float32), 16000)
                sf.write(os.path.join(args.wav_pairs_dir, f"{uid_safe}_gt.wav"), to16k(gt_wav, gt_sr).astype(np.float32), 16000)

    # ---- aggregate -----------------------------------------------------------------------------
    metrics = {}
    if not args.skip_audio:
        for k in ("p808", "sig", "bak", "ovr"):
            metrics[f"dnsmos_{k}"] = summarize([d[k] for d in per_utt["dnsmos"]])
        metrics["utmosv2"] = summarize(per_utt["utmosv2"])
    if not args.skip_wer:
        metrics["wer"] = {"mean": jiwer.wer(wer_refs, wer_hyps) if wer_refs else None, "ci95": None, "n": len(wer_refs)}
        # Normalized variant, same pair rule as articulatory-tts eval_full_testset.py after its GH #32
        # fix (commit b040aa1, 2026-09-08) and the sibling CosyVoice3/EmoSphere++ evals: a pair whose
        # normalized REFERENCE is empty is skipped; an empty hypothesis against a non-empty reference
        # stays in as a full deletion (jiwer's batch API accepts individual empty hypotheses).
        pairs = [(r["wer_reference_normalized"], r["wer_hypothesis_normalized"]) for r in per_utt_records.values()
                 if "wer_reference_normalized" in r and r["wer_reference_normalized"].strip()]
        metrics["wer_whisper_normalized"] = {
            "mean": jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None,
            "ci95": None, "n": len(pairs)}
    if not args.skip_speaker:
        metrics["speaker_cosine"] = summarize(per_utt["speaker_cosine"])

    def breakdown(key):
        groups = {}
        for rec in per_utt_records.values():
            groups.setdefault(rec.get(key) or "unknown", []).append(rec)
        out = {}
        for g, recs in sorted(groups.items()):
            entry = {"n": len(recs)}
            for mkey in ("utmosv2", "dnsmos_ovr", "dnsmos_p808", "speaker_cosine"):
                vals = [x[mkey] for x in recs if mkey in x]
                if vals:
                    entry[mkey] = summarize(vals)
            refs = [x["wer_reference"] for x in recs if "wer_reference" in x]
            hyps = [x["wer_hypothesis"] for x in recs if "wer_hypothesis" in x]
            if refs:
                entry["wer"] = {"mean": jiwer.wer(refs, hyps), "ci95": None, "n": len(refs)}
            out[g] = entry
        return out

    failed_uids = [it["uid"] for it in missing]
    results = {
        "checkpoint": args.checkpoint, "dataset": args.dataset,
        "split": os.path.basename(DATASETS[args.dataset]["split_path"]),
        "prompt_mode": args.prompt_mode,
        "n_split_total": n_split_total,
        "n_utterances": len(items), "n_predicted": len(ok), "n_synthesis_failed": len(missing),
        "failed_uids": failed_uids,
        "synthesis_errors": {u: meta[u].get("error") for u in failed_uids if u in meta and meta[u].get("status") == "failed"},
        "metrics": metrics,
    }
    if not args.skip_wer and len({it["accent_label"] for it in ok}) > 1:
        results["by_accent"] = breakdown("accent_label")
    n_spk = len({it["speaker"] for it in ok})
    if 1 < n_spk <= 50:
        results["by_speaker"] = breakdown("speaker")
    print(json.dumps(metrics, indent=2))

    if args.per_utt_out:
        write_json_atomic(per_utt_records, args.per_utt_out, indent=1)
        print(f"wrote per-utterance metrics for {len(per_utt_records)} utterances to {args.per_utt_out}")
    write_json_atomic(results, args.results_path, indent=2)
    print(f"wrote results to {args.results_path}")


if __name__ == "__main__":
    main()
