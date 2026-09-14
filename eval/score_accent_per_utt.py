"""Per-utterance accent similarity, for per-accent / per-speaker breakdowns.

articulatory-tts's score_side_metric.py --metric accent only writes the
aggregate {mean, ci95, n}. This script embeds the same 16 kHz
{uid}_pred.wav/{uid}_gt.wav pairs with the SAME model and call path
(GenAID, /home/xoy/articulatory-tts/genaid_accent.py: the authors' GenAID_v6
checkpoint, 64-dim preout_mlp embedding after statistics pooling over the
fine-tuned XLSR-53 frames, cosine per pair, 0.1 s minimum duration) but keeps
every per-utterance value, then folds per-accent and per-speaker summaries
into the dataset's results JSON ("by_accent"/"by_speaker" -> accent_cosine)
and asserts the overall mean reproduces the merged accent_cosine.

As extra context for accent analysis it also records the classifier's own
top label for the prediction and for the ground truth (GenAID's 13-way
label) and, per accent group, how often each side is labelled with the
intended class. That is an accent-classification view, not the
embedding-cosine metric; it is stored under "accent_label_agreement", never
under accent_cosine.

2026-09-14: GenAID replaces CommonAccent (Jzuluaga/accent-id-commonaccent_xlsr-en-english,
speechbrain foreign_class encode_batch()); see the reference repo's
genaid_accent.py for why. Run under /data/user_data/xoy/venvs/eval-genaid/bin/python
(the same venv the reference side metric uses) -- see eval/run_accent_per_utt.sbatch.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/data/user_data/xoy/.cache/huggingface")

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize_testset import load_items  # noqa: E402

ART_REPO = "/home/xoy/articulatory-tts"  # genaid_accent.py lives with the reference side metric
sys.path.insert(0, ART_REPO)

MIN_PAIR_DURATION_SEC = 0.1  # same floor as score_side_metric.py

# VCTK speaker-info accent group -> GenAID's 13 labels, where one exists (label-agreement view only):
# genaid_accent.VCTK_TO_GENAID (us, canadian, english, irish [also NorthernIrish], scottish, ...).


def summarize(values):
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": int(len(values))}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def too_short(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate < MIN_PAIR_DURATION_SEC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wav_pairs_dir", required=True)
    ap.add_argument("--results_path", required=True, help="eval_<dataset>.json to update in place (by_accent/by_speaker)")
    ap.add_argument("--per_utt_out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    import genaid_accent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.dataset)
    if args.limit:
        items = items[: args.limit]
    # identical model/loading to score_side_metric.py's score_accent()
    embedder = genaid_accent.GenAIDEmbedder(device)
    print("accent model:", genaid_accent.MODEL_TAG)
    VCTK_TO_GENAID = genaid_accent.VCTK_TO_GENAID
    cache = {}

    def run(path):
        if path not in cache:
            cache[path] = embedder.embed_and_label(path)
        return cache[path]

    def embed(path):
        return run(path)[0]

    def label(path):
        return run(path)[1], run(path)[2]

    records = {}
    n_skipped = 0
    with torch.no_grad():
        for it in tqdm(items, desc="accent embeddings"):
            uid_safe = it["uid"].replace("/", "_")
            pred = os.path.join(args.wav_pairs_dir, f"{uid_safe}_pred.wav")
            gt = os.path.join(args.wav_pairs_dir, f"{uid_safe}_gt.wav")
            if not (os.path.exists(pred) and os.path.exists(gt)):
                n_skipped += 1
                continue
            if too_short(pred) or too_short(gt):
                tqdm.write(f"WARNING: skipping {it['uid']} -- pred/gt wav too short (min={MIN_PAIR_DURATION_SEC}s)")
                n_skipped += 1
                continue
            pl, ps = label(pred)
            gl, gs = label(gt)
            records[it["uid"]] = {
                "speaker": it["speaker"], "accent_label": it["accent_label"], "xtts_accent": it["xtts_accent"],
                "accent_cosine": cosine(embed(pred), embed(gt)),
                "pred_label": pl, "pred_label_prob": ps,
                "gt_label": gl, "gt_label_prob": gs,
            }
    print(f"scored {len(records)} pairs, skipped {n_skipped}")

    overall = summarize([r["accent_cosine"] for r in records.values()])
    print("overall accent_cosine:", overall)

    def by(key):
        groups = {}
        for r in records.values():
            groups.setdefault(r[key], []).append(r)
        out = {}
        for g, recs in sorted(groups.items()):
            entry = {"n": len(recs), "accent_cosine": summarize([r["accent_cosine"] for r in recs])}
            target = VCTK_TO_GENAID.get(recs[0]["accent_label"]) if key in ("accent_label", "speaker") else None
            if target:
                entry["accent_label_agreement"] = {
                    "target_label": target,
                    "pred_labelled_as_target": float(np.mean([r["pred_label"] == target for r in recs])),
                    "gt_labelled_as_target": float(np.mean([r["gt_label"] == target for r in recs])),
                    "pred_matches_gt_label": float(np.mean([r["pred_label"] == r["gt_label"] for r in recs])),
                }
            out[g] = entry
        return out

    with open(args.results_path) as f:
        results = json.load(f)
    merged = results.get("metrics", {}).get("accent_cosine", {}).get("mean")
    if merged is not None and overall["mean"] is not None and not args.limit:
        diff = abs(merged - overall["mean"])
        print(f"consistency vs merged accent_cosine {merged:.4f}: |diff|={diff:.5f}")
        assert diff < 2e-3, "per-utterance recomputation does not reproduce the merged accent_cosine -- not writing"
    for key, field in (("accent_label", "by_accent"), ("speaker", "by_speaker")):
        if len({r[key] for r in records.values()}) > 1:
            results.setdefault(field, {})
            for g, entry in by(key).items():
                results[field].setdefault(g, {}).update(entry)
    results["accent_cosine_per_utt_source"] = (f"eval/score_accent_per_utt.py (same model/call path as articulatory-tts "
                                               f"score_side_metric.py; {genaid_accent.MODEL_TAG})")

    for path, obj, kw in ((args.per_utt_out, records, {"indent": 1}), (args.results_path, results, {"indent": 2})):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, **kw)
        os.replace(tmp, path)
    if "by_accent" in results:
        print(f"{'accent':14s} {'n':>5s} {'accent_cos':>10s} {'pred->target':>12s} {'gt->target':>10s}")
        for g, e in results["by_accent"].items():
            agr = e.get("accent_label_agreement", {})
            print(f"{g:14s} {e['n']:5d} {e['accent_cosine']['mean']:10.4f} {agr.get('pred_labelled_as_target', float('nan')):12.3f} {agr.get('gt_labelled_as_target', float('nan')):10.3f}")
    print(f"wrote {args.per_utt_out} and updated {args.results_path}")


if __name__ == "__main__":
    main()
