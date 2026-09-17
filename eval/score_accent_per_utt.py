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

2026-09-16: the published accent_cosine is CENTERED -- both the prediction and
ground-truth embeddings have genaid_accent.DEFAULT_CENTER_VECTOR (the mean of
the six speaker-balanced VCTK-training-speaker accent centroids; see the
reference repo's CLAUDE.md "Accent-metric diagnostic") subtracted before the
cosine, via genaid_accent.accent_cosines(). The raw (uncentered) GenAID cosine
is kept alongside as accent_cosine_genaid_raw, per utterance AND per group
(--no_center restores raw-only behaviour; --center_vector swaps the vector --
same flags as the reference repo's score_side_metric.py/score_side_per_utt.py).
The by_accent/by_speaker summaries and the overall consistency assertion
against the merged metrics.accent_cosine both use whichever value this run
actually wrote as accent_cosine (centered by default), since the merge this
script checks against comes from the same score_side_metric.py invocation.
When re-run over a results JSON whose by_accent/by_speaker groups still carry
a PREVIOUS (raw-GenAID) per-group accent_cosine, that old value is preserved
under accent_cosine_genaid_raw first, unless already present (idempotent
against re-runs); pre-existing *_commonaccent keys (from the 2026-09-14
CommonAccent -> GenAID rescore) are never touched. The centering vector
actually used (or null under --no_center) is recorded in
results["accent_center_vector"].

As extra context for accent analysis it also records the classifier's own
top label for the prediction and for the ground truth (GenAID's 13-way
label) and, per accent group, how often each side is labelled with the
intended class. That is an accent-classification view, not the
embedding-cosine metric (unaffected by centering, since centering only
changes the cosine, not the classifier's argmax); it is stored under
"accent_label_agreement", never under accent_cosine.

2026-09-14: GenAID replaces CommonAccent (Jzuluaga/accent-id-commonaccent_xlsr-en-english,
speechbrain foreign_class encode_batch()); see the reference repo's
genaid_accent.py for why. Run under /data/user_data/xoy/venvs/eval-genaid/bin/python
(the same venv the reference side metric uses) -- see eval/run_accent_per_utt.sbatch
and eval/run_rescore_accent_centered.sbatch.
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
    ap.add_argument("--center_vector", default=None,
                    help="path of the (64,) .npy centering vector subtracted from both embeddings before "
                         "the cosine (default: genaid_accent.DEFAULT_CENTER_VECTOR, the 2026-09-16 centroid mean)")
    ap.add_argument("--no_center", action="store_true", help="report the raw (uncentered) GenAID cosine only")
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    import genaid_accent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    center = None if args.no_center else (args.center_vector or genaid_accent.DEFAULT_CENTER_VECTOR)
    mu = genaid_accent.load_center_vector(center) if center else None
    model_tag = genaid_accent.centered_model_tag(center) if center else genaid_accent.MODEL_TAG

    items = load_items(args.dataset)
    if args.limit:
        items = items[: args.limit]
    # identical model/loading to score_side_metric.py's score_accent()
    embedder = genaid_accent.GenAIDEmbedder(device)
    print("accent model:", model_tag)
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
            c, r = genaid_accent.accent_cosines(embed(pred), embed(gt), mu)
            records[it["uid"]] = {
                "speaker": it["speaker"], "accent_label": it["accent_label"], "xtts_accent": it["xtts_accent"],
                "accent_cosine": c, "accent_cosine_genaid_raw": r,
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
            entry = {"n": len(recs), "accent_cosine": summarize([r["accent_cosine"] for r in recs]),
                     "accent_cosine_genaid_raw": summarize([r["accent_cosine_genaid_raw"] for r in recs])}
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
                e = results[field].setdefault(g, {})
                # Keep a previous raw-GenAID per-group accent_cosine (written by an earlier, uncentered run
                # of this script) as accent_cosine_genaid_raw, unless already present -- idempotent against
                # re-runs. Never touches *_commonaccent keys (a separate, unrelated axis from the 2026-09-14
                # CommonAccent -> GenAID rescore) or accent_label_agreement (unaffected by centering).
                if "accent_cosine" in e and "accent_cosine_genaid_raw" not in e:
                    e["accent_cosine_genaid_raw"] = e.pop("accent_cosine")
                e.update(entry)
    results["accent_cosine_per_utt_source"] = (f"eval/score_accent_per_utt.py (same model/call path as articulatory-tts "
                                               f"score_side_metric.py; {model_tag})")
    results["accent_center_vector"] = os.path.abspath(center) if center else None

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
            acc = e.get("accent_cosine") or {}
            acc_str = f"{acc['mean']:10.4f}" if acc.get("mean") is not None else f"{'--':>10s}"
            print(f"{g:14s} {e.get('n', 0):5d} {acc_str} "
                  f"{agr.get('pred_labelled_as_target', float('nan')):12.3f} {agr.get('gt_labelled_as_target', float('nan')):10.3f}")
    print(f"wrote {args.per_utt_out} and updated {args.results_path}")


if __name__ == "__main__":
    main()
