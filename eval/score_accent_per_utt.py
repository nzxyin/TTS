"""Per-utterance accent similarity, for per-accent / per-speaker breakdowns.

articulatory-tts's score_side_metric.py --metric accent only writes the
aggregate {mean, ci95, n}. This script embeds the same 16 kHz
{uid}_pred.wav/{uid}_gt.wav pairs with the SAME model and call path
(Jzuluaga/accent-id-commonaccent_xlsr-en-english via speechbrain
foreign_class, encode_batch() = pooled wav2vec2-XLSR encoder output before
the classification head, cosine per pair, 0.1 s minimum duration) but keeps
every per-utterance value, then folds per-accent and per-speaker summaries
into the dataset's results JSON ("by_accent"/"by_speaker" -> accent_cosine)
and asserts the overall mean reproduces the merged accent_cosine.

As extra context for accent analysis it also records the classifier's own
top label for the prediction and for the ground truth (classify_file ->
16-way CommonAccent label) and, per accent group, how often each side is
labelled with the intended CommonAccent class. That is an
accent-classification view, not the embedding-cosine metric; it is stored
under "accent_label_agreement", never under accent_cosine.

Run under /data/user_data/xoy/venvs/eval-accent/bin/python (the same venv
the reference side metric uses) -- see eval/run_accent_per_utt.sbatch.
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

MIN_PAIR_DURATION_SEC = 0.1  # same floor as score_side_metric.py

# VCTK speaker-info accent group -> the CommonAccent classifier's own 16 labels, where one exists
# (used only for the label-agreement view). Classifier label set: african, australia, bermuda,
# canada, england, hongkong, indian, ireland, malaysia, newzealand, philippines, scotland,
# singapore, southatlandtic, us, wales.
VCTK_TO_COMMONACCENT = {"American": "us", "Canadian": "canada", "English": "england", "Irish": "ireland",
                        "NorthernIrish": "ireland", "Scottish": "scotland", "Australian": "australia",
                        "Indian": "indian", "Welsh": "wales", "SouthAfrican": "african", "NewZealand": "newzealand"}


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
    from speechbrain.inference.interfaces import foreign_class
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_items(args.dataset)
    if args.limit:
        items = items[: args.limit]
    # identical construction to score_side_metric.py (savedir / save_path overrides included)
    classifier = foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": str(device)},
        savedir=os.path.join(os.environ["HF_HOME"], "speechbrain", "commonaccent_xlsr"),
        overrides={"wav2vec2": {"save_path": os.path.join(os.environ["HF_HOME"], "wav2vec2_checkpoints")}},
    )

    def embed(path):
        waveform = classifier.load_audio(path)
        return classifier.encode_batch(waveform.unsqueeze(0)).detach().cpu().numpy().reshape(-1)

    def label(path):
        out_prob, score, index, text_lab = classifier.classify_file(path)
        lab = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
        if hasattr(score, "detach"):  # cuda tensor -> host float
            score = score.detach().cpu()
        return lab, float(np.asarray(score).reshape(-1)[0])

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
                "pred_commonaccent_label": pl, "pred_commonaccent_score": ps,
                "gt_commonaccent_label": gl, "gt_commonaccent_score": gs,
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
            target = VCTK_TO_COMMONACCENT.get(recs[0]["accent_label"]) if key in ("accent_label", "speaker") else None
            if target:
                entry["accent_label_agreement"] = {
                    "target_commonaccent_label": target,
                    "pred_labelled_as_target": float(np.mean([r["pred_commonaccent_label"] == target for r in recs])),
                    "gt_labelled_as_target": float(np.mean([r["gt_commonaccent_label"] == target for r in recs])),
                    "pred_matches_gt_label": float(np.mean([r["pred_commonaccent_label"] == r["gt_commonaccent_label"] for r in recs])),
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
    results["accent_cosine_per_utt_source"] = "eval/score_accent_per_utt.py (same model/call path as articulatory-tts score_side_metric.py)"

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
