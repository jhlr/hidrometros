"""F3 orchestrator: rectify -> orient -> crop -> index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hidrometros import data, orientation, prep


def run_prep() -> dict[str, object]:
    """Build strips, fix orientation, write `digits/` and `digits/index.csv`.

    Returns the statistics dict (also written to `local/prep_stats.json`).
    """
    ann = data.load_annotations()
    discarded = ann[ann["label"].isna() | ann["split"].isna()]["photo_name"].tolist()
    valid = data.valid_annotations()
    strips, geom = prep.build_strips(valid)
    geom = geom.merge(valid[["photo_name", "split", "label", "image_id"]], on="photo_name")

    # step 2: red rule decides where it has evidence
    geom["flip_rule"] = geom["red_rule"] == "upside_down"
    undecided = geom["red_rule"].isin(["ambiguous", "no_red"])

    # step 3: classifier fitted on train-split upright strips only, decides the rest
    train_up = geom[(geom["split"] == "train") & (geom["red_rule"] == "upright")]["photo_name"]
    models, clf_metrics = orientation.train_ensemble([strips[n] for n in train_up])
    p_up = orientation.prob_upright(models, [strips[n] for n in geom["photo_name"]])
    geom["p_upright"] = p_up
    geom["flip_clf"] = undecided & (geom["p_upright"] < 0.5)

    # consistency of the classifier with the rule where the rule is confident (info only)
    decided = geom[~undecided].copy()
    decided["clf_agrees"] = (decided["p_upright"] >= 0.5) == (decided["red_rule"] == "upright")
    agree_by_split = decided.groupby("split")["clf_agrees"].mean().round(4).to_dict()

    # step 5: manual audit overrides
    overrides = prep.load_overrides()
    auto_flip = geom["flip_rule"] | geom["flip_clf"]
    geom["override"] = geom["photo_name"].map(lambda n: overrides.get(n))
    geom["flip_final"] = np.where(geom["override"].notna(), geom["override"].fillna(False).astype(bool), auto_flip)

    final = {n: (prep.rotate180(s) if f else s) for (n, s), f in zip(strips.items(), geom["flip_final"], strict=True)}
    geom.to_csv(prep.ORIENTATION_CSV, index=False)
    prep.save_strips(final)

    index = prep.write_crops(final, valid, data.DIGITS_DIR)
    index.to_csv(data.DIGITS_DIR / "index.csv", index=False)

    stats: dict[str, object] = {
        "n_annotated": len(ann),
        "n_valid": len(valid),
        "discarded": discarded,
        "n_crops": len(index),
        "crops_per_split": index["split"].value_counts().to_dict(),
        "images_per_split": valid["split"].value_counts().to_dict(),
        "rotated_by_geometry": int(geom["rotated_geometry"].sum()),
        "red_rule_counts": geom["red_rule"].value_counts().to_dict(),
        "flipped_by_red_rule": int(geom["flip_rule"].sum()),
        "flipped_by_classifier": int(geom["flip_clf"].sum()),
        "overrides_applied": len(overrides),
        "audit_false_flips_corrected_rule": int((geom["flip_rule"] & ~geom["flip_final"]).sum()),
        "audit_false_flips_corrected_clf": int((geom["flip_clf"] & ~geom["flip_final"]).sum()),
        "flips_kept_rule": int((geom["flip_rule"] & geom["flip_final"]).sum()),
        "flips_kept_clf": int((geom["flip_clf"] & geom["flip_final"]).sum()),
        "flipped_final": int(geom["flip_final"].sum()),
        "undecided_by_rule": int(undecided.sum()),
        "classifier_train_metrics": clf_metrics,
        "classifier_agreement_with_rule_by_split": agree_by_split,
        "digit_distribution": index["digit"].value_counts().sort_index().to_dict(),
        "images_in_multiple_splits": int((index.groupby("photo_name")["split"].nunique() > 1).sum()),
    }
    prep.stats_to_json(stats)
    return stats
