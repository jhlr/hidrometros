"""Smoke test: every model, 2 epochs, 256 train + 128 val crops, through the real code path.

Two epochs (not one) so the frozen-backbone epoch AND the unfreeze step are both exercised.
Writes only under local/smoke/.
"""

from __future__ import annotations

import time

import pandas as pd

from hidrometros.evaluate import evaluate_model
from hidrometros.models import MODEL_NAMES
from hidrometros.train import SMOKE, run_training


def main() -> int:
    rows = []
    for name in MODEL_NAMES:
        t0 = time.perf_counter()
        hist = run_training(name, variant=SMOKE, max_epochs=2, limit_train=256, limit_val=128)
        train_s = time.perf_counter() - t0
        row, *_ = evaluate_model(name, SMOKE)
        ok = len(hist) == 2 and bool(hist["train_loss"].notna().all()) and (SMOKE.ckpt_root / name / "last.pt").exists()
        rows.append({"model": name, "epochs": len(hist), "wall_s": round(train_s, 1), "ok": ok, "test_acc": round(float(row["acc_digit"]), 3)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return 0 if bool(df["ok"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
