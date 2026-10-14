# smallcnn: root cause of "it did not learn" (2026-10-14)

## Verdict

The failure was a **head design defect**, not data, labels, optimizer, early stopping or "small nets cannot do it". The scaffold's `SmallCNN` ended in `AdaptiveAvgPool2d((1, 1))`. With three 3x3 convs and two 2x2 max-pools the receptive field is 18 px on a 64x48 crop, and the global average then collapses the whole 16x12 map into 128 position-free numbers. The classifier only sees a bag of local stroke features and cannot tell where a stroke is, which is what separates digits (6 vs 9, 2 vs 5, 1 vs 7). The model stayed on the majority class (digit 0, 35.7% of crops).

Fix (confined to `SmallCNN`): drop the global pool, flatten the 128x16x12 map into `Linear(24576, 10)`. Trunk, input 64x48, 10 epochs, optimizer, LR, loss, seed, splits and early stopping are untouched. Parameters 0.09 M -> 0.34 M.

## Experiments (one change at a time, 10 epochs, validation split only, never in `results.csv`)

Script: `docs/2026-10-14-smallcnn-experiments/exp.py` (+ `archs.py`, `runall.sh`, logs `e*.log`). Same crops, seed 42, batch 128, MPS. "protocol" = AdamW(1e-3) + OneCycleLR + label smoothing 0.1 + protocol augmentation + ImageNet normalization. Val F1 is macro-F1 at epoch 10.

| id | change from the scaffold under the protocol | train acc | val acc | val F1 |
|---|---|---|---|---|
| overfit (`overfit64_gap.log`, scaffold head) | 64 crops, no aug, constant LR, no smoothing, 60 epochs | 1.00 (epoch 56) | n/a | n/a |
| e0 | none (scaffold, no early stopping) | 0.432 | 0.426 | 0.170 |
| e1 | no augmentation | 0.491 | 0.479 | 0.267 |
| e2 | no augmentation, constant LR (no OneCycle warm-up) | 0.573 | 0.453 | 0.205 |
| e8 | no normalization | 0.439 | 0.438 | 0.194 |
| e7 | LR 1e-2 (protocol change, diagnostic only) | 0.698 | 0.706 | 0.616 |
| e6 | third max-pool before the global average | 0.468 | 0.462 | 0.236 |
| e5 | pool to a 4x3 grid and flatten (keeps coarse layout) | 0.761 | 0.764 | 0.708 |
| e4 | third max-pool, flatten 8x6 map | 0.830 | 0.812 | 0.772 |
| **e3** | **flatten the 16x12 map (chosen)** | **0.854** | **0.825** | **0.785** |
| e9 | e3 without augmentation | 0.986 | 0.774 | 0.718 |

Note: the logs e0, e1, e2, e7, e8 were produced with the scaffold head before the fix; `--arch orig` now builds the fixed `SmallCNN`, so to reproduce them use `--arch gap` (added to `archs.py`).

Reading:
1. The overfit test passes (100% train accuracy on 64 crops), so the input path, labels, loss and loop work. Crops are 64 high x 48 wide (PIL size 48x64), matching `input_hw=(64, 48)`; a contact sheet of random crops showed correct digits (noisy real photos, mixed colours, partially cut neighbours).
2. e0 reproduces the earlier diagnostic (val F1 0.170 vs 0.1699 logged before). Normalization (e8), augmentation (e1) and the OneCycle warm-up (e2) are not the cause: removing each leaves the model far below 0.8.
3. Only the head matters: keeping spatial layout (e3, e4, e5) trains, a global average does not (e0, e6), and e7 shows a 10x LR partly hides the problem by brute force (not allowed by the protocol and still 0.62).
4. Early stopping was not the cause: e0 ran all 10 epochs and still failed. Under the fixed model the run never triggers it (F1 rises every epoch, best epoch 10).
5. e9: protocol augmentation helps the fixed model (val F1 0.718 -> 0.785), so it is not too strong for 64x48 crops.
6. The architecture was picked on validation F1 only (e3 vs e4 differ by 0.013, within single-run noise; e3 chosen as the smallest edit). Test data were not used.

## Result under the official protocol (test split)

`results.csv` smallcnn: acc_digit 0.8155, f1_macro 0.7711, acc_reading 0.3102, 0.34 M params, best_epoch 10 (before: 0.2246, 0.0904, 0.0000, 0.09 M, best_epoch 1). Extra challenge (`results_extra.csv`): 0.7527, 0.6988, 0.1818 (before: 0.2259, 0.0886, 0.0000).

## What was retrained and why

Only `smallcnn` (base and extra variants). The shared training code (`train.py`, protocol, transforms, early stopping) was **not** changed, so the other three models' results remain valid and their rows in `results.csv` / `results_extra.csv` are byte-identical to before. `evaluate.py` gained `--models` so one model can be re-evaluated without re-measuring the cost columns of the others (a tooling change; full evaluation behaves as before). Figures, reports and README tables were regenerated.

## Guard

`tests/test_smallcnn.py::test_smallcnn_uses_spatial_layout`: synthetic task where the label is only the location of a bright square on a 5x2 grid. RED before the fix (accuracy 0.404, 1 failed of 2 tests in the file), GREEN after. See `2026-10-13-guard-red.md`.

Diagnostics outside the protocol that are now obsolete: the "10 epochs without early stopping" run in `spec-deviations` item 10 (it was measuring the defective head).
