# Deviations from the spec and execution notes (2026-10-13)

Spec v1.3 (`docs/SPEC.md`) is kept verbatim. Everything that departs from it, or that it left open, is listed here.

## Deviations

1. **Orientation classifier is a 5-seed CPU ensemble**, not a single tiny model. A single-model run on MPS differed from the CPU ensemble on 6 borderline strips; single decisions with p(upright) in (0.3, 0.5) were unreliable (4 of 5 such flips were wrong). Trained only on train-split strips. See `2026-10-13-orientation-audit.md`.
2. **`data_overrides.csv` has 9 entries, all `flip=False`** (audit corrections of false automatic flips: 3 from the red rule, 6 from the classifier).
3. **Frozen epoch 1 keeps the backbone BatchNorm layers in eval mode** (running statistics are not updated while the backbone is frozen); the head and the optimizer groups are as in the spec.
4. **VGG16 adaptive average pooling runs on the CPU when the model is on MPS** (the MPS backend does not implement the non-divisible 4x4 to 7x7 resize at 128x128 input). Numerics are those of the reference layer; `ms_per_image` for `vgg16_bn` (20.80 ms) includes this round trip.
5. **Smoke test used 2 epochs** (spec said 1) so the frozen epoch and the unfreeze step are both exercised. 256 train + 128 val crops; all four models passed (wall seconds smallcnn 44.3, alexnet 49.5, vgg16_bn 83.8, resnet18 49.1, from `logs/smoke2.log`).
6. **`weights="DEFAULT"` of the scaffold replaced by explicit `IMAGENET1K_V1`**; per-model LR groups and input sizes fixed; scaffold credited to Mikael (git author of `152bc8b`).
7. **`s_per_epoch` is the median over all epochs run, including the frozen first epoch**, which is faster for pretrained models. Median, so one epoch does not dominate. Training was timed with `time.perf_counter()` around the training loop with `torch.mps.synchronize()` before starting and stopping the clock.
8. **The VGG16 epoch was timed first with a 50-step measurement** (1.154 s/step at batch 16, extrapolated to 502 s/epoch and 67 min for 8 epochs). The real run took 58.8 min in total (median 473.6 s/epoch). No epoch was shortened for any model.
9. **ImageNet mean/std is also applied to `smallcnn`** (protocol sameness; spec asked to justify it). Not ablated.
10. **Early stopping stopped `smallcnn` at epoch 4 (best epoch 1)**; the other three ran all 8 epochs. Result reported as is (test acc_digit 0.2246, below the 0.344 majority-class rate). Diagnostic outside the protocol, not in `results.csv`: `smallcnn` for 10 epochs without early stopping (`hidrometros train smallcnn --variant diag --no-early-stop`, local only): val F1-macro at epoch 10 = 0.1699; test acc_digit 0.4158, f1_macro 0.1807, acc_reading 0.0000. The cause of the failure was not isolated.
11. **Extra challenge (F6) result is negative**: the located-boundary split lowered every pretrained model (see `results_extra.csv`, README). The splitter's parameters (grid search ranges, prior weight 0.004, minimum right edge 372) were set by visual inspection of 16 random strips (drawn from all splits, only to look at boundary placement; no labels or predictions involved) and a boundary-ink indicator computed on 300 train images; the indicator uses the same signal the splitter optimizes, so it is optimistic. Test data were not used to choose any parameter.
12. **Workers and data loading:** crops are preloaded in memory as uint8 tensors; `DataLoader` uses 4 workers (spawn on macOS) with seeded generator and `worker_init_fn`.
13. **Notebook**: `notebooks/pratica_watermeters.ipynb` is a thin notebook committed with no outputs and not executed here. The gallery of the 16 most confident errors is a cell in it, to be run locally by the group and cleared before commit (not run by the agent).
14. **Guard tests (criterion 14)** were proven red afterwards by mutation and `git stash`, not by writing each test before its code; see `2026-10-13-guard-red.md`.
15. **Wrong-crop counts (answer 3.2)** are exact for rotated (136) and flipped (56) strips; the 5-wheel count comes from a 24-image sample (22 of 24) of the 36 images with decimals `000`, and the "ink on the border" indicator is an approximation, not a per-digit manual check.
16. **MPS timing and noise:** `ms_per_image` and `s_per_epoch` come from a single run on one Apple M1 with other processes present; no repeated runs were made, so the run-to-run noise is not measured.
17. **Guard fix found in F5:** the figure-provenance check was case-sensitive and rejected genuine matplotlib PNGs (fixed with a test, see `2026-10-13-guard-red.md`).
18. **Repo is still PRIVATE.** Criterion 13 (public, loads logged out) fails by design until the user passes gate 2. An aborted background poll loop (my own `pgrep` matched itself) was killed; no training process was left running.

## Run log (local, not committed)

Training logs: `logs/train_base.log`, `logs/train_extra.log`, `logs/diag_smallcnn.log`. Checkpoints in `checkpoints/` (gitignored).
