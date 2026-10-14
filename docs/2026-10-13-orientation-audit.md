# Orientation audit (numbers only, 2026-10-13)

Pipeline (`src/hidrometros/prep.py`, `orientation.py`, `build.py`), 1243 valid images:

| Step | Count |
|---|---|
| 1. Rectified by `cv2.minAreaRect` and warped to 384x64 | 1243 |
| Rotated by geometry (rectangle long axis not within 30 degrees of horizontal) | 136 (84 vertical, 52 tilted) |
| 2. Red-wheel rule: upright / upside-down / ambiguous / no red | 840 / 39 / 31 / 333 |
| 3. Left to the self-supervised classifier (ambiguous + no red) | 364 |
| Classifier: held-out accuracy on train strips, full strip / right 3/8 masked | 0.989 / 0.985 (5-model ensemble, CPU) |
| Candidate 180-degree flips: by rule / by classifier | 39 / 26 |
| 4. Audit: false flips corrected (rule / classifier) | 3 / 6 |
| Flips kept (rule / classifier) | 36 / 20 |
| **Strips flipped 180 degrees in the final dataset** | **56** |
| Manual overrides in `data_overrides.csv` (all `flip=False`) | 9 |

Notes:
- The classifier is trained only on train-split strips that the red rule judged upright (positives) plus their 180-degree rotations (negatives), never on val/test images, and its decisions were not tuned on any split. It runs on CPU with five seeds averaged, so the decisions are reproducible.
- An earlier single-model run on MPS differed from the CPU ensemble on 6 borderline strips; all 6 were re-inspected. Single-model decisions with p(upright) between 0.3 and 0.5 were unreliable (4 of 5 such flips were wrong), which is why the final decision uses the ensemble and why every borderline strip was inspected.
- The red rule fails when a red object other than the decimal wheels sits in the left quarter of the strip (3 false flips). The classifier and the rule disagreed on exactly those 3 strips.

Audit coverage (strips viewed individually on local contact sheets with the label overlaid, not committed):
- all 65 candidate flips (39 rule + 26 classifier): 9 wrong, corrected through `data_overrides.csv`;
- 50 random non-flipped strips: 0 wrong;
- the 54 classifier-decided strips with p(upright) in (0.02, 0.98), flipped or not: 5 of them were wrong flips (counted above), 0 wrong non-flips;
- the 4 strips where the rule said upright and the classifier disagreed: 0 wrong.
- After the audit: strips found still wrong = 0 among the roughly 165 strips inspected. The other strips (about 1080) were not inspected one by one; the 50 random non-flipped strips give no evidence of a missed flip, but with n = 50 the 95% upper bound on the missed-flip rate of the non-flipped population is about 6%.
- The final 56 flipped strips were viewed again after the overrides: all upright.

Wrong-crop types that survive (inputs to answer 3.2, from the sampled contact sheets, not an exhaustive count):
- 5-wheel displays: 36 images have decimals `000`; in a random sample of 24 of them, 22 have only 5 wheels, so the 3 decimal labels are phantom zeros and the equal-width grid cuts 5 wide cells into 8 slices.
- Cells not aligned with the 48 px grid (a digit split across two crops) and half-visible rolling wheels, mostly in the last decimal wheel: seen in most sampled strips; quantified with the better splitter in the extra challenge (see README).
