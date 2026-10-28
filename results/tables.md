**Table 1. Paired change against `standard`, two models on disjoint halves.** Percentage points, mean ± paired SE over 50 trials; Δ accuracy / Δ agreement.

| Arm | Adult (3k) | Breast Cancer | High Noise | Small Sample | High Dim |
|---|---|---|---|---|---|
| `dropout=0.5` | +0.0 ± 0.1 / -0.1 ± 0.2 | +0.1 ± 0.0 / +0.2 ± 0.1 | -0.7 ± 0.3 / -0.3 ± 0.6 | -0.4 ± 0.2 / +0.3 ± 0.3 | -0.5 ± 0.4 / +0.1 ± 0.9 |
| `dropout=0.7` | -0.1 ± 0.1 / -0.9 ± 0.4 | +0.2 ± 0.1 / +0.2 ± 0.2 | -2.4 ± 0.5 / -2.8 ± 1.1 | -0.5 ± 0.3 / -0.4 ± 0.6 | -2.3 ± 0.6 / -4.4 ± 1.5 |
| `wd=3` | +0.3 ± 0.1 / -0.6 ± 0.2 | +0.3 ± 0.1 / +0.0 ± 0.1 | +1.3 ± 0.3 / +1.7 ± 0.6 | +0.4 ± 0.2 / +0.8 ± 0.4 | +0.7 ± 0.2 / +1.1 ± 0.5 |
| `wd=10` | +0.5 ± 0.1 / +1.7 ± 0.2 | +0.3 ± 0.1 / +0.2 ± 0.2 | -0.7 ± 0.4 / +0.5 ± 1.0 | +1.2 ± 0.3 / +3.0 ± 0.7 | +2.2 ± 0.4 / +2.4 ± 0.8 |
| `5 passes` | -0.2 ± 0.1 / -0.7 ± 0.4 | +0.0 ± 0.1 / +0.2 ± 0.1 | +0.3 ± 0.2 / +0.4 ± 0.4 | +0.0 ± 0.2 / -0.2 ± 0.4 | -0.1 ± 0.2 / -0.1 ± 0.6 |
| `DCT lam=1.5` | -0.1 ± 0.1 / -0.7 ± 0.4 | +0.1 ± 0.1 / +0.1 ± 0.1 | +0.3 ± 0.3 / +0.5 ± 0.5 | +0.1 ± 0.1 / +0.3 ± 0.4 | +0.1 ± 0.2 / +0.1 ± 0.5 |
| `DCT lam=5` | +0.0 ± 0.1 / -0.8 ± 0.4 | +0.1 ± 0.0 / +0.2 ± 0.1 | +0.9 ± 0.3 / +1.3 ± 0.5 | +0.4 ± 0.2 / +0.9 ± 0.4 | +0.3 ± 0.2 / +0.3 ± 0.6 |
| `DCT lam=15` | -1.3 ± 0.2 / -1.9 ± 0.6 | +0.2 ± 0.1 / +0.2 ± 0.1 | +0.5 ± 0.3 / +0.6 ± 0.7 | +0.6 ± 0.3 / +1.6 ± 0.6 | +0.4 ± 0.3 / +0.1 ± 0.8 |
| `DCT lam=5, 2 passes` | -0.0 ± 0.1 / -0.7 ± 0.3 | +0.1 ± 0.1 / +0.1 ± 0.1 | +0.7 ± 0.3 / +0.5 ± 0.6 | +0.4 ± 0.2 / +1.0 ± 0.4 | +0.3 ± 0.2 / +0.1 ± 0.6 |
| `bagging-5` | +0.2 ± 0.1 / +1.8 ± 0.3 | +0.2 ± 0.1 / -0.1 ± 0.2 | -0.9 ± 0.3 / -0.5 ± 0.8 | -0.0 ± 0.3 / +1.0 ± 0.7 | -0.3 ± 0.4 / -0.3 ± 1.3 |
| `seed-ens-5` | +0.3 ± 0.1 / +1.0 ± 0.2 | +0.1 ± 0.1 / +0.1 ± 0.1 | +1.9 ± 0.2 / +2.4 ± 0.6 | +0.9 ± 0.2 / +1.7 ± 0.6 | +1.8 ± 0.4 / +4.1 ± 0.9 |
| `data=50%` | -0.7 ± 0.1 / -1.4 ± 0.4 | -0.7 ± 0.1 / -0.8 ± 0.2 | -8.4 ± 0.6 / -12.4 ± 1.4 | -4.3 ± 0.5 / -6.9 ± 1.1 | -6.5 ± 0.6 / -10.4 ± 2.0 |

**Table 2. Agreement relative to the single-pass frontier at the same accuracy.** The frontier is the best of the dropout-rate and weight-decay sweeps at each accuracy (upper envelope, linearly interpolated). Points, mean ± bootstrap SE over trials; negative means the frontier does better.

| Arm | Adult (3k) | Breast Cancer | High Noise | Small Sample | High Dim |
|---|---|---|---|---|---|
| `DCT lam=1.5` | -2.9 ± 0.5 | -0.3 ± 0.2 | -1.6 ± 0.5 | -2.8 ± 0.5 | -2.2 ± 0.7 |
| `DCT lam=5` | -2.8 ± 0.5 | -0.1 ± 0.2 | -0.8 ± 0.5 | -2.2 ± 0.5 | -2.1 ± 0.7 |
| `DCT lam=15` | -5.1 ± 0.6 | -0.1 ± 0.2 | -1.5 ± 0.6 | -1.5 ± 0.5 | -2.2 ± 0.7 |
| `DCT lam=5, 2 passes` | -2.8 ± 0.4 | -0.3 ± 0.2 | -1.6 ± 0.6 | -2.0 ± 0.5 | -2.3 ± 0.7 |
| `bagging-5` | -0.1 ± 0.4 | -0.4 ± 0.2 | -2.6 ± 0.7 | -2.0 ± 0.6 | -2.6 ± 1.2 |
| `seed-ens-5` | -0.8 ± 0.3 | -0.3 ± 0.2 | more accurate than any frontier point | -1.4 ± 0.4 | +1.7 ± 0.7 |

**Table 3. Where the disagreement goes.** Change vs `standard` in disagreement between two seeds on the same half (seed) and between the two halves (sample), in points. `standard` row: levels.

| Arm | Adult (3k) | Breast Cancer | High Noise | Small Sample | High Dim |
|---|---|---|---|---|---|
| `standard` (levels) | 5.2 seed, 7.8 sample | 1.2 seed, 3.1 sample | 13.5 seed, 25.9 sample | 7.8 seed, 15.2 sample | 20.2 seed, 35.0 sample |
| `dropout=0.5` | -0.1 seed, +0.1 sample | -0.0 seed, -0.2 sample | -1.1 seed, +0.3 sample | -0.5 seed, -0.3 sample | -0.5 seed, -0.1 sample |
| `wd=10` | -1.9 seed, -1.7 sample | -0.8 seed, -0.2 sample | -1.6 seed, -0.5 sample | -4.5 seed, -3.0 sample | -6.3 seed, -2.4 sample |
| `DCT lam=5` | -0.4 seed, +0.8 sample | -0.0 seed, -0.2 sample | -2.4 seed, -1.3 sample | -0.9 seed, -0.9 sample | -3.3 seed, -0.3 sample |
| `DCT lam=15` | -1.2 seed, +1.9 sample | -0.1 seed, -0.2 sample | -2.9 seed, -0.6 sample | -2.7 seed, -1.6 sample | -4.3 seed, -0.1 sample |
| `seed-ens-5` | -3.1 seed, -1.0 sample | -0.7 seed, -0.1 sample | -7.3 seed, -2.4 sample | -4.5 seed, -1.7 sample | -11.3 seed, -4.1 sample |
| `bagging-5` | -1.6 seed, -1.8 sample | -0.2 seed, +0.1 sample | -0.6 seed, +0.5 sample | -0.6 seed, -1.0 sample | -1.9 seed, +0.3 sample |

**Table 4. The vote ensemble.** Over the ~30 models that saw each test point: pairwise disagreement, mean error, and overconfidence (mean vote share of the majority class minus majority-vote accuracy). Under a calibrated vote ensemble disagreement would equal error (Jiang et al. 2022).

| Arm | Adult (3k) | Breast Cancer | High Noise | Small Sample | High Dim |
|---|---|---|---|---|---|
| `standard` | 7.2 / 16.5 / +10.6 | 2.9 / 3.4 / +0.5 | 22.5 / 28.6 / +8.2 | 14.0 / 18.5 / +5.8 | 31.8 / 34.5 / +5.4 |
| `wd=10` | 5.6 / 16.0 / +11.6 | 2.3 / 3.0 / +0.8 | 22.1 / 29.3 / +10.2 | 10.7 / 17.3 / +7.3 | 28.9 / 32.2 / +5.3 |
| `DCT lam=5` | 7.8 / 16.5 / +10.3 | 2.7 / 3.2 / +0.8 | 21.7 / 27.7 / +9.1 | 13.0 / 18.0 / +6.7 | 31.0 / 34.2 / +5.6 |
| `seed-ens-5` | 6.0 / 16.2 / +11.5 | 2.7 / 3.2 / +0.7 | 19.4 / 26.7 / +10.0 | 12.1 / 17.6 / +6.8 | 27.3 / 32.7 / +9.2 |

**Table 5. Linear simulation: DCT λ=5 against the sweeps.** Δ agreement at matched accuracy (points ± SE), and the dropout rate whose agreement equals DCT λ=5's, with the accuracy gap between them.

| Scenario, n | vs dropout sweep | vs ridge sweep | equivalent q | accuracy gap |
|---|---|---|---|---|
| bottom, 100 | -14.2 ± 1.6 | -15.9 ± 1.8 | 0.72 | -0.0 |
| bottom, 300 | -20.4 ± 1.1 | -23.5 ± 1.1 | 0.73 | +0.0 |
| top, 100 | +0.1 ± 0.2 | -0.1 ± 0.2 | 0.73 | +0.3 |
| top, 300 | +0.0 ± 0.1 | +0.0 ± 0.1 | 0.69 | +0.0 |

**Table 6. Protocol check: fixed 30 epochs, no validation split (the original protocol).** Δ accuracy / Δ agreement vs `standard`, points.

| Arm | Adult (3k) | Breast Cancer | High Noise | Small Sample | High Dim |
|---|---|---|---|---|---|
| `dropout=0.5` | +0.6 ± 0.1 / +2.0 ± 0.2 | +0.0 ± 0.1 / +0.4 ± 0.1 | -1.0 ± 0.2 / +1.6 ± 0.4 | +0.1 ± 0.2 / +1.4 ± 0.4 | +0.1 ± 0.2 / +1.2 ± 0.4 |
| `wd=10` | +1.3 ± 0.1 / +7.7 ± 0.3 | +0.3 ± 0.1 / +0.9 ± 0.2 | -1.9 ± 0.2 / +2.3 ± 0.6 | -0.2 ± 0.2 / +2.9 ± 0.5 | +1.1 ± 0.3 / +2.4 ± 0.5 |
| `DCT lam=5` | +0.6 ± 0.1 / +1.9 ± 0.2 | +0.0 ± 0.1 / -0.0 ± 0.1 | +0.5 ± 0.2 / +2.1 ± 0.4 | +0.5 ± 0.2 / +1.3 ± 0.3 | +0.0 ± 0.2 / +0.9 ± 0.4 |
| `DCT lam=15` | +1.0 ± 0.1 / +3.2 ± 0.2 | +0.1 ± 0.1 / +0.3 ± 0.1 | -1.0 ± 0.2 / +1.8 ± 0.6 | +0.2 ± 0.2 / +2.7 ± 0.5 | +0.1 ± 0.3 / +1.8 ± 0.4 |
| `seed-ens-5` | +0.4 ± 0.1 / +1.2 ± 0.2 | -0.0 ± 0.1 / +0.0 ± 0.1 | +1.0 ± 0.2 / +2.4 ± 0.4 | +0.5 ± 0.2 / +1.0 ± 0.2 | +0.9 ± 0.2 / +1.9 ± 0.4 |
