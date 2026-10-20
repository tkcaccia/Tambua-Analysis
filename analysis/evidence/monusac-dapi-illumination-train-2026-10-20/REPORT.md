# Train-only virtual-DAPI illumination diagnostic

This CPU-only experiment read 37 checkpointed **training** cases from the live,
source-frozen MoNuSAC expansion at its 59-case snapshot. No test case was
loaded. The input snapshot includes native uint16 GigaTIME DAPI and the
independent H&E-derived StarDist mask, with source hashes recorded in
`SNAPSHOT.json`. The scoring code used the unchanged CellPhenotyper nuclear
alignment and spatial-shadow gate. Recomputed raw status and AUC matched all
37 saved native QC reports exactly.

| Label-free transform | Cases passing all gates | Median nuclear AUC |
| --- | ---: | ---: |
| Saved raw DAPI | **4 / 37** | 0.685 |
| Divide by Gaussian background, σ = 48 / 96 model px | 0 / 1 | 0.652 / 0.662 |
| Subtract Gaussian background, σ = 48 / 96 model px | 2 / 4 | 0.679 / 0.677 |
| Subtract blurred logit, σ = 48 / 96 model px | 0 / 0 | 0.682 / 0.688 |

The train-only selection rule chose the **raw** saved image. No illumination
correction was deployed, no QC threshold was relaxed, and no test score was
computed from these candidates. Apparent marker staining cannot be made
credible simply by display stretching DAPI. Subsequent work needs a paired
measured-positive control and protein-specific spatial checks. The full
per-image scores and fixed candidate definitions are in `REPORT.json`.

UCT HPC GPUs were not used. This diagnostic did not launch GigaTIME or modify
the active Chiamaka CPU bundle.
