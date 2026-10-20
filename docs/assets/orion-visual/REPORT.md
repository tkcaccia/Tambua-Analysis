# ORION virtual-marker association audit — user-stopped partial cohort

The user stopped the approved Chiamaka bundle after approximately 20 cases. **Twenty-five patient cases had completed their case checkpoints; CRC26 was in progress and is excluded.** The supervisor, owned child processes and GPU compute context were verified stopped. This is a partial analysis of the planned 40-patient study, not its completed endpoint.

One registered 2048 × 2048 H&E/cell-mask window was sampled from each completed ORION colorectal patient (CRC01–CRC25). All source images had 0.325 µm/pixel metadata and were resampled to GigaTIME's 0.25 µm/pixel input grid. A mask-only sampling error initially chose nearly blank H&E for CRC22. The candidate selection now checks H&E tissue and nuclear texture as well as cell-mask occupancy; four affected windows were replaced before GPU inference. The final 25 windows have at least 66% stained H&E pixels and at least 1,154 H&E StarDist nuclei matched to published ORION cell IDs.

The audit fixed a CellPhenotyper compartment error: cytoplasm is now the published cell region with its same-ID StarDist nucleus removed. On CRC01, nucleus and cytoplasm areas and all checked marker sums exactly reconstructed the whole-cell values for 3,882 matched cells. The analysis uses 102,664 common cells across 25 patients (median 4,062 per patient). Comparisons use the same cells for every virtual-marker summary and calculate Spearman rho within each patient before taking an equal-patient median. No cells from different patients are pooled into one correlation.

The registered design has ten marker/dye pairs: six exact protein names (CD68, CD4, CD20, PD-L1, PD-1, Ki67), three related targets (CD8/CD8a, CD3/CD3e, CK/Pan-CK), and DAPI/Hoechst as a dye control. It does not substitute biologically different proteins. Eleven summaries were tested: whole-cell, cytoplasmic and nuclear means/maxima; two nucleus–cytoplasm contrasts; and three local-background contrasts. Six completed development patients selected one summary per pair. The remaining 19 completed patients were reserved for this comparison. With only six development patients, selection is unstable; treat this as exploratory.

| Virtual → measured | Development-selected summary | Baseline median rho, 19 patients | Selected median rho | Median paired change [95% patient bootstrap CI] |
|---|---|---:|---:|---:|
| DAPI → Hoechst | Whole-cell local contrast | +0.376 | +0.358 | −0.002 [−0.071, +0.021] |
| CD68 → CD68 | Nucleus minus cytoplasm | −0.245 | −0.096 | +0.172 [+0.146, +0.229] |
| CD4 → CD4 | Cytoplasm maximum | +0.168 | +0.177 | +0.002 [−0.003, +0.025] |
| CD8 → CD8a | Nucleus mean | +0.061 | +0.063 | −0.004 [−0.014, −0.001] |
| CD20 → CD20 | Cytoplasm mean | +0.141 | +0.129 | +0.002 [+0.000, +0.004] |
| PD-L1 → PD-L1 | Nucleus mean | −0.010 | +0.001 | +0.005 [−0.000, +0.011] |
| CD3 → CD3e | Cytoplasm maximum | +0.215 | +0.132 | −0.013 [−0.033, +0.006] |
| PD-1 → PD-1 | Cytoplasm maximum | +0.112 | +0.109 | −0.002 [−0.010, +0.009] |
| Ki67 → Ki67 | Cytoplasm mean | +0.079 | +0.097 | −0.001 [−0.027, +0.008] |
| CK → Pan-CK | Whole-cell maximum | +0.249 | +0.303 | +0.024 [+0.005, +0.048] |

The baseline is whole-cell virtual-marker mean. Median paired change is the median of within-patient selected-minus-baseline rho; it need not equal the difference between the two marginal medians. The bootstrap resamples patients, not cells. Across all 190 marker–patient pairs in the reserved subset, switching to cytoplasmic mean changed rho by a median of −0.0001, to nuclear mean by −0.0039, and to local whole-cell contrast by −0.0453. There is **no general compartment or shading correction that improves the panel**. CK's modest gain is for a related marker family, not an exact protein pair. CD68 remains negatively correlated even after its numerical gain and is unsuitable as a positive protein prediction result.

All 25 persisted DAPI images passed the recorded seam-continuity check; only 12/25 passed the independent H&E StarDist–virtual-DAPI nuclear QC (10/19 reserved patients). On those 10 reserved QC-passing cases, CK changed from median rho +0.189 to +0.268 (median paired change +0.035), while CD68 remained negative (−0.227 to −0.099). Passing DAPI QC does **not** validate the other proteins. The failed-QC cases are descriptive associations only and must not feed a validated Tambua classifier.

Important limits: ORION's published measured table gives **whole-cell fluorescence**, so this study compares virtual nuclear or cytoplasmic summaries with measured *whole-cell* abundance. It does not establish true measured-nucleus versus predicted-nucleus or measured-cytoplasm versus predicted-cytoplasm correlation; that requires registered raw IF compartment quantification. Each patient contributes one tissue window, not a full WSI. No UCT HPC job was run.

Reproduction files: `per_patient_spearman.csv` has every per-patient, per-marker, per-method rho and QC flag; `summary.json` has selected methods, paired estimates and patient records. `WINDOW_SELECTION_QC.json`, `ANALYSIS_PLAN_FROZEN.json`, `ANALYSIS_PLAN_AMENDMENT_QC.json` and `USER_STOPPED_BUNDLE.json` preserve sampling, method and stop provenance. The code is in `analysis/analyze_orion_proteins.py`, `analysis/acquire_orion_window.py` and the CellPhenotyper GigaTIME quantifiers.

## Across-patient pooled rank association

As an additional fixed-baseline analysis, [`cross-patient-rank.json`](cross-patient-rank.json) combines the same 10 reserved patients who passed both technical gates. For each of the ten prespecified marker pairs, it ranks virtual whole-cell means and published measured whole-cell values **within each patient**, pools the five rank moments with **equal patient weight**, and bootstraps whole patients 10,000 times for uncertainty. This uses 39,554 matched cells but does not give larger patients more weight. It does not fit or select a marker-specific correction on the reserved patients.

| Virtual → measured | Equal-patient pooled rank correlation | 95% patient-bootstrap interval |
| --- | ---: | ---: |
| DAPI → Hoechst | +0.444 | +0.366 to +0.519 |
| CD68 → CD68 | −0.225 | −0.345 to −0.106 |
| CD4 → CD4 | +0.168 | +0.042 to +0.284 |
| CD8 → CD8a | +0.093 | +0.015 to +0.184 |
| CD20 → CD20 | +0.213 | +0.062 to +0.374 |
| PD-L1 → PD-L1 | +0.019 | −0.088 to +0.112 |
| CD3 → CD3e | +0.223 | +0.108 to +0.342 |
| PD-1 → PD-1 | +0.121 | +0.030 to +0.213 |
| Ki67 → Ki67 | +0.173 | +0.068 to +0.278 |
| CK → Pan-CK | +0.228 | +0.083 to +0.378 |

This is a cross-patient *cell-abundance* association, not a correlation between raw measured and virtual protein image pixels. The present ORION GigaTIME stores retained DAPI but not the other spatial channels, so raw spatial protein comparisons still require a new, separately authorized inference and independent image-by-image QC. The weak or negative associations, especially CD68, do not support a claim that the current virtual proteins are biologically validated.
