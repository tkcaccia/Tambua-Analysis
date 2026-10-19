# ORION StarDist threshold audit (CPU only)

The 25 completed ORION colorectal windows from the stopped GigaTIME study were reused without any new GPU inference. Six patients (CRC01–CRC06) were used for parameter exploration; the other 19 (CRC07–CRC25) were reserved for the comparison. Each window has a 2048 × 2048 H&E image at 0.325 µm/pixel, a published computational cell mask, and a persisted GigaTIME virtual DAPI image. StarDist `2D_versatile_he` was run on H&E resampled to 0.25 µm/pixel. The same majority-overlap rule mapped each predicted nucleus to a published cell ID. The DAPI QC code was rerun unchanged on each new nuclear mask.

| StarDist probability / NMS | Reserved patients | Mean cells with matched nucleus | Mean virtual-DAPI AUC | QC pass |
|---|---:|---:|---:|---:|
| Built-in default, 0.692478 / 0.30 | 19 | 51.6% | 0.6896 | 10/19 |
| CellPhenotyper configuration, 0.52 / 0.28 | 19 | 62.2% | 0.6869 | 10/19 |
| Inclusive exploratory setting, 0.40 / 0.30 | 19 | 65.5% | 0.6856 | 10/19 |

Compared patient by patient with the built-in default, the CellPhenotyper setting increased matched-cell coverage in **all 19** reserved patients by a mean **10.6 percentage points** (95% patient-bootstrap interval 10.0–11.2). Mean AUC changed by **−0.0028** (95% interval −0.0043 to −0.0013). The more inclusive 0.40 setting gained 13.9 points of coverage (95% interval 13.2–14.7) but lowered AUC by 0.0040. The development patients showed the same direction. A stricter 0.80 setting raised development AUC from 0.700 to 0.721 yet cut mean matched-cell coverage from 46.8% to 23.3%; selecting settings only by AUC would preferentially discard difficult nuclei.

The ORION compartment-preparation script had silently used the StarDist model's built-in 0.692478/0.30 thresholds, whereas the CellPhenotyper pipeline already configured 0.52/0.28. Its defaults now match the pipeline and explicitly record both thresholds in provenance. This is a **source-code correction for future runs**; the prior 25-patient marker tables and protein correlations were produced with the earlier nuclear masks and have not been retroactively recomputed. The production StarDist process also applies ROI-aware normalization and an area filter, so this experiment isolates the threshold component rather than reproducing every production step.

The coverage denominator is the published ORION **computational** cell mask, not expert-drawn nuclei. It is a useful omission indicator but not a measured nuclear-recall estimate; added detections still need expert review for false positives. The AUC compares virtual DAPI with nuclei inferred independently from H&E, not with measured DAPI/Hoechst. Threshold tuning cannot repair diffuse or misplaced GigaTIME signal. The unchanged QC pass count leaves the failing windows blocked from validated downstream claims.

The example [H&E comparison](comparison-crc01.png) shows the same CRC01 tissue crop with outlines at the built-in default, pipeline setting and inclusive setting. Red contours are matched StarDist nuclei. The complete paired values are in [scores.csv](scores.csv) and [summary.json](summary.json); reproducible CPU-only scripts are `analysis/audit_stardist_thresholds.py` and `analysis/score_stardist_thresholds.py`.
