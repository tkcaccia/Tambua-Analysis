# Detected-cell quantification and nuclear-reference correction

The original MoNuSAC XML polygon geometry differs by class. In the first
120-image preparation, the median source macrophage area was 2,080 native
pixels versus 279 for lymphocytes; 41 of 257 macrophage polygons exceeded
5,000 pixels. Visual audit shows that some macrophage polygons surround a
whole cell while many lymphocyte polygons follow the nucleus. The source class
is still useful, but calling every source polygon a nuclear mask distorted
GigaTIME DAPI QC and compartment quantification.

We ran the released `2D_versatile_he` StarDist model on scanner-decoded RGB
using CellPhenotyper's 1st/99.8th percentile channel normalization, probability
threshold 0.48 and NMS threshold 0.30. This CPU-only detection covered all
213 prepared MoNuSAC images (115 fit, 32 development, 66 organizer test).
Independent expert classes were assigned one-to-one to detected nuclei at
IoU ≥ 0.5. A macrophage whole-cell polygon with lower nuclear IoU was accepted
only if exactly one unassigned detected nucleus had at least 75% of its pixels
inside that polygon; ambiguous polygons were withheld. Across the 66 organizer
test images, this maps 149 of 209 source macrophage polygons to a nucleus,
versus 27 under nuclear IoU alone. This is *label-matching coverage*, not
cell-type prediction accuracy or proof that unmatched detections are false
positives: some source images annotate only a subset of visible cells.

The saved 120-image GigaTIME predictions were reassessed against the actual
StarDist masks by the original CellPhenotyper nuclear-QC code, with **unchanged**
thresholds. Complete DAPI alignment, border-shadow and spatial-shadow passes
number 33 fit, 4 development and 10 organizer-test images, 47/120 total.
The original source-polygon reference yielded 32, 6 and 10, respectively.
For `TCGA-2Z-A9JG-01Z-00-DX1_6`, AUC rose from 0.391 against source polygons
to 0.799 against StarDist nuclei; it **still fails** the unchanged nuclear
boundary-contrast gate (0.0436 versus required 0.05). It is not admitted by
this reassessment. [QC records](stardist-nuclear-qc-summary.json) contain all
120 paired outcomes; reports and preview images remain on Chiamaka.

Separately, all 48 images that passed the *original* four-field QC were
postprocessed without new GigaTIME inference. The same saved 23-channel
model-grid Zarr was quantified over each detected StarDist nucleus and a
physically calibrated 3 µm nearest-nucleus cell halo. This gives 21,667
detected-cell feature rows, 15,297 with a conservative expert-type match;
unmatched cells remain available for future inference. Every case records
scanner RGB provenance, pixel spacing, input hashes, marker count and CPU
completion. These 48 images are **not** a final corrected-QC cohort because
the two reference definitions select different cases.

The original strict-QC fit has only 5 matched macrophages and 6 neutrophils;
its development set has no matched macrophages and its test set has only 2
matched macrophages and no neutrophils. The corrected StarDist-QC selection
also lacks macrophages in development and neutrophils in the organizer test.
Consequently, no credible four-class balanced accuracy or end-to-end
cell-identification score can be reported from the existing strict-QC subset.
The earlier 96.01% expert-region result is historical and **must not** be read
as a corrected StarDist-cell score.

Reproduction code: [`score_monusac_stardist_detection.py`](../../../analysis/score_monusac_stardist_detection.py),
[`match_monusac_expert_to_detected.py`](../../../analysis/match_monusac_expert_to_detected.py),
[`quantify_detected_monusac_cells.py`](../../../analysis/quantify_detected_monusac_cells.py),
[`run_detected_quantification_cpu.py`](../../../analysis/run_detected_quantification_cpu.py)
and [`recheck_monusac_nuclear_qc_with_stardist.py`](../../../analysis/recheck_monusac_nuclear_qc_with_stardist.py).
The [CPU quantification record](detected-cell-quantification-summary.json) and
[frozen development fold](candidate-development-fold-256.json) preserve counts
and split identities. No UCT HPC GPU job was run for this audit.
