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

We then re-quantified the **47 corrected StarDist-reference QC-passing images**
from their saved 23-channel GigaTIME model-grid arrays, without new GPU
inference. The current CellPhenotyper quantifier recorded scanner-RGB input,
physical spacing, the corrected-QC report hash, nuclear means and calibrated
3 µm nearest-nucleus cell-proxy means for every detected nucleus. This yielded
22,436 detected-cell feature rows; 15,851 had conservative expert-type matches.
The 33 fit, 4 development and 10 organizer-test images are patient-disjoint.

A CPU-only fastPLS **0.4** SIMPLS-LDA fit used all 23 GigaTIME channels in each
compartment (46 predictors) from 10,049 matched fit cells. It did not use the
four development images. On ten organizer-test images it assigned a cell type
to **all 5,916 detected nuclei**. Among the **3,975 nuclei conservatively
matched to expert labels**, accuracy is **94.26%**: epithelial recall
2305/2447 (94.20%), lymphocyte recall 1442/1526 (94.50%), macrophage recall
0/2, with **no matched test neutrophils**. Therefore, four-class balanced
accuracy and macro F1 are undefined. Unmatched detections are not counted as
correct or incorrect because MoNuSAC annotation coverage varies. This is
classification conditional on detection and matching, not detection precision
or an end-to-end four-class score. The organizer test was used in retrospective
pipeline audit, so it is development evaluation rather than an independent
final holdout. The target of >90% four-class balanced accuracy remains unmet.

The prior 48-image detected-cell quantification above is preserved as an audit
of the *original* QC selection. Its features and counts are not mixed with
this revised 47-image cohort. The earlier 96.01% expert-region result remains
historical and **must not** be read as a corrected StarDist-cell score.

Reproduction code: [`score_monusac_stardist_detection.py`](../../../analysis/score_monusac_stardist_detection.py),
[`match_monusac_expert_to_detected.py`](../../../analysis/match_monusac_expert_to_detected.py),
[`quantify_detected_monusac_cells.py`](../../../analysis/quantify_detected_monusac_cells.py),
[`run_detected_quantification_cpu.py`](../../../analysis/run_detected_quantification_cpu.py),
[`score_monusac_detected_cells.py`](../../../analysis/score_monusac_detected_cells.py),
[`build_detected_cell_gallery.py`](../../../analysis/build_detected_cell_gallery.py)
[`render_all_monusac_markers.py`](../../../analysis/render_all_monusac_markers.py)
and [`recheck_monusac_nuclear_qc_with_stardist.py`](../../../analysis/recheck_monusac_nuclear_qc_with_stardist.py).
The [CPU quantification record](detected-cell-quantification-summary.json) and
[revised 47-image quantification record](../monusac-detected/quantification-summary.json),
[machine-readable fastPLS score](../monusac-detected/study-summary.json) and
[frozen development fold](candidate-development-fold-256.json) preserve counts
and split identities. [The image viewer](../../detected-cells.html) shows all
47 cases. All 47 images were visually inspected in contact sheets of H&E,
virtual DAPI, StarDist nuclei, predicted-cell overlay and source expert
annotations. No obvious border shadow was seen at that scale; some DAPI
signals remain broad or saturated, and this visual check does not prove that
every cell is anatomically correct. No UCT HPC GPU job was run for this audit.

The viewer additionally exposes all **23 saved spatial GigaTIME channels** for
each of those 47 images. Their [manifest](../monusac-all-markers/manifest.json)
records native uint16 quantiles, physical pixel spacing and per-case quality
flags. WebP previews are contrast-stretched independently for anatomy review;
the saved pixel values, not the previews, supplied cell quantification. The
all-channel block-seam QC passed in all 47 cases. The marker-distribution check
returned `review` in four cases because selected channels had a sampled
5th–95th percentile range below 0.005; the viewer displays the affected
channel warnings. This flag is not proof of a border shadow, and passing DAPI
shadow QC is not proof that every virtual marker is biologically valid.
