# Corrected scanner-RGB MoNuSAC cell-typing audit

This is the 120-region frozen MoNuSAC cohort: 88 training images and 32
patient-separated test images from breast, kidney, lung, and prostate. Every
test tissue is represented in training. Image and expert-region rights remain
with the MoNuSAC contributors under CC BY-NC-SA 4.0.

The original benchmark used companion TIFFs whose red and blue channels were
exactly reversed relative to their scanner SVS images in 116/120 cases,
despite RGB TIFF tags. The four SVS-only cases were already RGB. All 120
corrected prepared images were independently verified pixel-for-pixel against
OpenSlide's RGB decode of their scanner SVS; labels, source pixel spacing, and
patient split were unchanged. The GigaTIME PyVips reader was also checked
against the corrected TIFF for `TCGA-2Z-A9JG-01Z-00-DX1_6` and produced
identical RGB bytes, not an R/B swap. See
[`corrected_120_input_audit.json`](../dapi-2z-a9jg/corrected_120_input_audit.json).

The pinned GigaTIME CNN revision is
`a6556030bdb8071857559e12b921620aa1bbac5b`. Source resolution was read
from each OME image and resampled to the model's 0.25 µm/pixel grid. Model
tiles were 256 pixels with a 128-pixel stride and raised-cosine overlap
weighting. All 23 uncalibrated virtual-marker channels were persisted as
native uint16 Zarr and quantified at the MoNuSAC regions and their 3 µm
nucleus-anchored cell proxies. Display JPEGs are not the quantification input.
The exact source for the classifier is
[`tkcaccia/fastPLS` commit `265108b`](https://github.com/tkcaccia/fastPLS/commit/265108b6aa250c2534f98ec89037bb5cd001dca0),
version 0.4, with a SIMPLS-LDA CPU fit on 46 features (23 nuclear plus 23 cell
proxy summaries). No UCT HPC GPU jobs were run.

The native virtual-DAPI nuclear QC passed 50 images and failed 70. The passing
subset contains 40 training and 10 test images from 29 and 10 patients,
respectively, across all four tissues. Its patient-separated test has 4,082
source-annotated regions. Accuracy is 95.71%, and equal-patient accuracy is
95.50%. Epithelial recall is 96.55%; lymphocyte recall is 94.54%; all three
macrophages were missed. There are no neutrophils in this QC-selected test
subset, so a four-class balanced accuracy or macro F1 is undefined. See
[`qc-pass-report.json`](qc-pass-report.json).

The all-image score in the gallery is **diagnostic only** because it includes
70 GigaTIME-QC failures. Across 10,776 held-out regions it has 96.65% accuracy,
48.36% four-class balanced accuracy and 48.38% macro F1. Both rare classes
have zero recall: 0/46 macrophages and 0/4 neutrophils. High overall accuracy
therefore overstates the biological breadth of this classifier. See
[`exploratory-all-report.json`](exploratory-all-report.json).

These are **conditional cell-type predictions at source MoNuSAC regions**, not
end-to-end cell detection using StarDist. The source annotations themselves
contain implausibly large regions marked macrophage, including nine in
`TCGA-2Z-A9JG-01Z-00-DX1_6`. The original XML contains those labels and
polygons; the overlay is not a StarDist output. Across the 120 images,
41/257 source macrophage regions exceed a 5,000-pixel *manual-review flag*.
No reference label was silently changed or excluded. See
[`annotation_geometry_audit.json`](annotation_geometry_audit.json). Source
annotation review and a detector-to-typing evaluation are still needed.

All cases and completion/cleanup evidence are retained on Chiamaka at
`/mnt/sata_ssd/Tambua-monusac-rgb-corrected-2026-10-20-v1`. The 120
inferences completed with individually verified process-group and GPU-context
cleanup. The evaluator and gallery code are in this repository's `analysis/`
directory. The public viewer is [`corrected-monusac.html`](../../corrected-monusac.html).
