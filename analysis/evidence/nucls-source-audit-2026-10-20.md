# NuCLS breast-cell candidate: source and eligibility audit

The [NuCLS corrected single-rater release](https://sites.google.com/view/nucls/single-rater) reports 1,744 breast fields, 59,485 annotated nuclei, and 19,680 outlined boundaries. Its [format specification](https://sites.google.com/view/nucls/data-format) states 0.2 µm/pixel, distinguishes `polyline` from `rectangle` in the per-field CSV, and warns that the convenient raster masks do **not** distinguish these annotation types. The corrected annotations were reviewed under pathologist supervision; this is not a claim that every outline is an expert-drawn segmentation.

For a non-GPU acquisition audit, the three Parquet shards of the [MedOtter NuCLS mirror](https://huggingface.co/datasets/MedOtter/NuCLS) were downloaded to Chiamaka under `/home/chiamaka/Tambua-nucls-corrected-source-2026-10-20`. Their SHA-256 digests, in shard order, are:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `train-00000-of-00003.parquet` | 127,841,602 | `9f59987fe978f8c8d9f1adf4fc87549d3d15de02a8241f4d403b46e2cfa52b46` |
| `train-00001-of-00003.parquet` | 153,370,157 | `abdfd25673abdfc03cbffd9b3d74eb79c5e08819baf9d259d90fc2b81261c3da` |
| `train-00002-of-00003.parquet` | 119,610,672 | `6bbc9f31954dc8f91f1d149dd3aa100a17596d61919cb2930f5c8e17e123a026` |

The mirror has 1,744 unique fields and 124 TCGA patients. Its aggregate metadata reproduce the official 59,485 nuclei and 19,680 polylines; another 39,694 annotations are rectangles. In its `fold_1`, 1,481 fields from 102 patients are training and 263 fields from 22 other patients are test, with no patient overlap. The outlined superclass counts are 8,585 stromal tumor-infiltrating lymphocytes (`sTIL`), 5,871 tumor nuclei (`tumor_any`), 3,150 other stromal nuclei (`nonTIL_stromal`), 1,982 ambiguous, and 92 other nuclei. Ambiguous and other labels need a preregistered mapping or exclusion.

An *exploratory* requirement of at least 20 outlined canonical nuclei, at least two of those three classes, and at least three outlines per represented class yields 86 fields: 69 from 39 training patients and 17 from 12 test patients. The training candidate subset is imbalanced (1,362 `sTIL`, 569 `nonTIL_stromal`, 110 `tumor_any`), so it would be unsound to train only on those 69 fields. A potential future breast experiment would use the full patient-separated training split and assess suitably multiclass test fields. It must reconstruct pixel masks from CSV `polyline` coordinates, never treat rectangular boxes in the supplied PNG masks as nuclear boundaries, and resample the calibrated 0.2-µm/pixel H&E to GigaTIME's 0.25-µm/pixel input contract without claiming new detail.

**Status:** acquired and audited as a candidate only. No NuCLS GigaTIME inference, CellPhenotyper quantification, fastPLS training, benchmark score, or public result has been produced. The fields are approximately 300–400 pixels wide, so the limited spatial context and breast-only tissue coverage may restrict this extension. It cannot replace the four-tissue MoNuSAC evaluation. UCT HPC GPUs were not used.

## Fine-label follow-up and frozen source cohort

The superclass summary above hides a crucial distinction: `nonTIL_stromal`
contains outlined macrophages. We re-read the CSV `main_classification` and
froze a separate, label-enriched **192-field** cohort (138 training, 54 test)
using patient-separated mirror `fold_1`. The selection includes every field
with an outlined macrophage plus deterministic, class-diverse supplements.
It is not a prevalence-representative test set. The preparation retains only
intact expert-drawn polylines, excludes boxes and overlapping/ambiguous
polygons, and saves native RGB OME-TIFF with 0.2 µm/pixel metadata. An
independent validator checked all 192 RGB images and target masks, including
pixel-level nonoverlap of retained and excluded annotations.

| Split | Images | Patients | Epithelial | Lymphocyte | Macrophage | Plasma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 138 | 59 | 217 | 1,351 | 359 | 276 |
| Test | 54 | 12 | 94 | 248 | 108 | 105 |

These are **prepared expert target counts**, not GigaTIME, StarDist or
classifier results. The local source audit and visual contact sheets are in
`analysis/evidence/nucls-fine-*`. No outlined test neutrophils are available;
NuCLS can assess the four classes in the table but cannot validate neutrophil
prediction. Any future virtual-marker inference must physically resample
0.2 µm/pixel source images to the pinned 0.25 µm/pixel GigaTIME grid, then
separately pass nuclear alignment and shadow QC before classification.
