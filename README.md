# Tambua Analysis

[Validation site](https://tkcaccia.github.io/Tambua-Analysis/) · [Tambua software](https://github.com/tkcaccia/Tambua)

This repository contains the Tambua benchmark code and public results. CellPhenotyper produces 23 GigaTIME channels. The [detected-cell viewer](https://tkcaccia.github.io/Tambua-Analysis/detected-cells.html) shows StarDist nuclei, fastPLS 0.4 cell-type predictions, expert annotations and virtual markers for all 47 images passing corrected DAPI and shadow QC. The [120-image diagnostic viewer](https://tkcaccia.github.io/Tambua-Analysis/corrected-monusac.html) retains clearly labelled historical predictions at expert-drawn regions and failed-QC cases.

## Corrected MoNuSAC benchmark

The organizer archives supply 88 selected training ROIs from 40 patients and 32 test ROIs from 19 different patients after a 512-native-pixel minimum-side gate and strict patient separation. Breast, kidney, lung and prostate appear in both splits. An audit found that 116/120 earlier companion TIFFs had red and blue reversed against the scanner SVS despite RGB tags. The complete 120-image run uses scanner-decoded RGB verified pixel-for-pixel, correct source spacing (0.19448–0.2527 µm/pixel), and a 0.25 µm/pixel GigaTIME model grid. Historical wrong-color inference is excluded from the current gallery. See the [image and score provenance](docs/assets/monusac-corrected/DATA_SOURCE.md).

The first DAPI QC inadvertently used heterogeneous source polygons as a nuclear reference; its 48/120 passes and 96.01% conditional region-level test accuracy are retained only as an archive. CPU-only reassessment of the saved images against actual H&E-derived StarDist nuclei, with unchanged QC thresholds, yields **47/120** complete passes: 33 fit, 4 development and 10 organizer-test images. Requantification of their saved 23-channel GigaTIME output over StarDist nuclei and calibrated 3 µm cell proxies yielded 22,436 detected cells, including 15,851 conservatively matched to expert types. A new fastPLS 0.4 model predicts all **5,916 detected test nuclei**; accuracy is **94.26% among 3,975 matched test nuclei**. Both matched test macrophages were missed and there are no matched neutrophils, so four-class balanced accuracy is undefined and the 90% balanced-accuracy goal has not been met. The organizer test was used in retrospective pipeline auditing and should be regarded as development evaluation. See the [detected-cell and QC audit](docs/assets/monusac-corrected/DETECTED_CELL_AUDIT.md) and [machine-readable score](docs/assets/monusac-detected/study-summary.json).

The original archives also contain smaller annotated ROIs. A frozen size audit found 209 training and 101 test images in total; a prospective 256-pixel minimum would yield 147 patient-separated training and 66 test ROIs, compared with 88/32 at 512 pixels, and would greatly increase rare-cell examples. The 93 additional images (59 training, 34 test) have been CPU-prepared from scanner SVS RGB with source pixel spacing and expert XML retained; see the [frozen candidate](docs/assets/monusac-corrected/candidate-expansion-256.json) and [preparation record](docs/assets/monusac-corrected/cpu-expansion-prep.json). They have **not** been run through GigaTIME or QC. This is the main data expansion candidate, not a claimed result.

The [ORION pixel viewer](https://tkcaccia.github.io/Tambua-Analysis/orion-pixel-dapi.html) shows registered H&E, raw measured Hoechst pixels, virtual DAPI pixels and an overlay for one 2048-pixel field from each of ten QC-passing reserved patients. Native-value pixel Spearman correlation has an equal-patient median of **0.806** (patient-bootstrap 95% interval **0.752–0.849**); virtual DAPI remains broader and more saturated than measured nuclei in some fields. The [separate marker comparison](https://tkcaccia.github.io/Tambua-Analysis/#orion-visual) paints released **per-cell fluorescence values onto cell masks** and must not be mistaken for measured protein images. Those ten fields do not cover all 40 ORION patients or validate other virtual proteins. ORION provides no expert MoNuSAC cell-type labels.

## Comparison design

Use the official MoNuSAC epithelial, lymphocyte, macrophage and neutrophil ontology. Freeze patient-disjoint training, validation and test groups. Compare the following on identical held-out H&E images and expert reference instances:

| Method | Input | Model source | Endpoint |
| --- | --- | --- | --- |
| Tambua | CellPhenotyper instances and calibrated GigaTIME scores | [Tambua](https://github.com/tkcaccia/Tambua), [fastPLS 0.4](https://github.com/tkcaccia/fastPLS) | End-to-end cell detection and type; separately classify on expert instances to isolate the marker/classifier step |
| CellViT++ | H&E | [official CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) | Native instance detection and type, with a prospectively fixed mapping to MoNuSAC classes |
| HoVer-Net | H&E | [official HoVer-Net](https://github.com/vqdang/hover_net) | Native instance detection and type using its MoNuSAC checkpoint |

CellViT++'s publicly described PanNuke taxonomy is **not** interchangeable with the four MoNuSAC types. A MoNuSAC-trained cell-type head or a documented external mapping must be fixed without looking at the held-out test set. HoVer-Net provides a MoNuSAC-specific pretrained checkpoint. Do not compare Tambua on expert-provided masks to competitors' end-to-end outputs as if they were equivalent. Report the expert-mask classifier analysis and end-to-end analysis separately. Dayao and colleagues' [H&E cell-type prediction using CODEX-derived labels](https://doi.org/10.1016/j.crmeth.2025.101204) is a close conceptual predecessor; its kidney-cohort scores cannot be directly compared with MoNuSAC. The [literature and benchmark note](docs/LITERATURE_BENCHMARK.md) records appropriate comparator and dataset roles for the supplied papers.

**Training-data provenance:** The pinned GigaTIME CNN was trained on 21 in-house paired lung-adenocarcinoma slides, according to its original paper. The current ORION colorectal field is a distinct technical demonstration, and MoNuSAC is now a completed expert-annotated transfer cohort; neither source is listed as GigaTIME translator training data. TCGA application in the GigaTIME paper is inference, not translator fitting. The original paper does not publish all training slide IDs, so exact patient-level disjointness for this MoNuSAC testing remains to be verified if that manifest becomes available. See the [source-by-source audit](docs/DATA_PROVENANCE.md). Test patients must also be excluded from Tambua fastPLS fitting and model selection.

For end-to-end evaluation, match predicted and expert instances one-to-one at mask IoU > 0.5, count unmatched instances, and report detection F1 and per-type F1 in addition to classification accuracy on matched cells. Report macro F1 and balanced accuracy only when every reference type is present. Aggregate by patient and add patient-bootstrap uncertainty when there are enough independent patients. Record physical resolution, model checkpoint, inference time, and hardware. Never treat cells from one patient as independent biological replicates.

## Physical scale and DAPI quality control

The current bounded ORION CRC33_01 source window is a registered same-section H&E/whole-cell mask at 0.325 µm/pixel. A strict GigaTIME grid at 0.25 µm/pixel should be 2663 × 2663 for a 2048 × 2048 source window. The [GigaTIME author's preprocessing clarification](https://github.com/prov-gigatime/GigaTIME/issues/4) states that 128 µm fields were resized to 512 pixels and then processed in 256-pixel windows, giving the 0.25 µm/pixel model grid. Record the exact transform and confirm the DAPI virtual channel spatially overlaps H&E nuclear morphology. A numerical H&E hematoxylin proxy can support, but cannot replace, visual review or a true expert nuclear mask. ORION's whole-cell mask should be called `cell`, not `nuclei`.

The earlier ORION inference produced four preview channels and all 23 per-cell virtual scores for 4,104 complete cells. Those archived images retain visible shading and are not on the current public page. Its high-signal overlap enrichment within the published cell regions was 2.068; this is a morphology proxy, not expert-mask Dice. See the [case record](docs/RESULTS.md) for the audit trail.

When a scale-valid GigaTIME result exists, run the spatial QC on the **same physical field of view**:

```bash
python -m analysis.qc_dapi --he he.ome.tif --cell-mask expert_cells.ome.tif \
  --dapi gigatime_output/jpg_image/DAPI.jpg \
  --manifest gigatime_output/gigatime_jpg_manifest.json \
  --output-dir results/dapi_qc
```

The tool refuses a source/target pixel-spacing or image-shape mismatch, saves an H&E/DAPI overlay, and reports enrichment of high virtual DAPI over an H&E hematoxylin nuclear **proxy**. That proxy is not an expert nuclear annotation; the resulting overlay still needs visual review.

For a frozen end-to-end comparison, create one CSV manifest per method with columns `sample_id,patient_id,reference_mask,reference_labels,prediction_mask,prediction_labels`. Each labels CSV has `label_id,cell_type`. Score only the test cases, with all methods using the same cases and four-class ontology:

```bash
python -m analysis.score_instances --manifest test_tambua.csv --method Tambua \
  --track end_to_end --output results/tambua.json
python -m analysis.score_instances --manifest test_cellvit.csv --method CellViT++ \
  --track end_to_end --output results/cellvit.json
python -m analysis.score_instances --manifest test_hovernet.csv --method HoVer-Net \
  --track end_to_end --output results/hovernet.json
```

The scorer rejects missing or extra typed instances and incompatible mask shapes. It reports per-sample, per-patient and pooled detection/type scores. Use `--track expert_mask_classification` only when prediction and reference instance masks are identical; this is a separate conditional-classification experiment.

## Archived exploratory MoNuSAC transfer result

The superseded evaluation used 27 lung training ROIs (9 patients) and 10 prostate test ROIs (6 distinct patients), with 5,243 and 3,584 expert-typed nuclei respectively. The original organizer image/XML pixel spacing was verified, then GigaTIME CNN inference was run at 0.25 µm/pixel. CellPhenotyper quantified 21 virtual channels on expert nuclear masks and 21 on 3 µm nucleus-anchored cell proxies. The latter are **not expert cell boundaries**. These images are retained in the repository audit and are no longer on the public page.

The earlier lung-only fastPLS classifier was tested on prostate, a tissue absent from training, and its virtual-marker input had the same RGB error. Its archived 89.0% pooled accuracy is invalidated and also hides three single-class images. On the other seven images, historical equal-image mean recall was 46.9% and macro F1 42.7%. The [class-coverage audit](docs/assets/monusac-transfer/class-coverage.json), [score report](docs/assets/monusac-transfer/fastpls-report.json), [per-image counts](docs/assets/monusac-transfer/results.json) and [fitted model](docs/assets/monusac-transfer/fastpls-lung-to-prostate.rds) are preserved for audit only.

To reproduce the CPU model fit and exact per-cell predictions from the saved, completed GigaTIME/CellPhenotyper bundle, install Tambua's pinned fastPLS 0.4 environment and run:

```bash
TAMBUA_R_LIB=/path/to/fastPLS-0.4/R-library \
  /path/to/Tambua/.venv/bin/python analysis/score_monusac_transfer.py \
  --source /path/to/completed-MoNuSAC-bundle \
  --output /path/to/new-empty-score-directory \
  --tambua /path/to/Tambua --historical-unvalidated
/path/to/Tambua/.venv/bin/python analysis/build_monusac_gallery.py \
  --results /path/to/completed-MoNuSAC-bundle \
  --scores /path/to/new-empty-score-directory --site docs
```

The first script never starts GPU inference; the completed source bundle must contain the frozen cohort, expert masks and labels, all 37 quantification outputs, and the one-shot bundle completion marker. The explicit flag labels this reproduction as unvalidated; new scoring requires a passing native DAPI/StarDist QC report and a validated CellPhenotyper quantification summary for every image. The original dataset acquisition and expert-mask preparation code live in [Tambua](https://github.com/tkcaccia/Tambua/tree/main/scripts). The published site contains derived small previews, not raw source archives. MoNuSAC imagery and labels retain CC BY-NC-SA 4.0 terms; the site's MIT license applies only to original code.

The ORION CRC33_01 study is a one-patient technical record with 4,104 computationally segmented cells and virtual-versus-measured marker associations. It has no expert cell-type truth. Its old image gallery is no longer on the public page. CellViT++ and HoVer-Net have **no verified same-cohort prediction** yet, so there is no head-to-head accuracy ranking.

## Sources

- [GigaTIME official code](https://github.com/prov-gigatime/GigaTIME) and [model card](https://huggingface.co/prov-gigatime/GigaTIME).
- [CellViT++ official code](https://github.com/TIO-IKIM/CellViT-plus-plus).
- [HoVer-Net official code](https://github.com/vqdang/hover_net).
- [ORION data release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md).
- [MoNuSAC challenge](https://monusac-2020.grand-challenge.org/Data/).

The code and site are MIT licensed. Third-party data and checkpoints retain their own terms.
