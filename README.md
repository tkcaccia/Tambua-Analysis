# Tambua Analysis

[Validation site](https://tkcaccia.github.io/Tambua-Analysis/) · [Tambua software](https://github.com/tkcaccia/Tambua)

This repository holds the public image gallery and reproducible scoring code for Tambua cell typing. Tambua uses CellPhenotyper GigaTIME virtual-marker scores and [GitHub fastPLS **0.4**](https://github.com/tkcaccia/fastPLS/tree/265108b6aa250c2534f98ec89037bb5cd001dca0). The primary MoNuSAC result trains and tests across the same four tissues with separate patients. It is **conditional cell classification** at expert-defined nuclei, not an end-to-end detection result or a matched comparison against CellViT++ and HoVer-Net.

## Matched-tissue MoNuSAC result

The [public gallery](https://tkcaccia.github.io/Tambua-Analysis/#matched-cohort) shows all **32 held-out ROIs** with original H&E, expert cell types, predicted types, errors, and virtual DAPI/CK/CD3/CD68. The original organizer archives provided **88 training ROIs from 40 patients** and **32 test ROIs from 19 different patients** after a 512-native-pixel minimum-side gate and exclusion of any training patient present anywhere in the original test archive. Breast (27 train/8 test), kidney (15/10), lung (27/4), and prostate (19/10) occur in both splits. Calibrated GigaTIME inference completed for all 120 ROIs; 37 prior completed outputs were reused only after decoded image, mask, annotation and pixel-spacing equivalence checks, and 83 new ROIs ran in one fail-fast bundle. The model input comprises 21 virtual markers over expert nuclei and the same 21 over 3 µm nucleus-anchored cell proxies.

One fixed fastPLS 0.4 SIMPLS–LDA model, fitted only on the 88 training ROIs, scored **10,776 test cells** in a historical run. Its **85.2% pooled accuracy** is misleading: nine of the 32 test images have only one expert class, accounting for 3,302 cells. Among the 23 multiclass images, equal-image mean recall was 56.7% and macro F1 was 50.5%. All these scores are **invalidated pre-QC results**, not a reliable GigaTIME-based typing benchmark. A retrospective screen of all 120 saved JPEG DAPI previews found 117 below the provisional expert-nucleus AUC threshold of 0.70; 116 source images also had red and blue reversed relative to the scanner RGB. The [class-coverage audit](docs/assets/monusac-matched/class-coverage.json) and [CPU scoring script](analysis/score_monusac_matched.py) now separate single-class images and require native nuclear QC for a new score; `--historical-unvalidated` is required to reproduce the old one. Original full-resolution images remain outside Git; public derived previews and the fitted model retain MoNuSAC's CC BY-NC-SA 4.0 terms.

The [source-format audit](analysis/audit_monusac_input_format.py) identified the reason this evaluation must be rerun. All 120 scanner SVS headers record 40× and 0.19448–0.2527 µm/pixel; the staged OME physical scale was correct and its Deflate compression was lossless. However, **116 companion TIFFs are mistagged as RGB while their red and blue channels are exactly swapped** relative to the matching scanner SVS. The previous preparation used these TIFFs without correcting the swap. The four SVS-only fields had the canonical RGB order. In `TCGA-2Z-A9JG-01Z-00-DX1_4` (0.2524 µm/pixel), the wrongly ordered input produced native DAPI with nearby expert-nucleus AUC 0.508 even after the overlap-edge correction. The [complete channel-order audit and case images](https://tkcaccia.github.io/Tambua-Analysis/#dapi-qc) document the error. The scanner SVS files themselves are JPEG2000-compressed; changing the lossless staging format cannot restore detail already lost there. The [Tambua preparer](https://github.com/tkcaccia/Tambua/blob/main/scripts/prepare_monusac_transfer.py) now decodes scanner SVS RGB for every case and checks companion TIFF channel order. **Both the 88 training and 32 test marker tables, then the fastPLS model, must be regenerated before reporting a corrected cell-type accuracy.**

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

The calibrated ORION inference has now completed with the pinned GigaTIME CNN checkpoint. It produced four preview channels and all 23 per-cell virtual scores for the 4,104 complete cells. The saved DAPI overlay was visually reviewed against hematoxylin-stained nuclei. Its high-signal overlap enrichment within the published cell regions was 2.068; this is a morphology proxy, not expert-mask Dice. See the [fixed results gallery](https://tkcaccia.github.io/Tambua-Analysis/) and [case record](docs/RESULTS.md).

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

The [gallery](https://tkcaccia.github.io/Tambua-Analysis/#monusac) contains all ten held-out prostate ROIs: source H&E, expert types, predicted types, error map, and virtual DAPI, CK, CD3 and CD68. The fixed evaluation used 27 lung training ROIs (9 patients) and 10 prostate test ROIs (6 distinct patients), with 5,243 and 3,584 expert-typed nuclei respectively. The original organizer image/XML pixel spacing was verified, then GigaTIME CNN inference was run at 0.25 µm/pixel. CellPhenotyper quantified 21 virtual channels on expert nuclear masks and 21 on 3 µm nucleus-anchored cell proxies. The latter are **not expert cell boundaries**.

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

The separate ORION CRC33_01 gallery is a one-patient technical demonstration with 4,104 computationally segmented cells and virtual-versus-measured marker associations. It has no expert cell-type truth. CellViT++ and HoVer-Net have **no verified same-cohort prediction** yet, so there is no head-to-head accuracy ranking.

## Sources

- [GigaTIME official code](https://github.com/prov-gigatime/GigaTIME) and [model card](https://huggingface.co/prov-gigatime/GigaTIME).
- [CellViT++ official code](https://github.com/TIO-IKIM/CellViT-plus-plus).
- [HoVer-Net official code](https://github.com/vqdang/hover_net).
- [ORION data release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md).
- [MoNuSAC challenge](https://monusac-2020.grand-challenge.org/Data/).

The code and site are MIT licensed. Third-party data and checkpoints retain their own terms.
