# Tambua Analysis

[Validation site](https://tkcaccia.github.io/Tambua-Analysis/) · [Tambua software](https://github.com/tkcaccia/Tambua)

This repository holds the prospective analysis plan and public, reproducible scoring code for Tambua cell typing. Tambua uses CellPhenotyper GigaTIME virtual-marker scores and [GitHub fastPLS **0.4**](https://github.com/tkcaccia/fastPLS/tree/265108b6aa250c2534f98ec89037bb5cd001dca0). It does **not** claim a matched biological accuracy comparison before the methods are evaluated on the same held-out patients.

## Comparison design

Use the official MoNuSAC epithelial, lymphocyte, macrophage and neutrophil ontology. Freeze patient-disjoint training, validation and test groups. Compare the following on identical held-out H&E images and expert reference instances:

| Method | Input | Model source | Endpoint |
| --- | --- | --- | --- |
| Tambua | CellPhenotyper instances and calibrated GigaTIME scores | [Tambua](https://github.com/tkcaccia/Tambua), [fastPLS 0.4](https://github.com/tkcaccia/fastPLS) | End-to-end cell detection and type; separately classify on expert instances to isolate the marker/classifier step |
| CellViT | H&E | [official CellViT](https://github.com/TIO-IKIM/CellViT) | Native instance detection and type, with a prospectively fixed mapping to MoNuSAC classes |
| HoVer-Net | H&E | [official HoVer-Net](https://github.com/vqdang/hover_net) | Native instance detection and type using its MoNuSAC checkpoint |

CellViT's publicly described PanNuke taxonomy is **not** interchangeable with the four MoNuSAC types. A MoNuSAC-trained cell-type head or a documented external mapping must be fixed without looking at the held-out test set. HoVer-Net provides a MoNuSAC-specific pretrained checkpoint. Do not compare Tambua on expert-provided masks to competitors' end-to-end outputs as if they were equivalent. Report the expert-mask classifier analysis and end-to-end analysis separately.

For end-to-end evaluation, match predicted and expert instances one-to-one at mask IoU > 0.5, count unmatched instances, and report detection F1 and per-type F1 in addition to classification accuracy on matched cells. Report macro F1 and balanced accuracy only when every reference type is present. Aggregate by patient and add patient-bootstrap uncertainty when there are enough independent patients. Record physical resolution, model checkpoint, inference time, and hardware. Never treat cells from one patient as independent biological replicates.

## Physical scale and DAPI quality control

The current bounded ORION CRC33_01 source window is a registered same-section H&E/whole-cell mask at 0.325 µm/pixel. A strict GigaTIME grid at 0.25 µm/pixel should be 2663 × 2663 for a 2048 × 2048 source window. Record the exact transform and confirm the DAPI virtual channel spatially overlaps H&E nuclear morphology. A numerical H&E hematoxylin proxy can support, but cannot replace, visual review or a true expert nuclear mask. ORION's whole-cell mask should be called `cell`, not `nuclei`.

The one approved Chiamaka inference stopped before model loading because an offline `main` cache snapshot had no weights. No DAPI image or cell-typing accuracy resulted. CellPhenotyper now pins the cached weight-bearing GigaTIME CNN revision and its CPU state dictionary loads successfully; the GPU case has **not** been rerun.

When a scale-valid GigaTIME result exists, run the spatial QC on the **same physical field of view**:

```bash
python -m analysis.qc_dapi --he he.ome.tif --cell-mask expert_cells.ome.tif \
  --dapi gigatime_output/jpg_image/DAPI.jpg \
  --manifest gigatime_output/gigatime_jpg_manifest.json \
  --output-dir results/dapi_qc
```

The tool refuses a source/target pixel-spacing or image-shape mismatch, saves an H&E/DAPI overlay, and reports enrichment of high virtual DAPI over an H&E hematoxylin nuclear **proxy**. That proxy is not an expert nuclear annotation; the resulting overlay still needs visual review. No QC threshold or biological accuracy is claimed from the failed ORION run.

For a frozen end-to-end comparison, create one CSV manifest per method with columns `sample_id,patient_id,reference_mask,reference_labels,prediction_mask,prediction_labels`. Each labels CSV has `label_id,cell_type`. Score only the test cases, with all methods using the same cases and four-class ontology:

```bash
python -m analysis.score_instances --manifest test_tambua.csv --method Tambua \
  --track end_to_end --output results/tambua.json
python -m analysis.score_instances --manifest test_cellvit.csv --method CellViT \
  --track end_to_end --output results/cellvit.json
python -m analysis.score_instances --manifest test_hovernet.csv --method HoVer-Net \
  --track end_to_end --output results/hovernet.json
```

The scorer rejects missing or extra typed instances and incompatible mask shapes. It reports per-sample, per-patient and pooled detection/type scores. Use `--track expert_mask_classification` only when prediction and reference instance masks are identical; this is a separate conditional-classification experiment.

## Data and current status

- MoNuSAC: 310 mirrored annotated images, but the mirror does not establish exact µm/pixel for GigaTIME. Use only after independently verifying scanner spacing. See [Tambua dataset notes](https://github.com/tkcaccia/Tambua/blob/main/docs/DATASETS.md).
- ORION CRC33_01: one 2048 × 2048 calibrated window, 4,104 complete measured-table-matched cell instances, 226 edge instances excluded. This is one patient; it cannot support a patient-held-out accuracy estimate.
- Matched Tambua / CellViT / HoVer-Net scores: **not yet measured**.

Raw images, model weights and annotations remain with their original providers. This repository does not redistribute them.

## Sources

- [GigaTIME official code](https://github.com/prov-gigatime/GigaTIME) and [model card](https://huggingface.co/prov-gigatime/GigaTIME).
- [CellViT official code](https://github.com/TIO-IKIM/CellViT).
- [HoVer-Net official code](https://github.com/vqdang/hover_net).
- [ORION data release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md).
- [MoNuSAC challenge](https://monusac-2020.grand-challenge.org/Data/).

The code and site are MIT licensed. Third-party data and checkpoints retain their own terms.
