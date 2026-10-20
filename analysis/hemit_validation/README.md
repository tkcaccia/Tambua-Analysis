# HEMIT as an independent GigaTIME validation set

Status: 287 source-eligible test pairs completed with the pinned GigaTIME CNN
on Chiamaka. All case checkpoints and technical seam checks passed; process
and GPU-context cleanup were verified. This is an external three-marker
comparison, not cell-type classification validation.

## Source and purpose

The public [HEMIT release](https://data.mendeley.com/datasets/3gx53zm49d/1)
(DOI 10.17632/3gx53zm49d.1, CC BY 4.0) contains paired colorectal H&E and
multiplex-IHC TIFFs from the same tissue sections. Its
[dataset description](https://github.com/BianChang/HEMIT-DATASET) reports eight
source whole slides, 1024 × 1024 registered pairs, three channels (DAPI,
pan-cytokeratin, CD3), and 945 official test pairs. The parent
[ImmunoAIzer report](https://pmc.ncbi.nlm.nih.gov/articles/PMC8036970/)
reports 40× Mantra scanning. A later
[MIPHEI-ViT study](https://arxiv.org/abs/2505.10294) describes HEMIT as
0.25 µm/pixel. The downloaded TIFFs do not carry independent physical-size
tags, so the literature-reported scale is a provenance limitation; objective
magnification alone is not a pixel-size measurement.
The [HEMIT paper](https://arxiv.org/html/2403.18501) states that its
train/validation/test partitions use different patients and that its H&E
patches were stain-normalized after registration.

This set tests external transfer of the pinned GigaTIME model on an independent
acquisition and assay. It complements the ORION evaluation, which has a broader
marker panel. HEMIT is a three-marker test; it cannot validate the other 20
GigaTIME channels. These multiplex-IHC image channels are not necessarily
calibrated protein-concentration measurements, so the primary endpoints should
be spatial localization and marker-positive-cell discrimination. Raw intensity
correlation is secondary and must be interpreted cautiously.

The [GigaTIME paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/12/Multimodal-AI-generates-virtual-population-for-tumor.pdf)
describes model training on 21 in-house paired lung slides generated with the
COMET platform. HEMIT is a separate colorectal release and is not described
as part of that training corpus. The pinned checkpoint's full private
training-image manifest is unavailable, so dataset non-overlap is supported
by the published acquisition descriptions rather than proven by a hash audit.

## Frozen evaluation design before inference

1. Verify the downloaded archive checksum, exact input/label pair count, TIFF
   dimensions, channel semantics, byte depth, color order, and physical pixel
   size from source metadata or an independently documented calibration.
   The [MIPHEI HEMIT preprocessing](https://github.com/Sanofi-Public/MIPHEI-ViT/blob/main/datasets/preprocessing/hemit_preprocess.py)
   assigns TIFF bands as Pan-CK, CD3, DAPI, rather than the prose ordering in
   the HEMIT README. Verify that mapping on actual pixels before scoring.
   The published 0.25 µm/pixel HEMIT calibration was used for inference.
2. Choose only official **test** pairs. Because 1024-pixel crops were generated
   on a 512-pixel stride, use the published
   [DiffVS nonoverlapping, nonempty 292-pair list](https://github.com/hvcl/DiffVS/tree/main/splits/hemit),
   frozen locally as `hemit_test_no_overlap_no_empty_ids.txt` (SHA-256
   `788656b2b9fcd4254bd65a8c1903ad4fed3014e089b61f220674d4293af50183`).
   Exclude the five source H&E images below the prespecified 100,000-pixel
   tissue gate before inference. Report patch and identifiable source counts.
   This list was fixed before viewing GigaTIME predictions.
3. Use the literature-reported 0.25 µm/pixel HEMIT pixels on GigaTIME's
   0.25 µm/pixel model grid; no geometric resampling is needed for this set.
   Preserve the released RGB channel order and use the corrected seam-free
   CellPhenotyper GigaTIME input contract. HEMIT's creators already applied
   stain normalization to H&E; record this domain change rather than calling
   the released TIFF raw scanner RGB. Record image hashes and model revision.
4. Score only the predeclared DAPI, CK↔panCK, and CD3↔CD3 pairs. Save every
   H&E/measurement/prediction triplet, run border-seam QC on every output and
   review the complete contact-sheet overview plus full-size score extremes.
   Exclude QC failures by fixed rules, never by protein score.
5. Report DAPI nuclear localization, CD3 and panCK positive-region AUROC and
   average precision (with positive prevalence), and tissue-pixel Spearman as
   a secondary diagnostic. Use source-slide-level uncertainty only if the
   released test split contains enough identifiable independent slides. If cell
   instances from the independent HEMIT nuclei resource are used, report their
   segmentation provenance rather than calling them expert masks.

## Data location

The source archive is saved on Chiamaka at
`/mnt/sata_ssd/Tambua-benchmark/HEMIT/HEMIT.7z`, SHA-256
`efe7dd5dc50ec0f789b0ce8a5258f5281a38001ac3dca81620f2a4be7041c943`.
It is not stored in Git. `download_hemit.sh` reproduces the public download
and archive checksum check. Acquisition and metadata audit use CPU and network
only. The completed GPU inference used a separate, bounded Chiamaka scope under
the local project policy at
`/Users/stefano/Documents/ChatGPT/Development Coordinator/docs/hpc-gpu-policy.md`
before inference; no UCT HPC GPU execution was used.

The exact Mendeley public-file URL is
`https://data.mendeley.com/public-files/datasets/3gx53zm49d/files/d197d039-397e-4df7-a990-0e7c79a10f47/file_downloaded`.
Its reported and downloaded length is 10,562,710,923 bytes. The archive
extracted with 7-Zip without errors. The official test split has 945 exact
filename-matched H&E/target pairs. The 292-name published selection is a
subset of the 300 even/even nonoverlapping candidates.

The released TIFFs are RGB uint8 at 1024 × 1024 pixels, but the five audited
pairs have **no usable physical-resolution TIFF tags**. The separate published
0.25 µm/pixel HEMIT calibration is therefore required and must be recorded in
each prediction's provenance. The target bands are visibly consistent with
red Pan-CK, green CD3 and blue DAPI. See `INPUT_AUDIT.json` and
`input-contact.webp` for the initial five-pair audit, which has no model output.

The one-case CPU pilot used the pinned GigaTIME CNN and the corrected
CellPhenotyper 256-pixel windows with 128-pixel stride and weighted overlap.
On `[16705,49971]_patch_4_4.tif`, DAPI measured-versus-virtual pixel Spearman
was 0.768, CD3-positive pixel AUROC 0.621, and Pan-CK-positive pixel AUROC
0.527. This is an exploratory feasibility result, **not a HEMIT cohort
estimate**. Its paired visual is `pilot-paired-contact.webp`; see
`PILOT_REPORT.json` for hashes, timings, target prevalence and full metrics.

The first authorized GPU bundle stopped cleanly before case 41 could be scored:
its H&E image had only 25,411 pixels under the fixed tissue gate of 100,000.
The process and GPU-context cleanup were verified. Its first 40 checkpoints
are preserved on Chiamaka but must not be presented as a cohort benchmark.
A subsequent **CPU-only source-image** audit found five low-tissue pairs among
the 292 and no constant measured channels among the other 287. This fixes the
eligibility list independently of virtual predictions; see
`SOURCE_TISSUE_ELIGIBILITY.json`. A fresh, separately authorized Chiamaka GPU
validation completed all 287 in `gpu-validation-287-v2`. The five exclusions
were selected only from source H&E tissue area before inference.
`hemit_test_eligible_287_ids.txt` and `ELIGIBLE_INPUT_AUDIT.json` freeze that
selection; no first-run partial scores were reused. `GPU_OUTCOME_287.txt`
records exit 0 and verified cleanup.

## Completed independent comparison

The full 287-image, three-marker report is
`docs/assets/hemit-validation/COHORT_SUMMARY.json` and every per-image score is
`docs/assets/hemit-validation/PER_IMAGE_MARKER_SCORES.csv`. The 287 paired
image contacts are in `docs/assets/hemit-validation/case-*/paired-contact.webp`.
The median tissue-pixel Spearman correlations (patch IQR) are DAPI 0.768
(0.737–0.793), CD3 0.127 (0.058–0.175), and Pan-CK 0.234 (0.092–0.379).
Against released marker pixels greater than zero, median AUROC is 0.645
for CD3 and 0.669 for Pan-CK. Median positive prevalence is 0.078 and
0.633, respectively; Pan-CK average precision must be interpreted against
its high base rate. These patch-level results are descriptive because the
source-slide/patient identities of the 15 coordinate prefixes are not
verified. No patient-level uncertainty interval is claimed.

All 287 paired images were inspected in ten low-resolution overview sheets;
the lowest and highest patch correlations for all three markers were inspected
at full contact resolution. No gross straight grid seam was visible in the
overview, consistent with 287 passing technical seam reports. Fine-scale
artifact absence cannot be certified from these overview sheets. DAPI usually
follows nuclei but can be too bright
or miss sparse nuclei. Several CD3 and Pan-CK images show misplaced or
diffuse virtual signal despite a passing technical seam gate. This independent
set therefore reinforces that virtual-DAPI alignment is not evidence that
the protein channels are quantitatively reliable for downstream cell typing.
The public gallery provides all 287 pairs for image-level review.
