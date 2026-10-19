# MoNuSAC-derived materials

The H&E previews, expert nuclear overlays, cell-type labels, and fitted model in this directory derive from the [MoNuSAC 2020 organizer data](https://monusac-2020.grand-challenge.org/Data/) and are shared under its **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International** terms. They are not covered by this repository's MIT code license. The original source images, expert XML annotations and full-resolution prepared images are not redistributed here.

The virtual-marker previews were computed by CellPhenotyper from the pinned [GigaTIME CNN model](https://huggingface.co/prov-gigatime/GigaTIME/tree/a6556030bdb8071857559e12b921620aa1bbac5b). They are uncalibrated virtual predictions. The `fastpls-lung-to-prostate.rds` model was trained with [fastPLS 0.4](https://github.com/tkcaccia/fastPLS) on the 27 MoNuSAC lung training ROIs identified in `cohort.json`. It is provided for reproducibility, subject to the MoNuSAC data terms.

The published image panels are resized, quality-reduced adaptations for inspection. Their precise per-image physical calibration, expert counts and score matrices are recorded in `results.json`. See the repository README for acquisition and reproduction instructions.
