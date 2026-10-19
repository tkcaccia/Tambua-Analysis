# TCGA-2Z-A9JG DAPI diagnostic

MoNuSAC image `TCGA-2Z-A9JG-01Z-00-DX1_4` and its expert nucleus mask
come from the acquired MoNuSAC 2020 organizer release. The source is
0.2524 µm/pixel. The pinned GigaTIME CNN revision
`a6556030bdb8071857559e12b921620aa1bbac5b` was run on the
0.25 µm/pixel model grid. `current_dapi.png` is a CPU replay with the
CellPhenotyper corrected overlap denominator; the red contours in
`current_dapi_expert_overlay.png` come from the independently provided
expert nucleus mask, not the virtual DAPI.

The replay on the historical red/blue-swapped input removes the legacy dark image edge but gives expert
nucleus-versus-nearby-background AUC 0.508; all nine local regions score below
0.60. The source image and old JPEG preview are already in the public
MoNuSAC gallery. `legacy_120_image_audit.json` checks all 120 historical JPEG
previews, of which 117 fall below global AUC 0.70. Those JPEG measurements
are retrospective diagnostics; they are not native-store marker validation.
No marker quantification or cell-type score is accepted from this field.

The second CPU replay used the scanner's canonical RGB, with the same
checkpoint, physical grid, tiling and expert mask. Its nearby-nucleus AUC is
0.828, all nine local sector AUCs exceed 0.785, and its edge/core nuclear
signal ratio is 1.012. Files prefixed `corrected_rgb_` hold this replay.
An approximate full QC computed from the saved 8-bit display PNG finds
boundary contrast 0.038, below the provisional 0.05 threshold. That PNG
check is not the native-store production gate and cannot validate other
virtual markers or the historical cell classifier.

`stardist_labels.tif` was independently predicted from this canonical
scanner-RGB H&E field on CPU with the cached StarDist `2D_versatile_he`
checkpoint (probability threshold 0.48, NMS 0.30). It contains 210 instances
at the source 755 × 601 grid. For the overlay only, labels were mapped by
nearest-neighbor to the 763 × 607 GigaTIME preview grid. StarDist is not a
fluorescence reference or a substitute for the expert nuclei.

`dapi_display.png` and `dapi_display_stardist_overlay.png` apply
CellPhenotyper's whole-preview, mask-independent 2nd–99th percentile and
gamma-2.0 display curve to the *linear 8-bit preview*. The virtual DAPI
native values, nuclear QC, and marker quantification are unchanged. The
nearby-StarDist-nucleus AUC is approximately 0.818 before and after the
display curve; visual contrast does not improve the underlying alignment.
`dapi_display_report.json` gives exact input hashes and curve parameters.

`input_format_120_audit.json` compares every staged input with its companion TIFF and canonical scanner SVS. All scanner headers report 40×, and OME pixel spacing matches the header. **The 116 companion TIFFs have their red and blue channels exactly reversed relative to scanner SVS RGB, despite RGB tags.** This case is among them: its opaque-alpha LZW TIFF was copied into a lossless Deflate staged image with the wrong channel order. The four SVS-only sources were correctly decoded with OpenSlide. All scanner SVS images use JPEG2000; lossless transcoding cannot undo that acquisition compression.

MoNuSAC imagery and annotations retain the organizer's CC BY-NC-SA 4.0
terms. The analysis code is in `analysis/audit_monusac_shadow.py`,
`analysis/check_current_gigatime_shadow_cpu.py`, and
`analysis/audit_monusac_input_format.py`. The reproducible visualization
command is `analysis/render_dapi_display.py` with the CellPhenotyper source
passed through `--cellphenotyper-code`.
