# Registered ORION pixel comparison

This viewer contains one previously selected 2048 × 2048 H&E field from each
of ten reserved ORION colorectal patients with passing saved GigaTIME nuclear
and seam QC. The H&E fields and GigaTIME virtual DAPI were produced by the
bounded Chiamaka study at
`/mnt/sata_ssd/Tambua-orion-all-patient-2026-10-19`. The source H&E and
measured multiplex fluorescence are registered same-section images from the
[ORION CRC release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md).
The exact measured OME-TIFF URL, source window origin, and physical spacing
are recorded for each patient in `manifest.json` and in the saved
`MEASURED_PIXEL_SOURCE.json` files on Chiamaka.

`analysis/acquire_orion_pixel_if.py` reads only the Hoechst channel and frozen
window from the public 19-channel raw OME-TIFF using HTTP range requests. It
checks the OME channel list and 0.325 µm/pixel source spacing against the
registered H&E provenance. `analysis/acquire_orion_pixel_cohort.py` records a
completion marker only after all ten patients are acquired. No GPU inference
was run for this extraction. The saved virtual DAPI was previously inferred at
0.25 µm/pixel and is bilinearly resampled to the 0.325 µm/pixel source grid.

`analysis/score_orion_pixel_dapi.py` computes per-patient Spearman correlation
from native signals sampled at four-pixel intervals. Its cohort statistic is
the median of ten patient correlations, with a 10,000-replicate patient
bootstrap interval. The result is in `summary.json`. The visualization script
applies separate display-only transforms to the two channels; those displayed
values are never scored. `manifest.json` lists the display transform.

The equal-patient median pixel Spearman correlation is 0.806 (95% patient
bootstrap interval 0.752–0.849). Per-patient correlations range 0.624–0.876.
Contact sheets of all ten fields were visually reviewed. The virtual DAPI
captures much of the nuclear spatial pattern but is broader and more saturated
than the measured image in some fields. This experiment covers one field per
patient, not the entire slides, and tests DAPI only. It does not validate
virtual protein pixels, compartment-specific protein abundance, or cell-type
classification.
