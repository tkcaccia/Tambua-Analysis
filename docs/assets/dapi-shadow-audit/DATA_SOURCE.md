# Residual border audit

This is a CPU-only, single-patch diagnostic using the same released paired
GigaTIME H&E and thresholded DAPI reference as the adjacent control. The
current output is the pinned CellPhenotyper result from sample
`0_24464_556_556`. The trial reflected 128 pixels of context on all four sides
of its calibrated 512 × 512 model-grid input, then applied the same 256-pixel
CNN patches, 128-pixel stride and raised-cosine averaging before cropping back
to the original field. The official CNN checkpoint was unchanged.

The reported edge AUC evaluates only the outer 24 model pixels against the
released binary DAPI after nearest-neighbor resizing to 512 × 512. The current
output scored 0.855 and reflected context scored 0.838. Because the trial
worsened the edge reference agreement, it was not introduced into
CellPhenotyper. This comparison cannot establish whether the visible MoNuSAC
shading is due to a model-domain shift, an imaging difference or another cause.

The exact numerical results are in `report.json`. The virtual-marker scores
are research outputs and are not measured fluorescence or clinical results.
