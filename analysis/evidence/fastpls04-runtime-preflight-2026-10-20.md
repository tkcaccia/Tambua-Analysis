# Isolated fastPLS 0.4 runtime preflight

The current Chiamaka default R library resolves `fastPLS` **0.99.61**, while
Tambua's classifier bridge requires the [GitHub fastPLS 0.4](https://github.com/tkcaccia/fastPLS)
release. Setting `R_LIBS_USER` to the prepared 0.4 library did not change
`.libPaths()` on this host; the global package still won. The dedicated library
at `/mnt/sata_ssd/Tambua-monusac-rgb-corrected-2026-10-20-v1/r-library`
contains `fastPLS` **0.4**.

Tambua's existing R bridge recognizes `TAMBUA_R_LIB` and prepends that path
explicitly. The expanded scorer now passes this variable and verifies the
resolved package version and path **before** making an output directory or
building the cohort table. On Chiamaka, a CPU-only four-row two-class train
and predict preflight with that isolated library fitted one component and
recovered the four expected labels. The Python preflight resolved:

`/mnt/sata_ssd/Tambua-monusac-rgb-corrected-2026-10-20-v1/r-library/fastPLS`

This verifies runtime selection only. It is not a cell-type accuracy result.
No GigaTIME or GPU work was launched, and the live MoNuSAC CPU bundle was not
modified.
