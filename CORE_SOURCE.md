# Core source provenance

The analysis core and Slicer extension in this repository originate, unmodified,
from the development repository below. This file records where the code came
from. The snapshot was subsequently relocated into this repository, given a
clean-install fix, and flattened into a standalone extension layout (core now
under `ZebrafishAnalysis/ZebrafishAnalysisCore/`, library under
`ZebrafishAnalysis/ZebrafishAnalysisLib/`). The recorded commit and tree hashes
below describe the original source paths, not the current layout.

## Canonical upstream

MarkDanielArndt/Zebrafish_webapp

## Development source

JonaRichter/Zebrafish_Analysis

## Development branch

pr/slicer-extension

## Development commit

0c98df2a88a1d50b004a929ac1142af4c108e36b

## Core Git tree SHA

`zebrafish_analysis/core` @ 0c98df2: 2b5ef92b6a8950d1ef60452ed1153c70dc934759

## Slicer extension Git tree SHA

`zebrafish_analysis/slicer_extension` @ 0c98df2: ce2453e33015d5f871d12464a667fa83ecb78c61

## Copy date

2026-06-24

## Local modifications after vendoring

The SHAs and commit above record the **vendored snapshot** that was copied in.
The tree has since diverged through a clean-install fix (optional matplotlib
import, analysis error handling) and a layout flatten (directory moves and
updated imports). The working tree is therefore **no longer byte-identical** to
commit `0c98df2`; the recorded commit is provenance of the original source, not
a description of the current contents.

## Notes

Runtime behavior is preserved across the relocation and flatten: analysis
algorithms, model URLs and cache behavior, dependency installation, NumPy
pinning, and the `ZebrafishAnalysis` module name are unchanged. Only the
directory layout and import paths were updated. Repository/Extension Index URLs
are handled separately.
