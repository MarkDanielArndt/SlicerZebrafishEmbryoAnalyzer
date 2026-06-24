# Core source provenance

The `zebrafish_analysis/` package (core + Slicer extension) and the `tests/`
in this repository were relocated, unmodified, from the development repository
below. This file records where the code came from.

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

## Notes

Relocation only — runtime behavior is preserved. Imports, `sys.path` handling,
CMake packaging, dependency installation, and model download behavior are
unchanged from the development commit. The `ZebrafishAnalysis` module name and
the nested directory structure are intentionally not yet flattened or renamed.
