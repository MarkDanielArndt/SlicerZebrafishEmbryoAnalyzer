# Core source

`ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerCore/` is a thin lazy
re-export layer over the PyPI library `zebrafish-measure`, pinned to
`==0.2.0` (repository: github.com/MarkDanielArndt/zebrafish-measure). The
library is derived from the reference webapp
(`markdanielarndt/Zebrafish_webapp`, live at
`https://huggingface.co/spaces/markdanielarndt/Zebrafish`) and is the single
source of truth for the analysis engine.

`length.py`, `seg.py`, `seg_helper.py` and `scalebar.py` in Core contain no
analysis code. Each is a PEP 562 module-level `__getattr__` that resolves a
name from `zebrafish_measure.<module>` on access. Importing a shim does not
import the library (nor torch, cv2, ...), and nothing is cached on the shim, so
`mock.patch` on these names works. `manual.py` is Slicer-only and stays here.

Switch date: 2026-10-21. Library source state: zebrafish-measure v0.2.0 (commit
`4c1b128`).

History: until this change Core was a manually maintained port of the webapp
(last synced 2026-07-20 for most files and 2026-07-29 for `seg.py` and
`compute_tube_metrics`). The per-file provenance table was removed with it.

## Installation and pin

`dependency_installer.py` installs the library into Slicer's Python with
`pip install --no-deps zebrafish-measure==0.2.0` (constants
`ZEBRAFISH_MEASURE_PACKAGE` / `ZEBRAFISH_MEASURE_REQUIREMENT`). `--no-deps`
keeps pip from touching Slicer's numpy, scipy, torch and friends. CI installs the
same pin (`.github/workflows/ci.yml`). The library needs Python >= 3.10.
`MIN_VERSIONS` in the same file (currently `segmentation_models_pytorch>=0.4.0`)
is checked by `get_missing_packages`: an installed but older package is listed
as missing and upgraded on the next install.

### Upgrade procedure

1. Bump the version in `ZEBRAFISH_MEASURE_REQUIREMENT` in
   `ZebrafishEmbryoAnalyzerLib/dependency_installer.py` and in the
   `pip install --no-deps zebrafish-measure==...` step of
   `.github/workflows/ci.yml`, together, in one change.
2. Re-run the parity check (results through the Core shims against the previous
   version on the same images, including the `logic.py` call sites) and the test
   suite.
3. Record any behaviour change in this file.

## Slicer-specific behaviour

Handled through arguments in `ZebrafishEmbryoAnalyzerLib/logic.py`
(`_install_model_cache`, `_seg_kwargs`):

- No Hugging Face download during analysis: the segmentation model calls pass
  `local_only=True`, so the library loads a local file only. The curvature
  classifier is local because `load_model(path)` is given a path
  (`weights_only=True`, `pretrained=False`, CUDA probe with CPU fallback).
  Downloading is `model_downloader.py`'s job, driven by `model_manifest.py`.
- Local-only loading uses `weights_only` checkpoint loading inside the library.
- Segmentation runs with `device="cpu"`.
- The fixed mask return order (eyes, edema, swim bladder) is preserved by the
  library.
- Swim bladder post-processing is the webapp's: `swimbladder_biggest_only=True`
  and `swimbladder_boundary_offset_px=-0.73` (the library defaults, passed
  explicitly so a future library default change cannot silently alter results).
  Eye metrics use `mask_fish` and `area_shrink_frac=0.05`, swim bladder metrics
  use `mask_fish` (as in the webapp). The Slicer output columns are unchanged.
- Each manifest entry names its `encoder` and `model_type` explicitly
  (`encoder_name` / `model_type` and the eye/edema/swim-bladder variants are
  passed per model).

Remains Slicer-only:

- `manual.py` (`compute_manual_length`).
- The model downloader with pinned revisions and SHA-256 verification, and
  `model_manifest.py`.
- The CUDA probe for the curvature classifier, now via the library's
  `select_torch_device`.

## Behaviour changes of the switch to the library

- Eye-assisted tube length now uses the library/webapp head-point routing and
  `length_smoothing_sigma=2.0` (the library default), so those lengths can differ
  from earlier versions.
- The library builds the smp networks with `encoder_weights=None`, so analysis no longer
  downloads ImageNet encoder weights (the checkpoint overwrites them anyway).
- The `compute_curvature*` plotting helpers are now available through the
  `length` shim but are unused by this extension.
- Optional models (eyes, edema, swim bladder) now raise `RuntimeError` on a
  corrupt checkpoint in local-only mode instead of returning an empty mask.

Measurement-value changes of the model alignment (2026-10-21):

- All masks change: the models are now the webapp's Segformer `mit_b3` models
  (8 files, revision `993a5d3`).
- Eye area is 5 % smaller (`area_shrink_frac=0.05`) and eye metrics use the fish
  mask.
- The swim bladder keeps only its biggest component, with a -0.73 px boundary
  offset, and its width is measured along the fish axis (`mask_fish`).
- Edema is now available on `general` (512 px model).
- Orphaned old model files stay in the cache dir (`user_cache_dir`
  `zebrafish_models`, about 0.7 GB). They may be deleted manually; there is no
  automatic cleanup.
- The per-run SHA-256 verification now covers about 1.1 GB of segmentation
  models plus the curvature model.
- Eye-assisted length routing only happens when Eyes is checked (`mask_eye` is
  passed only then), while the webapp always has eyes on. Slicer lengths with
  Eyes unchecked can therefore differ from the webapp.

## Model presets and weights

History: desy preset removed 2026-10-21: DESY models are webapp-only; the
library's eye-assisted length routing was not validated for that setup.

2026-10-21: Slicer now uses the webapp's models and post-processing.
`ZebrafishEmbryoAnalyzerLib/model_manifest.py` mirrors the webapp's
`SEG_MODEL_OPTIONS` for the 256 px and 512 px presets: every segmentation role
(body, eye, edema, swim bladder) of both presets is a Segformer `mit_b3` model
from `markdanielarndt/Zebrafish_Segmentation`, pinned to revision `993a5d3`
(`993a5d31cb3dde4ac8af72a5cbee389e1c25f239`), about 0.7 GB per preset. The
previous vgg/Unet/FPN files (revision `237d21d`) are no longer referenced.
The 1024 px webapp preset and DESY are not offered. The curvature classifier
entry is unchanged. The swim bladder filenames contain the real HF typo
`swimmbladder` (double m).

| Preset (webapp label) | `MODEL_SETS` key | Input size | Notes |
|---|---|---|---|
| `Fast & Easy (256 px, faster)` | `fast` | 256 | `*_256_segformer_mit_b3.pth` for body, eye, edema and swim bladder. |
| `Complex & Slower (512 px, slower)` | `general` | 512 | `*_512_segformer_mit_b3.pth` for body, eye, edema and swim bladder. |

The combo-box labels are the webapp's preset names with the timing figures
replaced by relative wording (faster/slower), so users recognise the same
presets in both tools. Slicer offers two of the webapp's presets. The stable ids (`fast`/`general`) are **not** the
webapp's strings — they are persisted in the MRML parameter node and travel
inside saved scenes, so they must not be renamed to match a display label.

Edema is offered on both presets (resolution-matched models); `widget.py` only
tests `"edema" in model_set`, so a preset without an edema model would grey the
checkbox out again.

## Convention

See `CLAUDE.md` -> "Code style" for the naming/signature convention. Slicer
code calls the library with the webapp's names and signatures; do not add
analysis logic to the Core shims. If a change in the analysis engine is needed,
make it in the webapp first, release it through `zebrafish-measure`, then bump the
pin as described above.
