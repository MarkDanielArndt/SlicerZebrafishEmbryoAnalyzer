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
- Swim bladder: `swimbladder_biggest_only=False` and
  `swimbladder_boundary_offset_px=None` (no boundary offset).
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
- The library builds Unet/FPN with `encoder_weights=None`, so analysis no longer
  downloads ImageNet encoder weights (the checkpoint overwrites them anyway).
- The `compute_curvature*` plotting helpers are now available through the
  `length` shim but are unused by this extension.
- Optional models (eyes, edema, swim bladder) now raise `RuntimeError` on a
  corrupt checkpoint in local-only mode instead of returning an empty mask.

## Model presets and weights

History: desy preset removed 2026-10-21: DESY models are webapp-only; the
library's eye-assisted length routing was not validated for that setup.

`ZebrafishEmbryoAnalyzerLib/model_manifest.py` mirrors the webapp's
`SEG_MODEL_OPTIONS` table. Synced 2026-07-29.

Read that table together with the comment directly above it and with the call
site in `process()`. **A `None` filename there means "use the pipeline default",
not "this preset has no such model."** Reading it the other way produced two
wrong implementations on 2026-07-29 (edema wrongly restricted to one
preset, and the Fast & Easy body model discarded as a legacy file). Ports of
this table should be checked against the *resolved* filenames, not the literal
cell contents.

| Preset (webapp label) | `MODEL_SETS` key | Input size | Notes |
|---|---|---|---|
| `Fast & Easy (256 px, ~2s/image)` | `fast` | 256 | Body/eye/edema/swim-bladder all resolve to the pipeline defaults. Its swim bladder model is Unet + vgg16, unlike the 512px presets' FPN + vgg19. |
| `Complex & Slower (512 px, ~7s/image)` | `general` | 512 | Names body, eye and swim bladder explicitly; edema falls through to the default. |

The combo-box labels are the webapp's verbatim so users recognise the same
presets in both tools. The stable ids (`fast`/`general`) are **not** the
webapp's strings — they are persisted in the MRML parameter node and travel
inside saved scenes, so they must not be renamed to match a display label.

**Deliberate deviation — edema on the 512px general preset.** The webapp offers
edema there and, having no 512px general edema model, falls back to the 256px
default and feeds it 512px input. This port omits the role for that preset
instead and greys the checkbox out.

Reason: Unet and FPN are fully convolutional and accept any input size divisible
by 32, so a resolution mismatch raises nothing — it yields a mask from which a
plausible-looking µm² figure is computed, indistinguishable from a correct one in
a result table. For a measurement tool a silently wrong number is worse than an
unavailable feature.

Rollback: if a 512px general edema model appears upstream, add it to `MODELS` and
wire it into `MODEL_SETS["general"]` under `"edema"`. The checkbox re-enables
itself — `widget.py` only tests `"edema" in model_set` — and no other change is
needed.

## Convention

See `CLAUDE.md` -> "Code style" for the naming/signature convention. Slicer
code calls the library with the webapp's names and signatures; do not add
analysis logic to the Core shims. If a change in the analysis engine is needed,
make it in the webapp first, release it through `zebrafish-measure`, then bump the
pin as described above.
