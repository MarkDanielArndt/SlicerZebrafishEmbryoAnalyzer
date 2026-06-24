# SlicerZebrafishEmbryoAnalyzer

A 3D Slicer extension for offline zebrafish morphometry from 2-D microscopy
images: body length, curvature classification, eye metrics, scalebar
detection, batch processing, and Excel/CSV export.

The extension and its shared core were relocated from the
[Zebrafish_Analysis](https://github.com/JonaRichter/Zebrafish_Analysis)
development repository. See [CORE_SOURCE.md](CORE_SOURCE.md) for exact
provenance.

## Layout

```
CMakeLists.txt              extension-level CMake
ZebrafishAnalysis.s4ext     extension descriptor
ZebrafishAnalysis/          3D Slicer module
  CMakeLists.txt
  ZebrafishAnalysis.py      module entry point
  ZebrafishAnalysisLib/     Slicer UI and adapters (widget, logic, tabs, overlay, export, dependency_installer)
  ZebrafishAnalysisCore/    shared analysis logic (seg, length, scalebar, manual, seg_helper)
  Resources/Icons/ZebrafishAnalysis.png
tests/                      core and Slicer-library tests runnable outside Slicer
```

## Installing the extension in 3D Slicer

1. Open 3D Slicer.
2. Edit → Application Settings → Modules.
3. Add the `ZebrafishAnalysis` directory (at the repository root) to the
   **Additional module paths**.
4. Restart Slicer. On first load the extension pip-installs its Python
   dependencies into Slicer's interpreter; a second restart is required after
   installation completes.
5. Open the **ZebrafishAnalysis** module from the Modules dropdown.

## Platform support

The extension is intended to run natively on Windows, macOS and Linux.

Development and clean-install testing have currently been performed on macOS.
Automated and manual verification on Windows and Linux is still being added.

## Tests

The core and Slicer-library tests under `tests/` run outside Slicer, in a
normal Python environment. They currently require the full project runtime
dependencies:

- numpy
- opencv-python
- torch
- torchvision
- segmentation-models-pytorch
- timm
- scipy
- scikit-image
- matplotlib
- pillow
- huggingface-hub
- openpyxl
- pytest

```bash
pytest tests/
```

A reproducible development environment (pinned requirements) will be added in
a follow-up change.
