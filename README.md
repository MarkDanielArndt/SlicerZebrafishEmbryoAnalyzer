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
zebrafish_analysis/
  __init__.py
  core/              shared analysis logic (seg, length, scalebar, manual, seg_helper)
  slicer_extension/  3D Slicer extension
    ZebrafishAnalysis.s4ext
    CMakeLists.txt
    ZebrafishAnalysis/
      ZebrafishAnalysis.py
      Resources/Icons/ZebrafishAnalysis.png
      ZebrafishAnalysisLib/   widget, logic, tabs, overlay, export, dependency_installer
tests/               core and Slicer-library tests runnable outside Slicer
```

## Installing the extension in 3D Slicer

1. Open 3D Slicer.
2. Edit → Application Settings → Modules.
3. Add `zebrafish_analysis/slicer_extension/ZebrafishAnalysis` to the
   **Additional module paths**.
4. Restart Slicer. On first load the extension pip-installs its Python
   dependencies into Slicer's interpreter; a second restart is required after
   installation completes.
5. Open the **ZebrafishAnalysis** module from the Modules dropdown.

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
