import sys

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


# Slicer puts this module's directory on sys.path, so ZebrafishAnalysisLib and
# ZebrafishAnalysisCore import as normal packages — no path manipulation needed.

_LIB_MODULES = (
    "ZebrafishAnalysisLib.widget",
    "ZebrafishAnalysisLib.gallery_tab",
    "ZebrafishAnalysisLib.detail_tab",
    "ZebrafishAnalysisLib.results_tab",
    "ZebrafishAnalysisLib.exclude_tab",
    "ZebrafishAnalysisLib.logic",
    "ZebrafishAnalysisLib.overlay",
    "ZebrafishAnalysisLib.export",
    "ZebrafishAnalysisLib.dependency_installer",
    "ZebrafishAnalysisLib.zoom_view",
)

def _evict_lib_modules():
    for _m in _LIB_MODULES:
        sys.modules.pop(_m, None)

_evict_lib_modules()


class ZebrafishAnalysis(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Zebrafish Analysis"
        self.parent.categories = ["Quantification"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jona Richter", "Mark Daniel Arndt"]
        self.parent.helpText = (
            "Segment zebrafish from 2-D microscopy images and measure "
            "body length, curvature class, length/straight-line ratio, "
            "and eye metrics."
        )
        self.parent.acknowledgementText = (
            "Based on the Zebrafish Webapp "
            "(github.com/MarkDanielArndt/Zebrafish_webapp)."
        )


class ZebrafishAnalysisWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _evict_lib_modules()

        from ZebrafishAnalysisLib.dependency_installer import check_and_install
        check_and_install()

        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget
        self._main = ZebrafishAnalysisMainWidget(self.layout)

        qt.QTimer.singleShot(500, self._prewarm_imports)

    def _prewarm_imports(self):
        import sys
        import threading
        # Skip on fresh install — torch not yet imported, first import takes several
        # seconds and would freeze the UI if the user opens a file dialog concurrently.
        if "torch" not in sys.modules:
            return

        def _work():
            try:
                from ZebrafishAnalysisCore.seg import segmentation_pipeline    # noqa: F401
                from ZebrafishAnalysisCore.length import load_model             # noqa: F401
            except Exception:
                pass

        threading.Thread(target=_work, daemon=True).start()

    def cleanup(self):
        pass


class ZebrafishAnalysisLogic(ScriptedLoadableModuleLogic):
    pass
