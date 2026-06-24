import sys

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin


# Slicer puts this module's directory on sys.path, so ZebrafishAnalysisLib and
# ZebrafishAnalysisCore import as normal packages — no path manipulation needed.

_LIB_MODULES = (
    "ZebrafishAnalysisLib.errors",
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


class ZebrafishAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._main = None
        self._sceneObserversRegistered = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _evict_lib_modules()

        from ZebrafishAnalysisLib.dependency_installer import check_and_install
        check_and_install()

        self.logic = ZebrafishAnalysisLogic()

        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget
        self._main = ZebrafishAnalysisMainWidget(self.layout, logic=self.logic)

        self._register_scene_observers()

        if hasattr(self, "_prewarm_timer"):
            self._prewarm_timer.stop()
        self._prewarm_timer = qt.QTimer()
        self._prewarm_timer.setSingleShot(True)
        self._prewarm_timer.setInterval(500)
        self._prewarm_timer.timeout.connect(self._prewarm_imports)
        self._prewarm_timer.start()

    def _register_scene_observers(self):
        """Register MRML scene close observers exactly once per setup."""
        if self._sceneObserversRegistered:
            return
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.StartCloseEvent,
            self._on_scene_start_close,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndCloseEvent,
            self._on_scene_end_close,
        )
        self._sceneObserversRegistered = True

    def enter(self):
        pass

    def exit(self):
        pass

    def cleanup(self):
        if hasattr(self, "_prewarm_timer"):
            self._prewarm_timer.stop()
        self.removeObservers()
        self._sceneObserversRegistered = False
        if self._main is not None:
            self._main.cleanup()

    def _on_scene_start_close(self, caller=None, event=None):
        # Invalidate background workers early so they discard results for the
        # closing scene; the UI reset happens in _on_scene_end_close once the
        # scene is fully closed.
        if self._main is not None:
            self._main._cancel_workers()

    def _on_scene_end_close(self, caller=None, event=None):
        if self._main is not None:
            self._main.reset_for_scene_close()

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


class ZebrafishAnalysisLogic(ScriptedLoadableModuleLogic):
    """Orchestrates analysis requests on behalf of the widget.

    Widget calls these methods; each delegates to the corresponding free
    function in ZebrafishAnalysisLib.logic so the widget never imports that
    module directly.  ZebrafishAnalysisCore remains Slicer-independent.
    """

    def run_analysis(self, image_paths, params, progress_callback=None):
        import math
        import os
        from collections.abc import Mapping, Sequence
        from ZebrafishAnalysisLib.errors import AnalysisInputError

        # image_paths: must be a non-string, non-empty Sequence of path-like values
        if not isinstance(image_paths, Sequence) or isinstance(image_paths, (str, bytes)):
            raise AnalysisInputError(
                f"image_paths must be a Sequence of paths (list or tuple), "
                f"got {type(image_paths).__name__!r}"
            )
        if not image_paths:
            raise AnalysisInputError("No images loaded")
        for p in image_paths:
            try:
                os.fspath(p)
            except TypeError:
                raise AnalysisInputError(
                    f"image_paths entries must be path-like strings, "
                    f"got {type(p).__name__!r}"
                )

        # params: must be a Mapping (isinstance check, not duck-typing .get())
        if not isinstance(params, Mapping):
            raise AnalysisInputError(
                f"params must be a Mapping, got {type(params).__name__!r}"
            )

        # um_per_px: numeric, finite, in UI range [0.001, 9999.0]
        raw_um = params.get("um_per_px", 22.99)
        try:
            um_per_px = float(raw_um)
        except (TypeError, ValueError):
            raise AnalysisInputError(
                f"params['um_per_px'] must be numeric, got {type(raw_um).__name__!r}"
            )
        if not math.isfinite(um_per_px):
            raise AnalysisInputError(
                f"params['um_per_px'] must be finite, got {um_per_px!r}"
            )
        if not (0.001 <= um_per_px <= 9999.0):
            raise AnalysisInputError(
                f"params['um_per_px'] must be in [0.001, 9999.0], got {um_per_px!r}"
            )

        # threshold: numeric, finite, in UI range [0.0, 1.0]
        raw_thr = params.get("threshold", 0.85)
        try:
            threshold = float(raw_thr)
        except (TypeError, ValueError):
            raise AnalysisInputError(
                f"params['threshold'] must be numeric, got {type(raw_thr).__name__!r}"
            )
        if not math.isfinite(threshold):
            raise AnalysisInputError(
                f"params['threshold'] must be finite, got {threshold!r}"
            )
        if not (0.0 <= threshold <= 1.0):
            raise AnalysisInputError(
                f"params['threshold'] must be in [0.0, 1.0], got {threshold!r}"
            )

        # Normalize: convert paths to str and write validated floats back.
        # Work on copies so the caller's list and dict are never mutated.
        normalized_paths = [os.fspath(p) for p in image_paths]
        normalized_params = dict(params)
        normalized_params["um_per_px"] = um_per_px
        normalized_params["threshold"] = threshold

        from ZebrafishAnalysisLib.logic import analyse_images
        return analyse_images(normalized_paths, normalized_params, progress_callback)

    def detect_scalebar(self, image_path, label_um=None):
        from ZebrafishAnalysisLib.logic import detect_scalebar
        return detect_scalebar(image_path, label_um=label_um)

    def preload_models(self, params):
        from ZebrafishAnalysisLib.logic import preload_models
        return preload_models(params)

    def apply_manual_correction(self, result, point1_orig, point2_orig, params=None):
        from ZebrafishAnalysisLib.logic import apply_manual_correction
        return apply_manual_correction(result, point1_orig, point2_orig, params)

    def revert_manual_correction(self, result):
        from ZebrafishAnalysisLib.logic import revert_manual_correction
        return revert_manual_correction(result)
