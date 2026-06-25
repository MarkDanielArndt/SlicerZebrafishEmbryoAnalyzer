import sys

import qt
import vtk
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
    "ZebrafishAnalysisLib.mrml",
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
        self._parameterNode = None
        self._sceneObserversRegistered = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _evict_lib_modules()

        self.logic = ZebrafishAnalysisLogic()

        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget
        self._main = ZebrafishAnalysisMainWidget(self.layout, logic=self.logic)
        self._main._on_settings_changed = self._on_settings_changed
        self._main.refresh_dependency_status()

        if hasattr(self, "_dep_check_timer"):
            self._dep_check_timer.stop()
        self._dep_check_timer = qt.QTimer()
        self._dep_check_timer.setSingleShot(True)
        self._dep_check_timer.setInterval(200)
        self._dep_check_timer.timeout.connect(self._check_deps_on_start)
        self._dep_check_timer.start()

        self._register_scene_observers()
        self.initializeParameterNode()

        if hasattr(self, "_prewarm_timer"):
            self._prewarm_timer.stop()
        self._prewarm_timer = qt.QTimer()
        self._prewarm_timer.setSingleShot(True)
        self._prewarm_timer.setInterval(500)
        self._prewarm_timer.timeout.connect(self._prewarm_imports)
        self._prewarm_timer.start()

    def _register_scene_observers(self):
        """Register MRML scene observers exactly once per setup."""
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
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndImportEvent,
            self._on_scene_end_import,
        )
        self._sceneObserversRegistered = True

    def enter(self):
        if hasattr(self, "logic"):
            self.initializeParameterNode()

    def exit(self):
        pass

    def _check_deps_on_start(self):
        if self._main is not None:
            self._main.prompt_install_if_missing()

    def cleanup(self):
        self.setParameterNode(None)
        if hasattr(self, "_prewarm_timer"):
            self._prewarm_timer.stop()
        if hasattr(self, "_dep_check_timer"):
            self._dep_check_timer.stop()
        self.removeObservers()
        self._sceneObserversRegistered = False
        if self._main is not None:
            self._main.cleanup()

    # ------------------------------------------------------------------
    # Parameter node
    # ------------------------------------------------------------------

    def initializeParameterNode(self):
        """Get the scene parameter node, fill missing params and normalize invalid ones."""
        import math
        from ZebrafishAnalysisLib.widget import (
            PARAM_DEFAULTS, _MODEL_BY_ID, _DEFAULT_MODEL_ID,
            PARAM_LENGTH_ENABLED, PARAM_CURVATURE_ENABLED, PARAM_RATIO_ENABLED,
            PARAM_EYES_ENABLED, PARAM_CONFIDENCE_THRESHOLD_ENABLED,
            PARAM_CONFIDENCE_THRESHOLD, PARAM_UM_PER_PX, PARAM_MODEL_ID,
        )
        node = self.logic.getParameterNode()
        wasModified = node.StartModify()
        try:
            for name, default in PARAM_DEFAULTS.items():
                if not node.GetParameter(name):
                    node.SetParameter(name, default)
            for key in (PARAM_LENGTH_ENABLED, PARAM_CURVATURE_ENABLED, PARAM_RATIO_ENABLED,
                        PARAM_EYES_ENABLED, PARAM_CONFIDENCE_THRESHOLD_ENABLED):
                if node.GetParameter(key) not in ("true", "false"):
                    node.SetParameter(key, PARAM_DEFAULTS[key])
            v = node.GetParameter(PARAM_CONFIDENCE_THRESHOLD)
            try:
                f = float(v)
                if not (math.isfinite(f) and 0.0 <= f <= 1.0):
                    raise ValueError
            except (ValueError, TypeError):
                node.SetParameter(PARAM_CONFIDENCE_THRESHOLD, PARAM_DEFAULTS[PARAM_CONFIDENCE_THRESHOLD])
            v = node.GetParameter(PARAM_UM_PER_PX)
            try:
                f = float(v)
                if not (math.isfinite(f) and 0.001 <= f <= 9999.0):
                    raise ValueError
            except (ValueError, TypeError):
                node.SetParameter(PARAM_UM_PER_PX, PARAM_DEFAULTS[PARAM_UM_PER_PX])
            if node.GetParameter(PARAM_MODEL_ID) not in _MODEL_BY_ID:
                node.SetParameter(PARAM_MODEL_ID, _DEFAULT_MODEL_ID)
        finally:
            node.EndModify(wasModified)
        # Compute before setParameterNode so we can detect early return for same node.
        same_node = node is self._parameterNode
        self.setParameterNode(node)
        # setParameterNode early-returns for the same node (no GUI update inside).
        # Always update here so enter() re-entering is reflected in the UI.
        if same_node and self._main is not None:
            self._main.updateGUIFromParameterNode(node)

    def setParameterNode(self, node):
        """Connect to a new parameter node and disconnect from the old one."""
        if node is self._parameterNode:
            return
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._on_parameter_node_modified,
            )
        self._parameterNode = node
        if node is not None:
            self.addObserver(
                node,
                vtk.vtkCommand.ModifiedEvent,
                self._on_parameter_node_modified,
            )
            if self._main is not None:
                self._main.updateGUIFromParameterNode(node)

    def _on_parameter_node_modified(self, caller=None, event=None):
        if self._main is not None:
            self._main.updateGUIFromParameterNode(self._parameterNode)

    def _on_settings_changed(self):
        if self._parameterNode is not None and self._main is not None:
            self._main.updateParameterNodeFromGUI(self._parameterNode)

    # ------------------------------------------------------------------
    # Scene events
    # ------------------------------------------------------------------

    def _on_scene_start_close(self, caller=None, event=None):
        # Disconnect parameter node before scene objects are destroyed.
        # Also invalidate background workers early.
        self.setParameterNode(None)
        if self._main is not None:
            self._main._cancel_workers()

    def _on_scene_end_close(self, caller=None, event=None):
        # Reset session UI state, then connect to the fresh scene's parameter node.
        if self._main is not None:
            self._main.reset_for_scene_close()
        self.initializeParameterNode()

    def _on_scene_end_import(self, caller=None, event=None):
        # Pick up parameter node values from the newly loaded scene.
        self.initializeParameterNode()

    def _prewarm_imports(self):
        from ZebrafishAnalysisLib.dependency_installer import get_missing_packages
        if get_missing_packages()["torch"]:
            return  # torch absent — skip thread to avoid ImportError log noise
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

    def dependency_status(self) -> dict:
        """Return availability of optional ML/vision dependencies.

        Thin wrapper so widget.py never imports ZebrafishAnalysisLib.logic directly.
        """
        from ZebrafishAnalysisLib.logic import dependency_status as _ds
        return _ds()

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

    def update_results_table(self, results):
        """Create or update the MRML table node with raw analysis results.

        Separate from run_analysis() so that a table update failure cannot
        discard a successful analysis.  Raises MRMLAdapterError on failure;
        the existing table content is preserved when possible.

        Returns
        -------
        vtkMRMLTableNode
        """
        from ZebrafishAnalysisLib.errors import MRMLAdapterError
        try:
            import slicer
            from ZebrafishAnalysisLib.mrml import (
                results_to_rows,
                build_vtk_table,
                get_or_create_table_node,
            )
            rows = results_to_rows(results)
            # Build the vtk table before touching the MRML scene: if this fails,
            # no node is created and no reference is stored.
            completed_table = build_vtk_table(rows)
            param_node = self.getParameterNode()
            table_node = get_or_create_table_node(param_node, slicer.mrmlScene)
            table_node.SetAndObserveTable(completed_table)
            return table_node
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update results table: {exc}"
            ) from exc

    def update_current_image_node(self, result, um_per_px):
        """Create or update the MRML vector volume node for the current image.

        Returns None silently if result["original"] is None (stub or error row).
        Raises MRMLAdapterError on MRML or VTK failure.
        Must be called on the Slicer main thread only.
        Does not use result["spacing"] — that is calibrated to 256x256 mask space.
        """
        original = result.get("original") if result else None
        if original is None:
            return None
        try:
            from ZebrafishAnalysisLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishAnalysisLib.mrml import (
                get_or_create_image_node,
                update_image_node,
            )
            param_node = self.getParameterNode()
            node = get_or_create_image_node(param_node, slicer.mrmlScene)
            update_image_node(original, um_per_px, node)
            return node
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update current image node: {exc}"
            ) from exc

    def update_current_segmentation_node(self, result, um_per_px):
        """Create or update the MRML segmentation node for the current image's masks.

        Returns None silently if result["original"] is None (stub or error row).
        Raises MRMLAdapterError on MRML or VTK failure.
        Must be called on the Slicer main thread only.
        """
        original = result.get("original") if result else None
        if original is None:
            return None
        try:
            from ZebrafishAnalysisLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishAnalysisLib.mrml import (
                get_or_create_segmentation_node,
                update_segmentation_node,
            )
            param_node = self.getParameterNode()
            image_node = param_node.GetNodeReference("CurrentImage")
            node = get_or_create_segmentation_node(param_node, slicer.mrmlScene)
            update_segmentation_node(result, um_per_px, node, image_node=image_node)
            return node
        except MRMLAdapterError:
            raise
        except Exception as exc:
            raise MRMLAdapterError(
                f"Failed to update current segmentation node: {exc}"
            ) from exc
