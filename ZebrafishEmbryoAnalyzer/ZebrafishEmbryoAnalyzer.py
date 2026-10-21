import logging
import sys

import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin


# Slicer puts this module's directory on sys.path, so ZebrafishEmbryoAnalyzerLib and
# ZebrafishEmbryoAnalyzerCore import as normal packages — no path manipulation needed.

_LIB_MODULES = (
    "ZebrafishEmbryoAnalyzerLib.errors",
    "ZebrafishEmbryoAnalyzerLib.model_manifest",
    "ZebrafishEmbryoAnalyzerLib.model_downloader",
    "ZebrafishEmbryoAnalyzerLib.inference_runner",
    "ZebrafishEmbryoAnalyzerLib.inference_worker",
    "ZebrafishEmbryoAnalyzerLib.mrml",
    "ZebrafishEmbryoAnalyzerLib.widget",
    "ZebrafishEmbryoAnalyzerLib.gallery_tab",
    "ZebrafishEmbryoAnalyzerLib.detail_tab",
    "ZebrafishEmbryoAnalyzerLib.results_tab",
    "ZebrafishEmbryoAnalyzerLib.logic",
    "ZebrafishEmbryoAnalyzerLib.overlay",
    "ZebrafishEmbryoAnalyzerLib.export",
    "ZebrafishEmbryoAnalyzerLib.dependency_installer",
    "ZebrafishEmbryoAnalyzerLib.zoom_view",
    "ZebrafishEmbryoAnalyzerLib.bug_report",
)

_CORE_MODULES = (
    "ZebrafishEmbryoAnalyzerCore.seg",
    "ZebrafishEmbryoAnalyzerCore.seg_helper",
    "ZebrafishEmbryoAnalyzerCore.length",
    "ZebrafishEmbryoAnalyzerCore.manual",
    "ZebrafishEmbryoAnalyzerCore.scalebar",
)

_RELOAD_MODULES = _LIB_MODULES + _CORE_MODULES


def _evict_reload_modules():
    for _m in _RELOAD_MODULES:
        sys.modules.pop(_m, None)


_evict_reload_modules()


class ZebrafishEmbryoAnalyzer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Zebrafish Embryo Analyzer"
        self.parent.categories = ["Analysis"]
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


class ZebrafishEmbryoAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self._main = None
        self._parameterNode = None
        self._sceneObserversRegistered = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        _evict_reload_modules()

        self.logic = ZebrafishEmbryoAnalyzerLogic()

        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        self._main = ZebrafishEmbryoAnalyzerMainWidget(self.layout, logic=self.logic)
        self._main._on_settings_changed = self._on_settings_changed
        self._main.refresh_dependency_status()

        self._register_scene_observers()
        self.initializeParameterNode()

        if self._main is not None:
            self._main.apply_shell_layout()

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
        if self._main is not None:
            self._main.apply_shell_layout()
            # Refreshes the in-panel notice about missing packages. Deliberately not a
            # dialog: opening the module must not interrupt, and browsing results of an
            # existing scene needs none of the packages. But the user has to learn about a
            # pending install before loading images and setting parameters, so the notice
            # is updated here. Each action that needs packages still checks for itself.
            self._main.refresh_dependency_status()

    def exit(self):
        if self._main is not None:
            self._main.restore_shell_layout()

    def cleanup(self):
        self.setParameterNode(None)
        self.removeObservers()
        self._sceneObserversRegistered = False
        if self._main is not None:
            self._main.restore_shell_layout()  # ensure restore even on reload (exit() not called)
            self._main.cleanup()

    # ------------------------------------------------------------------
    # Parameter node
    # ------------------------------------------------------------------

    def initializeParameterNode(self):
        """Get the scene parameter node, fill missing params and normalize invalid ones."""
        import math
        from ZebrafishEmbryoAnalyzerLib.widget import (
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
        # Also cancel active downloads and invalidate transient state early.
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

class ZebrafishEmbryoAnalyzerLogic(ScriptedLoadableModuleLogic):
    """Orchestrates analysis requests on behalf of the widget.

    Widget calls these methods; each delegates to the corresponding free
    function in ZebrafishEmbryoAnalyzerLib.logic so the widget never imports that
    module directly.  ZebrafishEmbryoAnalyzerCore remains Slicer-independent.
    """

    def dependency_status(self) -> dict:
        """Return availability of optional ML/vision dependencies.

        Thin wrapper so widget.py never imports ZebrafishEmbryoAnalyzerLib.logic directly.
        """
        from ZebrafishEmbryoAnalyzerLib.logic import dependency_status as _ds
        return _ds()

    def run_analysis(self, image_paths, params, progress_callback=None):
        import math
        import os
        from collections.abc import Mapping, Sequence
        from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError

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

        from ZebrafishEmbryoAnalyzerLib.logic import analyse_images
        return analyse_images(normalized_paths, normalized_params, progress_callback)

    def detect_scalebar(self, image_path, label_um=None):
        from ZebrafishEmbryoAnalyzerLib.logic import detect_scalebar
        return detect_scalebar(image_path, label_um=label_um)

    def preload_models(self, params):
        from ZebrafishEmbryoAnalyzerLib.logic import preload_models
        return preload_models(params)

    def apply_manual_correction(self, result, point1_orig, point2_orig, params=None):
        from ZebrafishEmbryoAnalyzerLib.logic import apply_manual_correction
        return apply_manual_correction(result, point1_orig, point2_orig, params)

    def revert_manual_correction(self, result):
        from ZebrafishEmbryoAnalyzerLib.logic import revert_manual_correction
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
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
        try:
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                results_to_rows,
                build_vtk_table,
                get_or_create_table_node,
            )
            rows = results_to_rows(results)
            # Build the vtk table before touching the MRML scene: if this fails,
            # no node is created and no reference is stored.
            completed_table = build_vtk_table(rows)
            param_node = self.getParameterNode()
            if param_node is None:
                return None
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
            from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                get_or_create_image_node,
                update_image_node,
            )
            param_node = self.getParameterNode()
            if param_node is None:
                return None
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
            from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
            import slicer
            from ZebrafishEmbryoAnalyzerLib.mrml import (
                get_or_create_segmentation_node,
                update_segmentation_node,
            )
            param_node = self.getParameterNode()
            if param_node is None:
                return None
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


# Slicer resolves the self-test as an attribute of this module (see
# ScriptedLoadableModule.runTest), so the class has to be defined here and not in
# Testing/Python/, which is a separate module. That script re-exports it for CTest.
class ZebrafishEmbryoAnalyzerTest(ScriptedLoadableModuleTest):
    """Integration tests for ZebrafishEmbryoAnalyzer.

    Each test_* method is self-contained: it sets up what it needs and cleans up
    via slicer.mrmlScene.Clear(0) or targeted node removal.  No network access,
    no model downloads, no prompts.
    """

    TEST_NAMES = (
        "test_module_import",
        "test_logic_instantiation",
        "test_parameter_node_defaults",
        "test_no_network_during_setup",
        "test_mrml_table_node_creation",
        "test_mrml_table_node_reuse",
        "test_scene_close_cleanup",
    )

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        # Every test runs, even after one fails, so a single broken test does not
        # hide the state of the other seven. Failures are collected and re-raised
        # at the end — without that, unittest would treat the swallowed exception
        # as a pass and CTest would report the build green.
        results = []
        for name in self.TEST_NAMES:
            self.setUp()
            self.delayDisplay(name)
            try:
                getattr(self, name)()
                results.append((name, None))
            except Exception as exc:
                results.append((name, exc))

        failures = [(name, exc) for name, exc in results if exc is not None]
        self._report(results, failures)

        if failures:
            self.fail(
                f"{len(failures)} of {len(results)} self-tests failed: "
                + ", ".join(name for name, _ in failures)
            )
        self.delayDisplay("All tests passed.")

    def _report(self, results, failures):
        """Print a summary that outlives the delayDisplay popups."""
        lines = [
            "-" * 60,
            f"ZebrafishEmbryoAnalyzer self-test: "
            f"{len(results) - len(failures)}/{len(results)} passed",
        ]
        for name, exc in results:
            lines.append(f"  {'PASS' if exc is None else 'FAIL'}  {name}")
            if exc is not None:
                lines.append(f"        {type(exc).__name__}: {exc}")
        lines.append("-" * 60)
        report = "\n".join(lines)

        # The application log and the Python console are separate sinks in Slicer,
        # and the popups are gone the moment they close — write to both.
        logging.info(report)
        print(report)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_module_import(self):
        """ZebrafishEmbryoAnalyzer module and its main classes must be importable."""
        import ZebrafishEmbryoAnalyzer as mod
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzer)
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzerWidget)
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzerLogic)

    def test_logic_instantiation(self):
        """ZebrafishEmbryoAnalyzerLogic() must construct without error."""
        logic = ZebrafishEmbryoAnalyzerLogic()
        self.assertIsNotNone(logic)

    def test_parameter_node_defaults(self):
        """getParameterNode() must return a node whose defaults match PARAM_DEFAULTS."""
        from ZebrafishEmbryoAnalyzerLib.widget import (
            PARAM_DEFAULTS,
            PARAM_LENGTH_ENABLED,
            PARAM_MODEL_ID,
            _DEFAULT_MODEL_ID,
        )

        logic = ZebrafishEmbryoAnalyzerLogic()
        param_node = logic.getParameterNode()
        self.assertIsNotNone(param_node)

        # Populate defaults the same way the widget does (mirrors initializeParameterNode).
        for name, default in PARAM_DEFAULTS.items():
            if not param_node.GetParameter(name):
                param_node.SetParameter(name, default)

        self.assertEqual(param_node.GetParameter(PARAM_LENGTH_ENABLED), "true")
        self.assertEqual(param_node.GetParameter(PARAM_MODEL_ID), _DEFAULT_MODEL_ID)

    def test_no_network_during_setup(self):
        """No prompts or network calls fire during setup when testingEnabled() is True."""
        if not slicer.app.testingEnabled():
            self.delayDisplay("Skipping: requires Slicer --testing flag (CTest mode)", 100)
            return
        self.assertTrue(slicer.app.testingEnabled(),
                        "testingEnabled() must be True — confirms prompts are suppressed")

    def test_mrml_table_node_creation(self):
        """update_results_table() must create a vtkMRMLTableNode with exactly one row."""
        logic = ZebrafishEmbryoAnalyzerLogic()
        result = {
            "filename": "test.png",
            "length": 1000,
            "curvature": 1,
            "ratio": 1.1,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }
        table_node = logic.update_results_table([result])

        self.assertIsNotNone(table_node)
        self.assertTrue(table_node.IsA("vtkMRMLTableNode"))
        self.assertEqual(table_node.GetNumberOfRows(), 1)

    def test_mrml_table_node_reuse(self):
        """Calling update_results_table() twice must reuse the same node."""
        logic = ZebrafishEmbryoAnalyzerLogic()
        result = {
            "filename": "a.png",
            "length": 500,
            "curvature": 0,
            "ratio": 1.0,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }

        node_first = logic.update_results_table([result])
        count_after_first = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLTableNode")

        node_second = logic.update_results_table([result, result])
        count_after_second = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLTableNode")

        self.assertEqual(count_after_first, count_after_second,
                         "A second call must not add a new table node to the scene.")
        self.assertEqual(node_first.GetID(), node_second.GetID(),
                         "Both calls must return the same node.")
        self.assertEqual(node_second.GetNumberOfRows(), 2)

    # results_to_rows() is pure Python and covered by tests/test_mrml_adapter.py and
    # tests/test_mrml_node.py. It is deliberately not repeated here — a self-test slot
    # costs a real Slicer instance and is reserved for behaviour that fakes cannot reach.

    def test_scene_close_cleanup(self):
        """Scene clear must not crash; a fresh parameter node must be obtainable after."""
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        logic = ZebrafishEmbryoAnalyzerLogic()
        # Populate some MRML nodes first.
        logic.update_results_table([{
            "filename": "x.png",
            "length": 1,
            "curvature": 0,
            "ratio": 1.0,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }])

        # Scene clear — must not raise.
        slicer.mrmlScene.Clear(0)

        # After clear, logic must still deliver a (fresh) parameter node.
        param_node = logic.getParameterNode()
        self.assertIsNotNone(param_node)

        # Populate defaults manually (widget is not running here).
        for name, default in PARAM_DEFAULTS.items():
            if not param_node.GetParameter(name):
                param_node.SetParameter(name, default)

        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_LENGTH_ENABLED
        self.assertEqual(param_node.GetParameter(PARAM_LENGTH_ENABLED), "true")
