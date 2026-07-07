"""
Tests for parameter node integration (Task D2).

Behavioral tests use fake/stub objects to avoid a real Slicer runtime.
Source-level tests verify structural contracts only where subprocess testing
is impractical.
"""

import ast
import os
import re
import subprocess
import sys
import textwrap

import pytest

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzer.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "widget.py")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

# Stub for tests that import ZebrafishEmbryoAnalyzerWidget (needs vtk + tracking mixin)
_WIDGET_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

class _BaseWidget:
    pass

class _TrackingMixin:
    def __init__(self):
        self._obs = []

    def addObserver(self, node, event, method):
        self._obs.append((id(node), event))

    def removeObserver(self, node, event, method=None):
        key = (id(node), event)
        if key in self._obs:
            self._obs.remove(key)

    def removeObservers(self, method=None):
        self._obs.clear()

    def hasObserver(self, node, event, method):
        return (id(node), event) in self._obs

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_BaseWidget,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=_TrackingMixin,
)

_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""

# Stub for tests that import ZebrafishEmbryoAnalyzerMainWidget only (no vtk needed)
_MAIN_WIDGET_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()
sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=object,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(VTKObservationMixin=object)
sys.modules["vtk"] = MagicMock()
"""

# Inline fake parameter node used by all behavioral tests
_FAKE_NODE = """\
class _FakeNode:
    def __init__(self, params=None):
        self._p = dict(params or {})
        self.modified_count = 0

    def GetParameter(self, name):
        return self._p.get(name, "")

    def SetParameter(self, name, value):
        self._p[name] = value
        self.modified_count += 1

    def StartModify(self):
        return 0

    def EndModify(self, state):
        pass
"""


def _run(stub: str, code: str) -> subprocess.CompletedProcess:
    full = stub + _FAKE_NODE + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# Source-level structural tests
# ---------------------------------------------------------------------------

def _src(path):
    return open(path).read()


def test_param_constants_defined_in_widget():
    """PARAM_DEFAULTS and all PARAM_* constants must be defined in widget.py."""
    src = _src(_WIDGET_PY)
    for name in ("PARAM_DEFAULTS", "PARAM_LENGTH_ENABLED", "PARAM_CURVATURE_ENABLED",
                 "PARAM_RATIO_ENABLED", "PARAM_EYES_ENABLED",
                 "PARAM_CONFIDENCE_THRESHOLD_ENABLED", "PARAM_CONFIDENCE_THRESHOLD",
                 "PARAM_UM_PER_PX", "PARAM_MODEL_ID"):
        assert name in src, f"{name} not found in widget.py"


def test_model_entries_uses_stable_ids():
    """ComboBox must store string model IDs ('general', 'desy'), not data tuples."""
    src = _src(_WIDGET_PY)
    assert "_MODEL_ENTRIES" in src, "_MODEL_ENTRIES not defined in widget.py"
    assert "_MODEL_BY_ID" in src, "_MODEL_BY_ID not defined in widget.py"
    assert '"general"' in src, "model ID 'general' not found in widget.py"
    assert '"desy"' in src, "model ID 'desy' not found in widget.py"


def test_widget_has_update_gui_method():
    src = _src(_WIDGET_PY)
    assert "def updateGUIFromParameterNode" in src


def test_widget_has_update_node_method():
    src = _src(_WIDGET_PY)
    assert "def updateParameterNodeFromGUI" in src


def test_main_py_imports_vtk():
    src = _src(_MAIN_PY)
    assert re.search(r"^import\s+vtk", src, re.M), "ZebrafishEmbryoAnalyzer.py must import vtk"


def test_main_py_has_initialize_parameter_node():
    src = _src(_MAIN_PY)
    assert "def initializeParameterNode" in src


def test_main_py_has_set_parameter_node():
    src = _src(_MAIN_PY)
    assert "def setParameterNode" in src


def test_no_numpy_array_in_param_defaults():
    """PARAM_DEFAULTS must not reference numpy, torch, cv2, or result dicts."""
    src = _src(_WIDGET_PY)
    # Find the PARAM_DEFAULTS block
    m = re.search(r"PARAM_DEFAULTS\s*=\s*\{([^}]*)\}", src, re.S)
    assert m, "PARAM_DEFAULTS not found or not a simple dict literal"
    block = m.group(1)
    for banned in ("numpy", "torch", "cv2", "ndarray", "result", "image_path"):
        assert banned not in block, f"PARAM_DEFAULTS must not reference {banned!r}"


# ---------------------------------------------------------------------------
# Behavioral: PARAM_DEFAULTS values
# ---------------------------------------------------------------------------

def test_param_defaults_match_ui_defaults():
    """PARAM_DEFAULTS string values must match the actual widget control defaults."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        assert PARAM_DEFAULTS.get("lengthEnabled")               == "true",  PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("curvatureEnabled")            == "true",  PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("ratioEnabled")                == "true",  PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("eyesEnabled")                 == "false", PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("confidenceThresholdEnabled")  == "false", PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("confidenceThreshold")         == "0.85",  PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("micrometersPerPixel")         == "22.99", PARAM_DEFAULTS
        assert PARAM_DEFAULTS.get("selectedModelId")             == "general", PARAM_DEFAULTS
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_param_defaults_has_exactly_eight_keys():
    """PARAM_DEFAULTS must have exactly the eight expected keys — no extras."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        expected = {
            "lengthEnabled", "curvatureEnabled", "ratioEnabled",
            "eyesEnabled", "confidenceThresholdEnabled",
            "confidenceThreshold", "micrometersPerPixel", "selectedModelId",
        }
        assert set(PARAM_DEFAULTS.keys()) == expected, (
            f"Unexpected keys: {set(PARAM_DEFAULTS.keys()) ^ expected}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: updateGUIFromParameterNode
# ---------------------------------------------------------------------------

def test_update_gui_applies_bool_and_float_values():
    """updateGUIFromParameterNode sets all controls from non-default stored values."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget, _MODEL_BY_ID

        node = _FakeNode({
            "lengthEnabled":               "false",
            "curvatureEnabled":            "true",
            "ratioEnabled":                "false",
            "eyesEnabled":                 "true",
            "confidenceThresholdEnabled":  "true",
            "confidenceThreshold":         "0.75",
            "micrometersPerPixel":         "12.5",
            "selectedModelId":             "desy",
        })

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)

        w._chk_length.setChecked.assert_called_with(False)
        w._chk_curvature.setChecked.assert_called_with(True)
        w._chk_ratio.setChecked.assert_called_with(False)
        w._chk_eyes.setChecked.assert_called_with(True)
        w._chk_hitl.setChecked.assert_called_with(True)
        assert w._threshold_slider.value == 0.75,  w._threshold_slider.value
        assert w._um_per_px.value == 12.5,         w._um_per_px.value
        w._model_combo.setCurrentIndex.assert_called_with(1)  # index of "desy"

        assert not w._updatingGUIFromParameterNode, "Guard not reset after call"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_unknown_model_id_falls_back_to_first():
    """updateGUIFromParameterNode falls back to index 0 for an unknown model ID."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode({"selectedModelId": "unknown_model_xyz"})

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)

        # Unknown ID → setCurrentIndex(0) (first / default entry)
        w._model_combo.setCurrentIndex.assert_called_with(0)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_missing_params_use_defaults():
    """updateGUIFromParameterNode uses Python defaults for parameters not in the node."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode()  # empty node

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)

        w._chk_length.setChecked.assert_called_with(True)
        w._chk_eyes.setChecked.assert_called_with(False)
        assert w._threshold_slider.value == 0.85, w._threshold_slider.value
        assert w._um_per_px.value == 22.99,       w._um_per_px.value
        w._model_combo.setCurrentIndex.assert_called_with(0)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_none_node_is_safe():
    """updateGUIFromParameterNode(None) must not raise."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        # No controls needed — must return early without touching anything.
        w.updateGUIFromParameterNode(None)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: updateParameterNodeFromGUI
# ---------------------------------------------------------------------------

def test_update_node_from_gui_writes_all_params():
    """updateParameterNodeFromGUI must write all setting control values to the node."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode()

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = False
        w._chk_curvature = MagicMock(); w._chk_curvature.isChecked.return_value = True
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = False
        w._chk_eyes      = MagicMock(); w._chk_eyes.isChecked.return_value      = True
        w._chk_hitl      = MagicMock(); w._chk_hitl.isChecked.return_value      = True
        w._threshold_slider = MagicMock(); w._threshold_slider.value = 0.6
        w._um_per_px     = MagicMock(); w._um_per_px.value = 10.0
        w._model_combo   = MagicMock(); w._model_combo.currentData = "desy"

        w.updateParameterNodeFromGUI(node)

        assert node.GetParameter("lengthEnabled")               == "false",  node._p
        assert node.GetParameter("curvatureEnabled")            == "true",   node._p
        assert node.GetParameter("ratioEnabled")                == "false",  node._p
        assert node.GetParameter("eyesEnabled")                 == "true",   node._p
        assert node.GetParameter("confidenceThresholdEnabled")  == "true",   node._p
        assert node.GetParameter("confidenceThreshold")         == "0.6",    node._p
        assert node.GetParameter("micrometersPerPixel")         == "10.0",   node._p
        assert node.GetParameter("selectedModelId")             == "desy",   node._p
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_node_from_gui_invalid_model_id_falls_back():
    """updateParameterNodeFromGUI stores 'general' if currentData is not a known model ID."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode()

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = True
        w._chk_curvature = MagicMock(); w._chk_curvature.isChecked.return_value = True
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = True
        w._chk_eyes      = MagicMock(); w._chk_eyes.isChecked.return_value      = False
        w._chk_hitl      = MagicMock(); w._chk_hitl.isChecked.return_value      = False
        w._threshold_slider = MagicMock(); w._threshold_slider.value = 0.85
        w._um_per_px     = MagicMock(); w._um_per_px.value = 22.99
        w._model_combo   = MagicMock(); w._model_combo.currentData = "unknown_xyz"

        w.updateParameterNodeFromGUI(node)

        assert node.GetParameter("selectedModelId") == "general", node.GetParameter("selectedModelId")
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_node_guarded_during_gui_update():
    """updateParameterNodeFromGUI must be a no-op when _updatingGUIFromParameterNode is True."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode()

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = True  # guard active
        w._chk_length = MagicMock()
        w._chk_length.isChecked.return_value = False

        w.updateParameterNodeFromGUI(node)

        # Nothing must have been written
        assert node.GetParameter("lengthEnabled") == "", node._p
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_node_none_is_safe():
    """updateParameterNodeFromGUI(None) must not raise."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w.updateParameterNodeFromGUI(None)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: round-trip
# ---------------------------------------------------------------------------

def test_round_trip_preserves_all_values():
    """Write controls → node → read back into new controls: values must be preserved."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode()

        # ── Write phase ──────────────────────────────────────────────────────
        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = False
        w._chk_curvature = MagicMock(); w._chk_curvature.isChecked.return_value = True
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = True
        w._chk_eyes      = MagicMock(); w._chk_eyes.isChecked.return_value      = True
        w._chk_hitl      = MagicMock(); w._chk_hitl.isChecked.return_value      = False
        w._threshold_slider = MagicMock(); w._threshold_slider.value = 0.42
        w._um_per_px     = MagicMock(); w._um_per_px.value = 5.1234
        w._model_combo   = MagicMock(); w._model_combo.currentData = "desy"

        w.updateParameterNodeFromGUI(node)

        # ── Read phase ───────────────────────────────────────────────────────
        w2 = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w2._updatingGUIFromParameterNode = False
        w2._chk_length    = MagicMock()
        w2._chk_curvature = MagicMock()
        w2._chk_ratio     = MagicMock()
        w2._chk_eyes      = MagicMock()
        w2._chk_hitl      = MagicMock()
        w2._threshold_slider = MagicMock()
        w2._um_per_px     = MagicMock()
        w2._model_combo   = MagicMock()
        w2._model_combo.count = 2
        w2._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w2.updateGUIFromParameterNode(node)

        w2._chk_length.setChecked.assert_called_with(False)
        w2._chk_curvature.setChecked.assert_called_with(True)
        w2._chk_eyes.setChecked.assert_called_with(True)
        assert w2._threshold_slider.value == 0.42,  w2._threshold_slider.value
        assert w2._um_per_px.value == 5.1234,       w2._um_per_px.value
        w2._model_combo.setCurrentIndex.assert_called_with(1)  # desy = index 1
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: setParameterNode observer lifecycle
# ---------------------------------------------------------------------------

def test_set_parameter_node_adds_modified_observer():
    """setParameterNode(node) must add a ModifiedEvent observer on the node."""
    r = _run(_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None

        class _FakeParamNode:
            pass

        node = _FakeParamNode()
        w.setParameterNode(node)

        assert (id(node), 33) in w._obs, f"ModifiedEvent observer not added: {w._obs}"
        assert w._parameterNode is node
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_parameter_node_removes_old_observer():
    """setParameterNode(new) must remove the ModifiedEvent observer from the old node."""
    r = _run(_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None

        class _N: pass

        n1, n2 = _N(), _N()
        w.setParameterNode(n1)
        assert (id(n1), 33) in w._obs

        w.setParameterNode(n2)
        assert (id(n1), 33) not in w._obs, "Old observer not removed"
        assert (id(n2), 33) in w._obs,     "New observer not added"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_parameter_node_none_removes_observer():
    """setParameterNode(None) must remove the ModifiedEvent observer from the current node."""
    r = _run(_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None

        class _N: pass
        node = _N()
        w.setParameterNode(node)
        assert (id(node), 33) in w._obs

        w.setParameterNode(None)
        assert (id(node), 33) not in w._obs, "Observer not removed on None"
        assert w._parameterNode is None
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_set_parameter_node_repeated_same_node_no_duplicate():
    """setParameterNode called twice with the same node must not add duplicate observers."""
    r = _run(_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None

        class _N: pass
        node = _N()

        w.setParameterNode(node)
        count_first = len([e for e in w._obs if e == (id(node), 33)])

        w.setParameterNode(node)  # same node again
        count_second = len([e for e in w._obs if e == (id(node), 33)])

        assert count_first  == 1, f"Expected 1 after first call, got {count_first}"
        assert count_second == 1, f"Expected 1 after second call, got {count_second} (duplicate!)"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cleanup_disconnects_parameter_node():
    """cleanup() must remove the parameter node ModifiedEvent observer."""
    r = _run(_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None
        w._sceneObserversRegistered = False

        class _N: pass
        node = _N()

        # Simulate having connected to a node
        w.setParameterNode(node)
        assert (id(node), 33) in w._obs

        # cleanup must disconnect it
        w.cleanup()
        assert (id(node), 33) not in w._obs, "Parameter node observer not removed by cleanup"
        assert w._parameterNode is None
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: no MRML node refs, no heavy objects in parameter node
# ---------------------------------------------------------------------------

def test_no_mrml_refs_in_initialize_parameter_node():
    """initializeParameterNode must not introduce vtkMRML node references."""
    src = open(_MAIN_PY).read()
    # Find the initializeParameterNode method body
    tree = ast.parse(src)
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ZebrafishEmbryoAnalyzerWidget":
            for fn in ast.walk(cls):
                if isinstance(fn, ast.FunctionDef) and fn.name == "initializeParameterNode":
                    lines = src.splitlines()
                    body = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
                    refs = re.findall(r'\bvtkMRML\w+\b', body)
                    assert not refs, f"initializeParameterNode has MRML node refs: {refs}"
                    return
    pytest.fail("initializeParameterNode not found in ZebrafishEmbryoAnalyzerWidget")


def test_no_result_dicts_or_paths_in_defaults():
    """PARAM_DEFAULTS values must all be plain strings, not lists or dicts."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        for k, v in PARAM_DEFAULTS.items():
            assert isinstance(v, str), f"PARAM_DEFAULTS[{k!r}] is not a string: {v!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_notify_settings_changed_calls_callback():
    """_notify_settings_changed must call _on_settings_changed when not updating from node."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        called = []

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._on_settings_changed = lambda: called.append(1)

        w._notify_settings_changed()

        assert called == [1], f"Callback not called: {called}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_notify_settings_changed_suppressed_during_gui_update():
    """_notify_settings_changed must not call the callback when _updatingGUIFromParameterNode."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        called = []

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = True  # guard active
        w._on_settings_changed = lambda: called.append(1)

        w._notify_settings_changed()

        assert called == [], f"Callback must not fire during GUI update: {called}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: initializeParameterNode
# ---------------------------------------------------------------------------

def test_initialize_parameter_node_empty_node_sets_all_defaults():
    """initializeParameterNode must populate all 8 parameters on an empty node."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        node = _FakeNode()
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None
        w.logic = logic

        w.initializeParameterNode()

        for key, expected in PARAM_DEFAULTS.items():
            actual = node.GetParameter(key)
            assert actual == expected, f"{key}: expected {expected!r}, got {actual!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_initialize_parameter_node_does_not_overwrite_valid_values():
    """initializeParameterNode must not replace values that are already valid."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        node = _FakeNode({
            "lengthEnabled":               "false",
            "curvatureEnabled":            "true",
            "ratioEnabled":                "true",
            "eyesEnabled":                 "true",
            "confidenceThresholdEnabled":  "true",
            "confidenceThreshold":         "0.6",
            "micrometersPerPixel":         "50.0",
            "selectedModelId":             "desy",
        })
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None
        w.logic = logic

        w.initializeParameterNode()

        assert node.GetParameter("lengthEnabled")       == "false",   node._p
        assert node.GetParameter("eyesEnabled")         == "true",    node._p
        assert node.GetParameter("selectedModelId")     == "desy",    node._p
        assert node.GetParameter("confidenceThreshold") == "0.6",     node._p
        assert node.GetParameter("micrometersPerPixel") == "50.0",    node._p
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_initialize_parameter_node_idempotent():
    """Calling initializeParameterNode twice yields identical parameter values."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        node = _FakeNode()
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None
        w.logic = logic

        w.initializeParameterNode()
        snap1 = dict(node._p)

        w.initializeParameterNode()
        snap2 = dict(node._p)

        assert snap1 == snap2, f"Values changed on second init: {snap2}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_initialize_parameter_node_normalizes_invalid_values():
    """initializeParameterNode must correct invalid stored values to canonical defaults."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        node = _FakeNode({
            "lengthEnabled":               "yes",       # invalid bool → "true"
            "curvatureEnabled":            "1",          # invalid bool → "true"
            "ratioEnabled":                "true",       # valid → preserved
            "eyesEnabled":                 "false",      # valid → preserved
            "confidenceThresholdEnabled":  "no",         # invalid bool → "false"
            "confidenceThreshold":         "abc",        # invalid float → "0.85"
            "micrometersPerPixel":         "999999.0",   # out-of-range → "22.99"
            "selectedModelId":             "unknown",    # unknown ID → "general"
        })
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = None
        w.logic = logic

        w.initializeParameterNode()

        assert node.GetParameter("lengthEnabled")              == "true",    node._p
        assert node.GetParameter("curvatureEnabled")           == "true",    node._p
        assert node.GetParameter("ratioEnabled")               == "true",    node._p
        assert node.GetParameter("eyesEnabled")                == "false",   node._p
        assert node.GetParameter("confidenceThresholdEnabled") == "false",   node._p
        assert node.GetParameter("confidenceThreshold")        == "0.85",    node._p
        assert node.GetParameter("micrometersPerPixel")        == "22.99",   node._p
        assert node.GetParameter("selectedModelId")            == "general", node._p
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_initialize_parameter_node_gui_updated_on_same_node_reinit():
    """GUI must be updated on every initializeParameterNode call, including same-node re-init."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        node = _FakeNode({
            "lengthEnabled": "true", "curvatureEnabled": "true",
            "ratioEnabled": "true", "eyesEnabled": "false",
            "confidenceThresholdEnabled": "false",
            "confidenceThreshold": "0.85",
            "micrometersPerPixel": "22.99",
            "selectedModelId": "general",
        })
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        update_calls = []

        class _FakeMain:
            _on_settings_changed = None
            def updateGUIFromParameterNode(self, n):
                update_calls.append(n)
            def _cancel_workers(self): pass
            def cleanup(self): pass

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = _FakeMain()
        w.logic = logic

        w.initializeParameterNode()
        count_after_first = len(update_calls)

        w.initializeParameterNode()  # same node → setParameterNode early-returns
        count_after_second = len(update_calls)

        assert count_after_first  >= 1, f"Expected >=1 GUI updates after first init, got {count_after_first}"
        assert count_after_second > count_after_first, (
            f"Expected more GUI updates after second (same-node) init: "
            f"{count_after_second} > {count_after_first}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_no_modified_loop_on_initialize():
    """Repeated initializeParameterNode calls must not cause unbounded GUI updates."""
    r = _run(_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        node = _FakeNode()
        logic = MagicMock()
        logic.getParameterNode.return_value = node

        update_calls = [0]

        class _SafeMain:
            _on_settings_changed = None
            def updateGUIFromParameterNode(self, n):
                update_calls[0] += 1
                if update_calls[0] > 10:
                    raise RuntimeError(
                        f"updateGUIFromParameterNode called too many times: {update_calls[0]}"
                    )
            def _cancel_workers(self): pass
            def cleanup(self): pass

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._parameterNode = None
        w._main = _SafeMain()
        w.logic = logic

        w.initializeParameterNode()
        w.initializeParameterNode()

        assert update_calls[0] <= 10, f"Too many GUI updates: {update_calls[0]}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: robust deserialization in updateGUIFromParameterNode
# ---------------------------------------------------------------------------

def test_update_gui_invalid_bool_string_falls_back():
    """updateGUIFromParameterNode accepts only 'true'/'false'; others fall back to defaults."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode({
            "lengthEnabled":               "yes",    # → default True
            "curvatureEnabled":            "1",       # → default True
            "ratioEnabled":                "True",    # capital T → default True
            "eyesEnabled":                 "FALSE",   # all-caps → default False
            "confidenceThresholdEnabled":  "0",       # → default False
            "confidenceThreshold":         "0.85",
            "micrometersPerPixel":         "22.99",
            "selectedModelId":             "general",
        })

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)

        w._chk_length.setChecked.assert_called_with(True)
        w._chk_curvature.setChecked.assert_called_with(True)
        w._chk_ratio.setChecked.assert_called_with(True)
        w._chk_eyes.setChecked.assert_called_with(False)
        w._chk_hitl.setChecked.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_nan_and_infinity_fall_back():
    """updateGUIFromParameterNode falls back to defaults for NaN and ±Infinity."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        for bad_val in ("nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"):
            node = _FakeNode({
                "lengthEnabled": "true", "curvatureEnabled": "true",
                "ratioEnabled": "true", "eyesEnabled": "false",
                "confidenceThresholdEnabled": "false",
                "confidenceThreshold": bad_val,
                "micrometersPerPixel": bad_val,
                "selectedModelId": "general",
            })

            w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
            w._updatingGUIFromParameterNode = False
            w._chk_length    = MagicMock()
            w._chk_curvature = MagicMock()
            w._chk_ratio     = MagicMock()
            w._chk_eyes      = MagicMock()
            w._chk_hitl      = MagicMock()
            w._threshold_slider = MagicMock()
            w._um_per_px     = MagicMock()
            w._model_combo   = MagicMock()
            w._model_combo.count = 2
            w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

            w.updateGUIFromParameterNode(node)

            assert w._threshold_slider.value == 0.85, f"For {bad_val!r}: {w._threshold_slider.value}"
            assert w._um_per_px.value == 22.99,       f"For {bad_val!r}: {w._um_per_px.value}"

        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_out_of_range_numeric_falls_back():
    """updateGUIFromParameterNode falls back to defaults for out-of-range numeric values."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode({
            "lengthEnabled": "true", "curvatureEnabled": "true",
            "ratioEnabled": "true", "eyesEnabled": "false",
            "confidenceThresholdEnabled": "false",
            "confidenceThreshold": "-0.1",    # below [0.0, 1.0] → 0.85
            "micrometersPerPixel": "10000.0", # above [0.001, 9999.0] → 22.99
            "selectedModelId": "general",
        })

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)

        assert w._threshold_slider.value == 0.85, w._threshold_slider.value
        assert w._um_per_px.value == 22.99,       w._um_per_px.value
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_non_numeric_string_no_exception():
    """updateGUIFromParameterNode must not raise for non-numeric strings in numeric fields."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        node = _FakeNode({
            "lengthEnabled": "true", "curvatureEnabled": "true",
            "ratioEnabled": "true", "eyesEnabled": "false",
            "confidenceThresholdEnabled": "false",
            "confidenceThreshold": "abc",
            "micrometersPerPixel": "hello world",
            "selectedModelId": "general",
        })

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._um_per_px     = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.updateGUIFromParameterNode(node)  # must not raise

        assert w._threshold_slider.value == 0.85, w._threshold_slider.value
        assert w._um_per_px.value == 22.99,       w._um_per_px.value
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Model config contract
# ---------------------------------------------------------------------------

def test_update_node_from_gui_end_modify_called_on_exception():
    """EndModify must be called exactly once even when SetParameter raises."""
    r = _run(_MAIN_WIDGET_STUB, """
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        end_modify_calls = []

        class _BrokenNode:
            def __init__(self):
                self._p = {}
                self._call_count = 0
            def GetParameter(self, name):
                return self._p.get(name, "")
            def SetParameter(self, name, value):
                self._call_count += 1
                if self._call_count == 1:
                    raise RuntimeError("simulated SetParameter failure")
                self._p[name] = value
            def StartModify(self):
                return 42
            def EndModify(self, state):
                end_modify_calls.append(state)

        node = _BrokenNode()

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock(); w._chk_length.isChecked.return_value    = True
        w._chk_curvature = MagicMock(); w._chk_curvature.isChecked.return_value = True
        w._chk_ratio     = MagicMock(); w._chk_ratio.isChecked.return_value     = True
        w._chk_eyes      = MagicMock(); w._chk_eyes.isChecked.return_value      = False
        w._chk_hitl      = MagicMock(); w._chk_hitl.isChecked.return_value      = False
        w._threshold_slider = MagicMock(); w._threshold_slider.value = 0.85
        w._um_per_px     = MagicMock(); w._um_per_px.value = 22.99
        w._model_combo   = MagicMock(); w._model_combo.currentData = "general"

        try:
            w.updateParameterNodeFromGUI(node)
            raise AssertionError("Expected RuntimeError was not raised")
        except RuntimeError as exc:
            assert "simulated SetParameter failure" in str(exc), str(exc)

        assert end_modify_calls == [42], (
            f"EndModify must be called exactly once with wasModified; got: {end_modify_calls}"
        )
        assert not w._updatingGUIFromParameterNode, (
            "_updatingGUIFromParameterNode must be False after exception"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_model_config_contract():
    """_MODEL_BY_ID must contain the original file names and encoder for both models."""
    r = _run(_MAIN_WIDGET_STUB, """
        from ZebrafishEmbryoAnalyzerLib.widget import _MODEL_BY_ID

        general = _MODEL_BY_ID["general"]
        assert general[0] == "best_model_body_3400_vgg19.pth", general
        assert general[1] == "vgg19",                          general
        assert general[2] is None,                             general

        desy = _MODEL_BY_ID["desy"]
        assert desy[0] == "best_model_body_finetuned.pth",  desy
        assert desy[1] == "vgg19",                           desy
        assert desy[2] == "best_model_eye_finetuned.pth",   desy

        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: reset_for_scene_close — scalebar state
# ---------------------------------------------------------------------------

# Minimal attribute set for reset_for_scene_close tests.
_RESET_SETUP = """\
from unittest.mock import MagicMock
from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

def _make_w(**overrides):
    w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
    w._results        = []
    w._detail         = MagicMock()
    w._image_paths    = ["/img/a.tif", "/img/b.tif"]
    w._excluded       = {"b.tif"}
    w._queue_list     = MagicMock()
    w._gallery        = MagicMock()
    w._results_tab    = MagicMock()
    w._exclude_tab    = MagicMock()
    w._run_stack      = MagicMock()
    w._scale_status   = MagicMock()
    w._bar_um_edit    = MagicMock()
    w._um_per_px      = MagicMock()
    w._um_per_px.value = 55.5   # simulate a value set by detect/apply
    for k, v in overrides.items():
        setattr(w, k, v)
    return w
"""


def test_reset_for_scene_close_resets_scale_status_text_and_style():
    """reset_for_scene_close must restore scale_status text and stylesheet to defaults."""
    r = _run(_MAIN_WIDGET_STUB + _RESET_SETUP, """
        w = _make_w()
        w.reset_for_scene_close()

        w._scale_status.setText.assert_called_with("Load images first.")
        w._scale_status.setStyleSheet.assert_called_with("color: #888; font-size: 11px;")
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_reset_for_scene_close_clears_bar_um_edit():
    """reset_for_scene_close must clear the Physical bar length input field."""
    r = _run(_MAIN_WIDGET_STUB + _RESET_SETUP, """
        w = _make_w()
        w.reset_for_scene_close()

        w._bar_um_edit.setText.assert_called_with("")
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_reset_for_scene_close_does_not_touch_um_per_px():
    """reset_for_scene_close must not modify _um_per_px — it is synced from the parameter node."""
    r = _run(_MAIN_WIDGET_STUB + _RESET_SETUP, """
        w = _make_w()
        w.reset_for_scene_close()

        # value attribute must still be 55.5 (never overwritten by reset)
        assert w._um_per_px.value == 55.5, (
            f"_um_per_px.value must not be changed by reset; got {w._um_per_px.value!r}"
        )
        w._um_per_px.setValue.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_gui_after_reset_applies_node_um_per_px():
    """After reset, updateGUIFromParameterNode syncs _um_per_px from the parameter node."""
    r = _run(_MAIN_WIDGET_STUB + _RESET_SETUP, """
        node = _FakeNode({
            "lengthEnabled": "true", "curvatureEnabled": "true",
            "ratioEnabled": "true", "eyesEnabled": "false",
            "confidenceThresholdEnabled": "false",
            "confidenceThreshold": "0.85",
            "micrometersPerPixel": "12.34",
            "selectedModelId": "general",
        })

        w = _make_w()
        w._updatingGUIFromParameterNode = False
        w._chk_length    = MagicMock()
        w._chk_curvature = MagicMock()
        w._chk_ratio     = MagicMock()
        w._chk_eyes      = MagicMock()
        w._chk_hitl      = MagicMock()
        w._threshold_slider = MagicMock()
        w._model_combo   = MagicMock()
        w._model_combo.count = 2
        w._model_combo.itemData.side_effect = lambda i: ["general", "desy"][i]

        w.reset_for_scene_close()
        # Simulate the caller syncing from the new parameter node immediately after.
        w.updateGUIFromParameterNode(node)

        assert w._um_per_px.value == 12.34, (
            f"After updateGUIFromParameterNode, _um_per_px must reflect node value; "
            f"got {w._um_per_px.value!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_reset_for_scene_close_preserves_existing_session_resets():
    """reset_for_scene_close must still clear image_paths, excluded, and results."""
    r = _run(_MAIN_WIDGET_STUB + _RESET_SETUP, """
        w = _make_w()
        assert w._image_paths  # non-empty before reset
        assert w._excluded     # non-empty before reset

        w.reset_for_scene_close()

        assert w._image_paths == [], f"image_paths not cleared: {w._image_paths}"
        assert w._excluded    == set(), f"excluded not cleared: {w._excluded}"
        assert w._results     == [], f"results not cleared: {w._results}"
        w._queue_list.clear.assert_called_once()
        w._gallery.populate.assert_called_with([])
        w._results_tab.populate.assert_called_with([], set())
        w._run_stack.setCurrentIndex.assert_called_with(0)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
