"""
Tests for MRML node creation, reuse, and update_results_table orchestration.

Static checks verify source-level contracts without subprocess overhead.
Behavioral node tests use small fake objects directly — conftest.py adds the
ZebrafishEmbryoAnalyzer directory to sys.path so mrml.py imports cleanly.
Subprocess tests cover the full update_results_table flow, which requires
the Slicer module stub so ZebrafishEmbryoAnalyzer.py can be imported.
"""

import math
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzer.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "widget.py")
_LOGIC_PY  = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "logic.py")
_MRML_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "mrml.py")
_CMAKE     = os.path.join(
    os.path.dirname(_MODULE_DIR), "ZebrafishEmbryoAnalyzer", "CMakeLists.txt"
)


# ---------------------------------------------------------------------------
# Subprocess helper (used for update_results_table integration tests)
# ---------------------------------------------------------------------------

_SLICER_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

class _BaseWidget:
    pass

class _VTKMixin:
    def addObserver(self, *a, **kw): pass
    def removeObservers(self, *a, **kw): pass
    def removeObserver(self, *a, **kw): pass
    def hasObserver(self, *a, **kw): return False

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_BaseWidget,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=_VTKMixin,
)
_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""


def _run(code: str) -> subprocess.CompletedProcess:
    full = _SLICER_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# Fake MRML objects for direct (non-subprocess) node tests
# ---------------------------------------------------------------------------

class _FakeTableNode:
    _counter = 0

    def __init__(self):
        _FakeTableNode._counter += 1
        self._id = f"vtkMRMLTableNode{_FakeTableNode._counter}"
        self._name = ""
        self._table = None

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLTableNode"

    def SetAndObserveTable(self, vtk_table):
        self._table = vtk_table


class _FakeNonTableNode:
    """Simulates a foreign node (e.g. volume node) stored under the ResultsTable role."""

    def __init__(self):
        self._id = "vtkMRMLVolumeNode1"

    def GetID(self):
        return self._id

    def IsA(self, class_name):
        return class_name == "vtkMRMLVolumeNode"


class _FakeScene:
    def __init__(self):
        self._nodes = []
        self._add_count = 0

    def AddNewNodeByClass(self, class_name):
        self._add_count += 1
        node = _FakeTableNode()
        self._nodes.append(node)
        return node


class _FakeParamNode:
    def __init__(self, existing_node=None):
        self._existing = existing_node
        self._stored_role = None
        self._stored_id = None
        self._set_ref_calls = 0

    def GetNodeReference(self, role):
        return self._existing

    def SetNodeReferenceID(self, role, node_id):
        self._stored_role = role
        self._stored_id = node_id
        self._set_ref_calls += 1


# Fake vtk module for populate_table_node / build_vtk_table tests
class _FakeVTKArray:
    def __init__(self):
        self._name = ""
        self._data = {}
        self._n = 0

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def SetNumberOfTuples(self, n):
        self._n = n

    def SetValue(self, i, val):
        self._data[i] = val

    def GetValue(self, i):
        return self._data.get(i)

    def GetNumberOfTuples(self):
        return self._n


class _FakeVTKTable:
    def __init__(self):
        self._cols = []

    def AddColumn(self, col):
        self._cols.append(col)

    def GetNumberOfColumns(self):
        return len(self._cols)

    def GetColumn(self, i):
        return self._cols[i]


def _make_fake_vtk():
    fake = types.ModuleType("vtk")
    fake.vtkTable = _FakeVTKTable
    fake.vtkDoubleArray = _FakeVTKArray
    fake.vtkStringArray = _FakeVTKArray
    return fake


# ---------------------------------------------------------------------------
# Fake MRML objects for image node tests
# ---------------------------------------------------------------------------

class _FakeVectorVolumeNode:
    _counter = 0

    def __init__(self):
        _FakeVectorVolumeNode._counter += 1
        self._id = f"vtkMRMLVectorVolumeNode{_FakeVectorVolumeNode._counter}"
        self._name = ""

    def GetID(self):
        return self._id

    def SetName(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def IsA(self, class_name):
        return class_name == "vtkMRMLVectorVolumeNode"


class _FakeNonVectorVolumeNode:
    """Simulates a wrong-type foreign node stored under the CurrentImage role."""

    def __init__(self):
        self._id = "vtkMRMLScalarVolumeNode1"

    def GetID(self):
        return self._id

    def IsA(self, class_name):
        return class_name == "vtkMRMLScalarVolumeNode"


class _FakeImageScene:
    def __init__(self):
        self._nodes = []
        self._add_count = 0
        self._last_class_name = None

    def AddNewNodeByClass(self, class_name, display_name=""):
        self._add_count += 1
        self._last_class_name = class_name
        node = _FakeVectorVolumeNode()
        node.SetName(display_name)
        self._nodes.append(node)
        return node


class _FakeImageParamNode:
    def __init__(self, existing_node=None):
        self._existing = existing_node
        self._stored_role = None
        self._stored_id = None
        self._set_ref_calls = 0

    def GetNodeReference(self, role):
        return self._existing

    def SetNodeReferenceID(self, role, node_id):
        self._stored_role = role
        self._stored_id = node_id
        self._set_ref_calls += 1


# ---------------------------------------------------------------------------
# Source-level static checks
# ---------------------------------------------------------------------------

def test_logic_py_does_not_import_mrml():
    """ZebrafishEmbryoAnalyzerLib.logic must not import the mrml adapter."""
    src = open(_LOGIC_PY).read()
    import re
    assert not re.search(r'^(?:import|from)\s+.*mrml', src, re.MULTILINE), \
        "logic.py must not import mrml"


def test_mrml_in_cmake():
    """ZebrafishEmbryoAnalyzer/CMakeLists.txt must list ZebrafishEmbryoAnalyzerLib/mrml.py."""
    content = open(_CMAKE).read()
    assert "ZebrafishEmbryoAnalyzerLib/mrml.py" in content, (
        "CMakeLists.txt does not include ZebrafishEmbryoAnalyzerLib/mrml.py"
    )


def test_mrml_in_reload_eviction_list():
    """ZebrafishEmbryoAnalyzer.py _LIB_MODULES must include ZebrafishEmbryoAnalyzerLib.mrml."""
    src = open(_MAIN_PY).read()
    assert '"ZebrafishEmbryoAnalyzerLib.mrml"' in src, (
        "_LIB_MODULES must include 'ZebrafishEmbryoAnalyzerLib.mrml'"
    )


def test_no_get_first_node_by_name_in_mrml():
    """mrml.py must not use GetFirstNodeByName for ownership lookups."""
    src = open(_MRML_PY).read()
    assert "GetFirstNodeByName" not in src, (
        "mrml.py must not use GetFirstNodeByName — use node references instead"
    )


def test_widget_has_no_persistent_table_node_pointer():
    """widget.py must not store a persistent _table_node attribute."""
    src = open(_WIDGET_PY).read()
    assert "self._table_node" not in src, (
        "widget.py must not keep a persistent _table_node pointer — "
        "ownership is via parameter node reference"
    )


def test_widget_calls_update_results_table_not_mrml_directly():
    """widget.py must call update_results_table via logic, not import mrml directly."""
    src = open(_WIDGET_PY).read()
    assert "update_results_table" in src, (
        "widget.py must call self._logic.update_results_table()"
    )
    assert "from ZebrafishEmbryoAnalyzerLib.mrml" not in src, (
        "widget.py must not import ZebrafishEmbryoAnalyzerLib.mrml directly"
    )
    assert "from ZebrafishEmbryoAnalyzerLib import mrml" not in src, (
        "widget.py must not import ZebrafishEmbryoAnalyzerLib.mrml directly"
    )


def test_mrml_module_has_no_global_slicer_import():
    """mrml.py must not have a module-level 'import slicer'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import slicer\b", stripped):
            pytest.fail("mrml.py has a module-level 'import slicer'")


def test_mrml_module_has_no_global_vtk_import():
    """mrml.py must not have a module-level 'import vtk'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import vtk\b", stripped):
            pytest.fail("mrml.py has a module-level 'import vtk'")


# ---------------------------------------------------------------------------
# Behavioral: get_or_create_table_node (direct, using fake objects)
# ---------------------------------------------------------------------------

def test_existing_node_reference_is_reused():
    """get_or_create_table_node returns the existing node without creating a new one."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

    existing = _FakeTableNode()
    existing.SetName("My renamed table")
    param_node = _FakeParamNode(existing_node=existing)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is existing, "existing node reference not reused"
    assert scene._add_count == 0, "new node created despite existing reference"
    assert param_node._set_ref_calls == 0, "SetNodeReferenceID called unexpectedly"


def test_missing_reference_creates_node_with_display_name():
    """get_or_create_table_node creates exactly one new node with the canonical name."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert node is not None
    assert scene._add_count == 1, f"expected 1 new node, got {scene._add_count}"
    assert node.GetName() == "ZebrafishEmbryoAnalyzer Results"


def test_new_node_id_stored_in_param_node():
    """get_or_create_table_node stores the new node ID in the parameter node."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node, ROLE_RESULTS_TABLE

    param_node = _FakeParamNode(existing_node=None)
    scene = _FakeScene()

    node = get_or_create_table_node(param_node, scene)

    assert param_node._set_ref_calls == 1, "SetNodeReferenceID not called"
    assert param_node._stored_role == ROLE_RESULTS_TABLE
    assert param_node._stored_id == node.GetID()


def test_renamed_node_is_reused():
    """A node renamed by the user is still found via the stored reference."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    existing = _FakeTableNode()
    existing.SetName("User renamed this")
    param_node = _FakeParamNode(existing_node=existing)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is existing
    assert result.GetName() == "User renamed this", "user name was overwritten"
    assert scene._add_count == 0


def test_wrong_node_type_creates_new_table_node():
    """A reference to a non-table node triggers creation of a new table node."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_table_node

    wrong_node = _FakeNonTableNode()
    param_node = _FakeParamNode(existing_node=wrong_node)
    scene = _FakeScene()

    result = get_or_create_table_node(param_node, scene)

    assert result is not wrong_node, "wrong-type foreign node must not be reused"
    assert scene._add_count == 1, "expected exactly one new table node"
    assert param_node._stored_id != wrong_node.GetID(), (
        "reference must point to the new node, not the wrong-type node"
    )
    assert result.IsA("vtkMRMLTableNode"), "new node must be a table node"


# ---------------------------------------------------------------------------
# Behavioral: populate_table_node / build_vtk_table (direct, fake vtk)
# ---------------------------------------------------------------------------

def test_populate_table_node_columns_and_names(monkeypatch):
    """populate_table_node creates one correctly named column per TABLE_SCHEMA entry."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import TABLE_SCHEMA

    fake_vtk = _make_fake_vtk()
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]
    node = _FakeTableNode()
    mrml_mod.populate_table_node(rows, node)

    assert node._table is not None
    assert node._table.GetNumberOfColumns() == len(TABLE_SCHEMA)
    expected_names = [col for col, _, _ in TABLE_SCHEMA]
    actual_names = [node._table.GetColumn(i).GetName()
                    for i in range(node._table.GetNumberOfColumns())]
    assert actual_names == expected_names


def test_populate_table_node_applies_atomically(monkeypatch):
    """populate_table_node only calls SetAndObserveTable after full construction."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod
    from ZebrafishEmbryoAnalyzerLib.mrml import TABLE_SCHEMA

    fake_vtk = _make_fake_vtk()
    set_observe_calls = []

    class _TrackingNode(_FakeTableNode):
        def SetAndObserveTable(self, t):
            set_observe_calls.append(t.GetNumberOfColumns())
            super().SetAndObserveTable(t)

    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]
    node = _TrackingNode()
    mrml_mod.populate_table_node(rows, node)

    assert len(set_observe_calls) == 1, "SetAndObserveTable must be called exactly once"
    assert set_observe_calls[0] == len(TABLE_SCHEMA), (
        "SetAndObserveTable called with incomplete table"
    )


def test_populate_table_node_existing_table_preserved_on_failure(monkeypatch):
    """If vtk construction fails, the existing table on the node is not replaced."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod

    class _BrokenVTK:
        def vtkTable(self):
            raise RuntimeError("vtk construction failed")

    monkeypatch.setitem(sys.modules, "vtk", _BrokenVTK())

    original_sentinel = object()
    node = _FakeTableNode()
    node._table = original_sentinel

    rows = [{"Filename": "a.png", "Length_um": 1.0, "CurvatureClass": "1",
              "LengthStraightRatio": 1.05, "EyeArea_um2": math.nan,
              "EyeDiameter_um": math.nan, "Error": ""}]

    with pytest.raises(Exception):
        mrml_mod.populate_table_node(rows, node)

    assert node._table is original_sentinel, "existing table was overwritten on error"


def test_input_results_not_mutated_by_update(monkeypatch):
    """update_results_table must not mutate the input results list or dicts."""
    from ZebrafishEmbryoAnalyzerLib import mrml as mrml_mod

    fake_vtk = _make_fake_vtk()
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    results = [
        {
            "filename": "fish.png", "length": 1.0, "curvature": 2, "ratio": 1.05,
            "eye_area": None, "eye_diameter": None, "error": None,
        }
    ]
    original_results = [dict(r) for r in results]

    rows = mrml_mod.results_to_rows(results)
    node = _FakeTableNode()
    mrml_mod.populate_table_node(rows, node)

    assert results[0] == original_results[0], "input result dict was mutated"


# ---------------------------------------------------------------------------
# Subprocess: update_results_table integration
# ---------------------------------------------------------------------------

def test_update_results_table_calls_mrml_functions():
    """update_results_table builds the vtk table then resolves/creates the MRML node."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_table = MagicMock()
        fake_node = MagicMock()
        fake_node.GetID.return_value = "nodeID1"

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   return_value=fake_table) as mock_build, \\
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   return_value=fake_node) as mock_get:
            import slicer
            result = logic.update_results_table([
                {"filename": "a.png", "length": 1.0, "curvature": 1, "ratio": 1.0,
                 "eye_area": None, "eye_diameter": None, "error": None}
            ])

        assert mock_build.called, "build_vtk_table not called"
        assert mock_get.called, "get_or_create_table_node not called"
        fake_node.SetAndObserveTable.assert_called_once_with(fake_table)
        assert result is fake_node
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_vtk_build_failure_creates_no_node():
    """If build_vtk_table raises, get_or_create_table_node must not be called."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        node_creation_calls = []

        def _record_create(param_node, scene):
            node_creation_calls.append(1)
            return MagicMock()

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   side_effect=RuntimeError("vtk unavailable")), \\
             patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_table_node",
                   _record_create):
            try:
                logic.update_results_table([
                    {"filename": "a.png", "length": None, "curvature": None,
                     "ratio": None, "eye_area": None, "eye_diameter": None, "error": None}
                ])
            except MRMLAdapterError:
                pass

        assert not node_creation_calls, (
            f"get_or_create_table_node was called despite build failure: {node_creation_calls}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_results_table_wraps_unexpected_exception_as_mrml_error():
    """update_results_table wraps unexpected exceptions as MRMLAdapterError."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.build_vtk_table",
                   side_effect=RuntimeError("vtk broke")):
            try:
                logic.update_results_table([
                    {"filename": "a.png", "length": None, "curvature": None,
                     "ratio": None, "eye_area": None, "eye_diameter": None, "error": None}
                ])
                print("NO_ERROR")
            except MRMLAdapterError as exc:
                print(f"OK:{exc}")
            except Exception as exc:
                print(f"WRONG_TYPE:{type(exc).__name__}:{exc}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK:"), r.stdout
    assert "NO_ERROR" not in r.stdout


def test_widget_mrml_failure_preserves_results_via_helper():
    """_try_update_mrml_table must not affect self._results on MRMLAdapterError."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._results = [{"filename": "fish.png"}]

        mock_logic = MagicMock()
        mock_logic.update_results_table.side_effect = MRMLAdapterError("simulated")
        w._logic = mock_logic

        import slicer

        # Call the actual production method, not a hand-written copy
        w._try_update_mrml_table(w._results)

        assert w._results == [{"filename": "fish.png"}], (
            f"_results changed: {w._results!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_run_analysis_has_no_mrml_calls():
    """run_analysis() must not call update_results_table or any MRML function."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()

        calls = []

        def _fake_update(results):
            calls.append("update_results_table")

        logic.update_results_table = _fake_update

        with patch("ZebrafishEmbryoAnalyzerLib.logic.analyse_images",
                   return_value=[{"filename": "x.png"}]):
            logic.run_analysis(["/x.png"], {"um_per_px": 1.0})

        assert not calls, (
            f"run_analysis() called update_results_table: {calls}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# ---------------------------------------------------------------------------
# Static checks for E2b
# ---------------------------------------------------------------------------

def test_mrml_module_has_no_global_numpy_import():
    """mrml.py must not have a module-level 'import numpy'."""
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import numpy\b", stripped):
            pytest.fail("mrml.py has a module-level 'import numpy'")


def test_widget_source_does_not_contain_persistent_image_node_pointer():
    """widget.py must not store a persistent node pointer like self._image_node."""
    src = open(_WIDGET_PY).read()
    assert "self._image_node" not in src, (
        "widget.py must not keep a persistent _image_node pointer — "
        "ownership is via parameter node reference"
    )


def test_widget_source_calls_update_current_image_node():
    """widget.py must call update_current_image_node."""
    src = open(_WIDGET_PY).read()
    assert "update_current_image_node" in src, (
        "widget.py must call self._logic.update_current_image_node()"
    )


def test_mrml_module_exports_image_functions():
    """mrml.py must export all required E2b symbols."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "ROLE_CURRENT_IMAGE")
    assert hasattr(mrml, "image_geometry")
    assert hasattr(mrml, "get_or_create_image_node")
    assert hasattr(mrml, "update_image_node")


# ---------------------------------------------------------------------------
# Static checks for E2c
# ---------------------------------------------------------------------------

def test_mrml_module_exports_segmentation_symbols():
    """mrml.py must export all required E2c symbols."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "ROLE_CURRENT_SEGMENTATION")
    assert hasattr(mrml, "resample_mask_to_original")
    assert hasattr(mrml, "get_or_create_segmentation_node")
    assert hasattr(mrml, "update_segmentation_node")


def test_widget_source_does_not_contain_persistent_segmentation_node_pointer():
    """widget.py must not store a persistent node pointer like self._segmentation_node."""
    src = open(_WIDGET_PY).read()
    assert "self._segmentation_node" not in src, (
        "widget.py must not keep a persistent _segmentation_node pointer — "
        "ownership is via parameter node reference"
    )


def test_widget_source_calls_try_update_mrml_segmentation():
    """widget.py must contain _try_update_mrml_segmentation."""
    src = open(_WIDGET_PY).read()
    assert "_try_update_mrml_segmentation" in src, (
        "widget.py must define and call _try_update_mrml_segmentation()"
    )


def test_on_gallery_select_calls_segmentation_after_image():
    """_on_gallery_select must call _try_update_mrml_segmentation after _try_update_mrml_image."""
    src = open(_WIDGET_PY).read()
    img_pos = src.find("_try_update_mrml_image(self._results[index])")
    seg_pos = src.find("_try_update_mrml_segmentation(self._results[index])")
    assert img_pos != -1, "_try_update_mrml_image call not found in _on_gallery_select"
    assert seg_pos != -1, "_try_update_mrml_segmentation call not found in _on_gallery_select"
    assert seg_pos > img_pos, (
        "_try_update_mrml_segmentation must appear after _try_update_mrml_image"
    )


# ---------------------------------------------------------------------------
# Behavioral: get_or_create_image_node (direct, using fake objects)
# ---------------------------------------------------------------------------

def test_get_or_create_image_node_creates_new_when_no_reference():
    """get_or_create_image_node creates a new node when no reference exists."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_image_node, ROLE_CURRENT_IMAGE

    param_node = _FakeImageParamNode(existing_node=None)
    scene = _FakeImageScene()

    node = get_or_create_image_node(param_node, scene)

    assert node is not None
    assert scene._add_count == 1, f"expected 1 new node, got {scene._add_count}"
    assert param_node._set_ref_calls == 1, "SetNodeReferenceID not called"
    assert param_node._stored_role == ROLE_CURRENT_IMAGE
    assert param_node._stored_id == node.GetID()


def test_get_or_create_image_node_creates_node_with_display_name():
    """get_or_create_image_node names the new node 'ZebrafishEmbryoAnalyzer Current Image'."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_image_node

    param_node = _FakeImageParamNode(existing_node=None)
    scene = _FakeImageScene()

    node = get_or_create_image_node(param_node, scene)

    assert node.GetName() == "ZebrafishEmbryoAnalyzer Current Image"


def test_get_or_create_image_node_reuses_existing_reference():
    """get_or_create_image_node returns the same node on repeated calls."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_image_node

    existing = _FakeVectorVolumeNode()
    param_node = _FakeImageParamNode(existing_node=existing)
    scene = _FakeImageScene()

    result = get_or_create_image_node(param_node, scene)

    assert result is existing, "existing node reference not reused"
    assert scene._add_count == 0, "new node created despite existing reference"
    assert param_node._set_ref_calls == 0, "SetNodeReferenceID called unexpectedly"


def test_get_or_create_image_node_idempotent_calls_add_once():
    """AddNewNodeByClass is called exactly once even across two separate invocations."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_image_node

    param_node = _FakeImageParamNode(existing_node=None)
    scene = _FakeImageScene()

    node1 = get_or_create_image_node(param_node, scene)
    # Simulate the param_node now holding the reference to the created node
    param_node._existing = node1
    node2 = get_or_create_image_node(param_node, scene)

    assert node1 is node2, "second call must return the same node"
    assert scene._add_count == 1, "AddNewNodeByClass must be called exactly once"


def test_get_or_create_image_node_wrong_type_creates_new():
    """A wrong-type foreign node is left in scene; a fresh vector volume node is created."""
    from ZebrafishEmbryoAnalyzerLib.mrml import get_or_create_image_node

    wrong_node = _FakeNonVectorVolumeNode()
    param_node = _FakeImageParamNode(existing_node=wrong_node)
    scene = _FakeImageScene()

    result = get_or_create_image_node(param_node, scene)

    assert result is not wrong_node, "wrong-type foreign node must not be reused"
    assert scene._add_count == 1, "expected exactly one new vector volume node"
    assert param_node._stored_id != wrong_node.GetID(), (
        "reference must point to the new node, not the wrong-type node"
    )
    assert result.IsA("vtkMRMLVectorVolumeNode"), "new node must be a vector volume node"


# ---------------------------------------------------------------------------
# Subprocess: update_current_image_node integration
# ---------------------------------------------------------------------------

def test_update_current_image_node_none_original_returns_none():
    """update_current_image_node returns None when result['original'] is None."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        result = logic.update_current_image_node({"original": None}, 22.99)
        assert result is None, f"expected None, got {result!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_current_image_node_none_result_returns_none():
    """update_current_image_node returns None when result itself is None."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        result = logic.update_current_image_node(None, 22.99)
        assert result is None, f"expected None, got {result!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_update_current_image_node_wraps_exception_as_mrml_error():
    """Unexpected exceptions are wrapped as MRMLAdapterError."""
    r = _run("""
        from unittest.mock import patch, MagicMock
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.errors import MRMLAdapterError
        import numpy as np

        logic = ZebrafishEmbryoAnalyzerLogic()
        logic.getParameterNode = MagicMock(return_value=MagicMock())

        fake_image = np.zeros((10, 10, 3), dtype="uint8")

        with patch("ZebrafishEmbryoAnalyzerLib.mrml.get_or_create_image_node",
                   side_effect=RuntimeError("mrml broke")):
            try:
                logic.update_current_image_node({"original": fake_image}, 22.99)
                print("NO_ERROR")
            except MRMLAdapterError as exc:
                print(f"OK:{exc}")
            except Exception as exc:
                print(f"WRONG_TYPE:{type(exc).__name__}:{exc}")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK:"), r.stdout
    assert "NO_ERROR" not in r.stdout


# ---------------------------------------------------------------------------
# Behavioral: update_image_node with fake VTK (numpy.int64 round-trip)
# ---------------------------------------------------------------------------

def test_update_image_node_calls_set_dimensions_with_correct_values():
    """update_image_node derives (W, H, 1) dimensions from the image array."""
    import numpy as np
    import importlib
    from unittest.mock import MagicMock

    # Build fake VTK objects
    fake_vtk_array = MagicMock()
    fake_numpy_support = MagicMock()
    fake_numpy_support.numpy_to_vtk.return_value = fake_vtk_array

    fake_image_data = MagicMock()
    fake_vtk_module = MagicMock()
    fake_vtk_module.vtkImageData.return_value = fake_image_data
    fake_vtk_module.vtkMatrix4x4.return_value = MagicMock()
    fake_vtk_module.VTK_UNSIGNED_CHAR = 3

    fake_vtk_util = MagicMock()
    fake_vtk_util.numpy_support = fake_numpy_support

    fake_node = MagicMock()

    # Inject fake vtk into sys.modules for the duration of this test
    original_vtk = sys.modules.pop("vtk", None)
    original_vtk_util = sys.modules.pop("vtk.util", None)
    original_vtk_util_ns = sys.modules.pop("vtk.util.numpy_support", None)
    try:
        sys.modules["vtk"] = fake_vtk_module
        sys.modules["vtk.util"] = fake_vtk_util
        sys.modules["vtk.util.numpy_support"] = fake_numpy_support

        # Force reimport of mrml to pick up fake vtk
        import ZebrafishEmbryoAnalyzerLib.mrml as mrml_mod
        importlib.reload(mrml_mod)

        image = np.zeros((10, 8, 3), dtype="uint8")
        mrml_mod.update_image_node(image, 22.99, fake_node)

        # Dimensions must be (W=8, H=10, 1)
        fake_image_data.SetDimensions.assert_called_once_with((8, 10, 1))
        # SetSpacing must be called before SetAndObserveImageData
        assert fake_node.SetSpacing.called
        assert fake_node.SetAndObserveImageData.called
        # update_image_node calls node.SetSpacing(*spacing) which unpacks the tuple
        spacing_args = fake_node.SetSpacing.call_args[0]
        assert spacing_args[0] == pytest.approx(22.99 / 1000.0)
        assert spacing_args[1] == pytest.approx(22.99 / 1000.0)
        assert spacing_args[2] == pytest.approx(1.0)
    finally:
        sys.modules.pop("vtk", None)
        sys.modules.pop("vtk.util", None)
        sys.modules.pop("vtk.util.numpy_support", None)
        if original_vtk is not None:
            sys.modules["vtk"] = original_vtk
        if original_vtk_util is not None:
            sys.modules["vtk.util"] = original_vtk_util
        if original_vtk_util_ns is not None:
            sys.modules["vtk.util.numpy_support"] = original_vtk_util_ns
        # Reload mrml again with real (absent) vtk to restore state
        importlib.reload(mrml_mod)
