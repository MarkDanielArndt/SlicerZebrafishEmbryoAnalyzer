"""
Tests for the Slicer module lifecycle (Task D1).

Source-level tests verify the structural contract (class hierarchy, method
presence, observer call sites) where full instantiation outside Slicer is not
practical.  Subprocess tests verify observable state changes using minimal
Qt/Slicer stubs.
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
    "ZebrafishAnalysis",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishAnalysis.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib", "widget.py")
_DETAIL_PY = os.path.join(_MODULE_DIR, "ZebrafishAnalysisLib", "detail_tab.py")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

# VTKObservationMixin stub that actually tracks addObserver / removeObservers
# calls so behavioral tests can inspect them without a Slicer runtime.
# removeObservers() matches the real Slicer VTKObservationMixin API.
_SLICER_STUB = """\
import sys, types
from unittest.mock import MagicMock

sys.modules["qt"]  = MagicMock()
sys.modules["ctk"] = MagicMock()
sys.modules["slicer"] = MagicMock()

# ScriptedLoadableModuleWidget must be a distinct class (not object itself) so
# that class ZebrafishAnalysisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin)
# can resolve its MRO — Python C3 rejects (object, SomeSubclassOfObject).
class _BaseWidget:
    pass

# VTKObservationMixin lives in slicer.util in the real Slicer runtime.
# _TrackingMixin tracks addObserver/removeObservers calls for behavioral assertions.
class _TrackingMixin:
    def __init__(self):
        self._obs = []

    def addObserver(self, node, event, method):
        self._obs.append(event)

    def removeObservers(self, method=None):
        self._obs.clear()

    def hasObserver(self, node, event, method):
        return any(e == event for e in self._obs)

sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=_BaseWidget,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=_TrackingMixin,
)
"""


# DetailTab inherits from qt.QWidget, so qt.QWidget must be a real Python class
# for object.__new__(DetailTab) to work.  The ZebrafishAnalysisWidget tests only
# need qt as a MagicMock (no actual subclassing from qt types).
_DETAIL_STUB = """\
import sys, types
from unittest.mock import MagicMock

# Any qt.QXxx used as a base class (QWidget, QLabel, etc.) must be a real Python
# type so that classes defined in the import chain (zoom_view, detail_tab) work
# with object.__new__.  __getattr__ on the module provides a real class as fallback
# for any attribute not explicitly set.
class _FakeQBase:
    pass

class _QtModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (_FakeQBase,), {})
        setattr(self, name, cls)
        return cls

_qt = _QtModule("qt")
_qt.Qt = MagicMock()   # Qt.StrongFocus, Qt.AlignCenter, etc. stay as MagicMock
sys.modules["qt"]    = _qt
sys.modules["ctk"]   = MagicMock()
sys.modules["slicer"] = MagicMock()
sys.modules["slicer.ScriptedLoadableModule"] = types.SimpleNamespace(
    ScriptedLoadableModule=object,
    ScriptedLoadableModuleWidget=object,
    ScriptedLoadableModuleLogic=object,
    ScriptedLoadableModuleTest=object,
)
sys.modules["slicer.util"] = types.SimpleNamespace(
    VTKObservationMixin=object,
)
"""


def _run(code: str) -> subprocess.CompletedProcess:
    full = _SLICER_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


def _run_detail(code: str) -> subprocess.CompletedProcess:
    """Like _run but with qt.QWidget as a real class so DetailTab subclasses it."""
    full = _DETAIL_STUB + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _MODULE_DIR},
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_class(path: str, class_name: str) -> ast.ClassDef:
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {path}")


def _method_source(path: str, class_name: str, method_name: str) -> str:
    src = open(path).read()
    cls = _parse_class(path, class_name)
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            lines = src.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _has_method(path: str, class_name: str, method_name: str) -> bool:
    try:
        _method_source(path, class_name, method_name)
        return True
    except AssertionError:
        return False


def _class_base_names(path: str, class_name: str) -> list:
    cls = _parse_class(path, class_name)
    return [
        (b.id if isinstance(b, ast.Name) else
         b.attr if isinstance(b, ast.Attribute) else repr(b))
        for b in cls.bases
    ]


# ---------------------------------------------------------------------------
# Source-level structural tests
# ---------------------------------------------------------------------------

def test_vtk_observation_mixin_imported_from_slicer_util():
    """VTKObservationMixin must be imported from slicer.util, not slicer.ScriptedLoadableModule."""
    src = open(_MAIN_PY).read()
    # Must have: from slicer.util import VTKObservationMixin
    assert re.search(r"from\s+slicer\.util\s+import\b.*VTKObservationMixin", src), (
        "VTKObservationMixin must be imported from slicer.util"
    )
    # Must NOT appear inside the slicer.ScriptedLoadableModule import block
    slm_block_match = re.search(
        r"from\s+slicer\.ScriptedLoadableModule\s+import\s*\(([^)]*)\)", src, re.S
    )
    if slm_block_match:
        assert "VTKObservationMixin" not in slm_block_match.group(1), (
            "VTKObservationMixin must not be imported from slicer.ScriptedLoadableModule"
        )


def test_widget_inherits_vtk_observation_mixin():
    """ZebrafishAnalysisWidget must list VTKObservationMixin as a direct base."""
    bases = _class_base_names(_MAIN_PY, "ZebrafishAnalysisWidget")
    assert "VTKObservationMixin" in bases, (
        f"VTKObservationMixin missing from bases: {bases}"
    )


def test_setup_registers_start_close_event_observer():
    """setup() (via _register_scene_observers) must reference StartCloseEvent."""
    # The call may be in the extracted helper method; check the whole class.
    src = open(_MAIN_PY).read()
    cls = _parse_class(_MAIN_PY, "ZebrafishAnalysisWidget")
    lines = src.splitlines()
    cls_src = "\n".join(lines[cls.lineno - 1 : cls.end_lineno])
    assert "StartCloseEvent" in cls_src, (
        "ZebrafishAnalysisWidget must register StartCloseEvent"
    )


def test_setup_registers_end_close_event_observer():
    """setup() (via _register_scene_observers) must reference EndCloseEvent."""
    src = open(_MAIN_PY).read()
    cls = _parse_class(_MAIN_PY, "ZebrafishAnalysisWidget")
    lines = src.splitlines()
    cls_src = "\n".join(lines[cls.lineno - 1 : cls.end_lineno])
    assert "EndCloseEvent" in cls_src, (
        "ZebrafishAnalysisWidget must register EndCloseEvent"
    )


def test_cleanup_calls_remove_observers():
    """cleanup() must call removeObservers() — the real Slicer VTKObservationMixin API."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "cleanup")
    assert "removeObservers" in body, "cleanup() must call removeObservers()"
    assert "removeAllObservers" not in body, (
        "removeAllObservers() does not exist in Slicer VTKObservationMixin; use removeObservers()"
    )


def test_cleanup_guards_main_attribute():
    """cleanup() must guard self._main access so it is safe before setup()."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "cleanup")
    if "_main" in body:
        assert ("is not None" in body or "hasattr" in body), (
            "cleanup() accesses self._main without a None/hasattr guard"
        )


def test_enter_method_is_defined():
    """enter() must be defined on ZebrafishAnalysisWidget."""
    assert _has_method(_MAIN_PY, "ZebrafishAnalysisWidget", "enter"), (
        "ZebrafishAnalysisWidget must define enter()"
    )


def test_exit_method_is_defined():
    """exit() must be defined on ZebrafishAnalysisWidget."""
    assert _has_method(_MAIN_PY, "ZebrafishAnalysisWidget", "exit"), (
        "ZebrafishAnalysisWidget must define exit()"
    )


def test_scene_start_close_handler_calls_cancel_workers():
    """_on_scene_start_close() must call _cancel_workers() to stop background jobs early."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "_on_scene_start_close")
    assert "_cancel_workers" in body, (
        "_on_scene_start_close() must call _cancel_workers() on self._main"
    )


def test_scene_end_close_handler_calls_reset():
    """_on_scene_end_close() must call reset_for_scene_close() on self._main."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "_on_scene_end_close")
    assert "reset_for_scene_close" in body, (
        "_on_scene_end_close() must call self._main.reset_for_scene_close()"
    )


def test_main_widget_has_cancel_workers():
    """ZebrafishAnalysisMainWidget must define _cancel_workers()."""
    assert _has_method(_WIDGET_PY, "ZebrafishAnalysisMainWidget", "_cancel_workers"), (
        "ZebrafishAnalysisMainWidget must define _cancel_workers()"
    )


def test_main_widget_has_reset_for_scene_close():
    """ZebrafishAnalysisMainWidget must define reset_for_scene_close()."""
    assert _has_method(_WIDGET_PY, "ZebrafishAnalysisMainWidget", "reset_for_scene_close"), (
        "ZebrafishAnalysisMainWidget must define reset_for_scene_close()"
    )


def test_main_widget_cleanup_stops_detail_timer():
    """ZebrafishAnalysisMainWidget.cleanup() must stop the DetailTab poll timer."""
    body = _method_source(_WIDGET_PY, "ZebrafishAnalysisMainWidget", "cleanup")
    assert "_poll_timer" in body or "self._detail.cleanup" in body, (
        "ZebrafishAnalysisMainWidget.cleanup() must stop the detail_tab poll timer "
        "(either directly or via self._detail.cleanup())"
    )


def test_detail_tab_has_cleanup():
    """DetailTab must define cleanup() to stop its 40 ms poll timer."""
    assert _has_method(_DETAIL_PY, "DetailTab", "cleanup"), (
        "DetailTab must define cleanup()"
    )


def test_detail_tab_cleanup_calls_invalidate_cache():
    """DetailTab.cleanup() must call invalidate_cache() to discard stale worker results."""
    body = _method_source(_DETAIL_PY, "DetailTab", "cleanup")
    assert "invalidate_cache" in body, (
        "DetailTab.cleanup() must call self.invalidate_cache() before stopping the timer"
    )


def test_setup_stores_prewarm_timer():
    """setup() must store the prewarm QTimer so cleanup() can stop it."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "setup")
    assert "_prewarm_timer" in body, (
        "setup() must assign the prewarm timer to self._prewarm_timer "
        "so cleanup() can stop it on module reload"
    )


def test_cleanup_stops_prewarm_timer():
    """cleanup() must stop the prewarm timer to prevent stale callbacks after reload."""
    body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", "cleanup")
    assert "_prewarm_timer" in body, (
        "cleanup() must stop self._prewarm_timer"
    )


# ---------------------------------------------------------------------------
# Behavioral tests (subprocess with stubs)
# ---------------------------------------------------------------------------

def test_reset_for_scene_close_clears_results_and_paths():
    """reset_for_scene_close() clears _results, _image_paths and _excluded."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget

        # Bypass __init__ (needs Qt layout); set all referenced attributes manually.
        w = object.__new__(ZebrafishAnalysisMainWidget)
        w._results     = [{"filename": "a.png"}, {"filename": "b.png"}]
        w._image_paths = ["/a.png", "/b.png"]
        w._excluded    = {"b.png"}
        w._queue_list  = MagicMock()
        w._detail      = MagicMock()
        w._gallery     = MagicMock()
        w._results_tab = MagicMock()
        w._exclude_tab = MagicMock()
        w._run_stack   = MagicMock()

        w.reset_for_scene_close()

        assert w._results     == [],    f"_results: {w._results!r}"
        assert w._image_paths == [],    f"_image_paths: {w._image_paths!r}"
        assert w._excluded    == set(), f"_excluded: {w._excluded!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_reset_for_scene_close_invalidates_cache_and_clears_gallery():
    """reset_for_scene_close() calls invalidate_cache() and gallery.populate([])."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget

        w = object.__new__(ZebrafishAnalysisMainWidget)
        w._results     = [{"filename": "a.png"}]
        w._image_paths = ["/a.png"]
        w._excluded    = set()
        w._queue_list  = MagicMock()
        w._detail      = MagicMock()
        w._gallery     = MagicMock()
        w._results_tab = MagicMock()
        w._exclude_tab = MagicMock()
        w._run_stack   = MagicMock()

        w.reset_for_scene_close()

        w._detail.invalidate_cache.assert_called_once()

        populate_call = w._gallery.populate.call_args
        assert populate_call is not None, "gallery.populate was not called"
        assert populate_call[0][0] == [], (
            f"gallery.populate expected [], got {populate_call[0][0]!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cancel_workers_replaces_results_and_invalidates_cache():
    """_cancel_workers() replaces _results and calls detail.invalidate_cache()."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysisLib.widget import ZebrafishAnalysisMainWidget

        w = object.__new__(ZebrafishAnalysisMainWidget)
        w._results = [{"filename": "a.png"}]
        w._detail  = MagicMock()

        old_results = w._results
        w._cancel_workers()

        assert w._results is not old_results, "_results must be replaced (sentinel)"
        assert w._results == [], f"_results must be empty list, got {w._results!r}"
        w._detail.invalidate_cache.assert_called_once()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cleanup_is_safe_before_setup():
    """cleanup() must not raise if called before setup() (_main is None)."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisWidget

        # Create without calling __init__ or setup()
        w = object.__new__(ZebrafishAnalysisWidget)
        w._main = None
        w._obs  = []   # _TrackingMixin state (normally set by __init__)

        w.cleanup()
        w.cleanup()   # second call must also be safe
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cleanup_clears_registered_observers():
    """cleanup() calls removeObservers() so all scene observers are deregistered."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisWidget

        w = object.__new__(ZebrafishAnalysisWidget)
        w._main = None
        # Simulate two observers having been registered (StartClose, EndClose)
        w._obs  = [1001, 1002]

        w.cleanup()

        assert w._obs == [], f"Observers not cleared: {w._obs!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_prewarm_timer_stopped_on_cleanup():
    """cleanup() stops the stored _prewarm_timer to prevent stale callbacks after reload."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysis import ZebrafishAnalysisWidget

        w = object.__new__(ZebrafishAnalysisWidget)
        w._main = None
        w._obs  = []
        mock_timer = MagicMock()
        w._prewarm_timer = mock_timer

        w.cleanup()

        mock_timer.stop.assert_called_once()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_observer_registration_is_idempotent():
    """_register_scene_observers() must not add duplicate observers on repeated calls."""
    r = _run("""
        import slicer
        from ZebrafishAnalysis import ZebrafishAnalysisWidget

        w = object.__new__(ZebrafishAnalysisWidget)
        w._obs = []
        w._sceneObserversRegistered = False

        w._register_scene_observers()
        count_after_first = len(w._obs)

        w._register_scene_observers()
        count_after_second = len(w._obs)

        assert count_after_first == 2, f"Expected 2 observers after first call, got {count_after_first}"
        assert count_after_second == 2, (
            f"Second call must not add duplicates; got {count_after_second} observers"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cleanup_resets_observer_registration_flag():
    """cleanup() resets _sceneObserversRegistered so a subsequent setup can re-register."""
    r = _run("""
        from ZebrafishAnalysis import ZebrafishAnalysisWidget
        import slicer

        w = object.__new__(ZebrafishAnalysisWidget)
        w._main = None
        w._obs  = [1001, 1002]
        w._sceneObserversRegistered = True

        w.cleanup()

        assert w._sceneObserversRegistered is False, (
            "_sceneObserversRegistered must be False after cleanup"
        )
        # Re-registration must work exactly once after a cleanup
        w._register_scene_observers()
        assert len(w._obs) == 2, (
            f"Re-registration should add 2 observers; got {len(w._obs)}"
        )
        assert w._sceneObserversRegistered is True
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_cleanup_invalidates_cache_and_stops_timer():
    """DetailTab.cleanup() increments generation, clears cache/pending, and stops the timer."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysisLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._generation = 5
        d._cache   = {"a": "pixmap"}
        d._jobs    = {0, 1}
        d._pending = {(0, 5): "rgb_array"}
        mock_timer = MagicMock()
        d._poll_timer = mock_timer

        d.cleanup()

        assert d._generation == 6, f"Generation not incremented: {d._generation}"
        assert d._cache   == {}, f"Cache not cleared: {d._cache}"
        assert d._jobs    == set(), f"Jobs not cleared: {d._jobs}"
        assert d._pending == {}, f"Pending not cleared: {d._pending}"
        mock_timer.stop.assert_called_once()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_cleanup_is_idempotent():
    """DetailTab.cleanup() is safe to call twice — second call must not raise."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishAnalysisLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._generation = 0
        d._cache   = {}
        d._jobs    = set()
        d._pending = {}
        d._poll_timer = MagicMock()

        d.cleanup()
        d.cleanup()   # second call must not raise
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_no_new_mrml_node_references():
    """D1 must not introduce MRML node references in setup, enter, exit, or cleanup."""
    for method in ("setup", "enter", "exit", "cleanup", "_on_scene_start_close",
                   "_on_scene_end_close", "_register_scene_observers"):
        if not _has_method(_MAIN_PY, "ZebrafishAnalysisWidget", method):
            continue
        body = _method_source(_MAIN_PY, "ZebrafishAnalysisWidget", method)
        mrml_refs = re.findall(r'\bvtkMRML\w+\b', body)
        assert not mrml_refs, (
            f"{method}() introduces MRML node references: {mrml_refs} — D1 must not add MRML nodes"
        )
