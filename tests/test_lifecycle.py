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
    "ZebrafishEmbryoAnalyzer",
)
_MAIN_PY   = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzer.py")
_WIDGET_PY = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "widget.py")
_DETAIL_PY = os.path.join(_MODULE_DIR, "ZebrafishEmbryoAnalyzerLib", "detail_tab.py")


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
# that class ZebrafishEmbryoAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin)
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

    def removeObserver(self, node, event, method=None):
        if event in self._obs:
            self._obs.remove(event)

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
# vtk is imported at the top of ZebrafishEmbryoAnalyzer.py
_vtk = types.ModuleType("vtk")
_vtk.vtkCommand = types.SimpleNamespace(ModifiedEvent=33)
sys.modules["vtk"] = _vtk
import vtk  # noqa
"""


# DetailTab inherits from qt.QWidget, so qt.QWidget must be a real Python class
# for object.__new__(DetailTab) to work.  The ZebrafishEmbryoAnalyzerWidget tests only
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
_qt.Qt = MagicMock()     # Qt.StrongFocus, Qt.AlignCenter, etc. stay as MagicMock
_qt.QTimer = MagicMock() # QTimer.singleShot() must be callable in tests
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
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {path}")


def _method_source(path: str, class_name: str, method_name: str) -> str:
    src = open(path, encoding="utf-8").read()
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
    src = open(_MAIN_PY, encoding="utf-8").read()
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
    """ZebrafishEmbryoAnalyzerWidget must list VTKObservationMixin as a direct base."""
    bases = _class_base_names(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget")
    assert "VTKObservationMixin" in bases, (
        f"VTKObservationMixin missing from bases: {bases}"
    )


def test_setup_registers_start_close_event_observer():
    """setup() (via _register_scene_observers) must reference StartCloseEvent."""
    # The call may be in the extracted helper method; check the whole class.
    src = open(_MAIN_PY, encoding="utf-8").read()
    cls = _parse_class(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget")
    lines = src.splitlines()
    cls_src = "\n".join(lines[cls.lineno - 1 : cls.end_lineno])
    assert "StartCloseEvent" in cls_src, (
        "ZebrafishEmbryoAnalyzerWidget must register StartCloseEvent"
    )


def test_setup_registers_end_close_event_observer():
    """setup() (via _register_scene_observers) must reference EndCloseEvent."""
    src = open(_MAIN_PY, encoding="utf-8").read()
    cls = _parse_class(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget")
    lines = src.splitlines()
    cls_src = "\n".join(lines[cls.lineno - 1 : cls.end_lineno])
    assert "EndCloseEvent" in cls_src, (
        "ZebrafishEmbryoAnalyzerWidget must register EndCloseEvent"
    )


def test_cleanup_calls_remove_observers():
    """cleanup() must call removeObservers() — the real Slicer VTKObservationMixin API."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "cleanup")
    assert "removeObservers" in body, "cleanup() must call removeObservers()"
    assert "removeAllObservers" not in body, (
        "removeAllObservers() does not exist in Slicer VTKObservationMixin; use removeObservers()"
    )


def test_cleanup_guards_main_attribute():
    """cleanup() must guard self._main access so it is safe before setup()."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "cleanup")
    if "_main" in body:
        assert ("is not None" in body or "hasattr" in body), (
            "cleanup() accesses self._main without a None/hasattr guard"
        )


def test_enter_method_is_defined():
    """enter() must be defined on ZebrafishEmbryoAnalyzerWidget."""
    assert _has_method(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "enter"), (
        "ZebrafishEmbryoAnalyzerWidget must define enter()"
    )


def test_exit_method_is_defined():
    """exit() must be defined on ZebrafishEmbryoAnalyzerWidget."""
    assert _has_method(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "exit"), (
        "ZebrafishEmbryoAnalyzerWidget must define exit()"
    )


def test_scene_start_close_handler_calls_cancel_workers():
    """_on_scene_start_close() must cancel active async operations early."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "_on_scene_start_close")
    assert "_cancel_workers" in body, (
        "_on_scene_start_close() must call _cancel_workers() on self._main"
    )


def test_scene_end_close_handler_calls_reset():
    """_on_scene_end_close() must call reset_for_scene_close() on self._main."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "_on_scene_end_close")
    assert "reset_for_scene_close" in body, (
        "_on_scene_end_close() must call self._main.reset_for_scene_close()"
    )


def test_main_widget_has_cancel_workers():
    """ZebrafishEmbryoAnalyzerMainWidget must define _cancel_workers()."""
    assert _has_method(_WIDGET_PY, "ZebrafishEmbryoAnalyzerMainWidget", "_cancel_workers"), (
        "ZebrafishEmbryoAnalyzerMainWidget must define _cancel_workers()"
    )


def test_main_widget_has_reset_for_scene_close():
    """ZebrafishEmbryoAnalyzerMainWidget must define reset_for_scene_close()."""
    assert _has_method(_WIDGET_PY, "ZebrafishEmbryoAnalyzerMainWidget", "reset_for_scene_close"), (
        "ZebrafishEmbryoAnalyzerMainWidget must define reset_for_scene_close()"
    )


def test_main_widget_cleanup_cleans_detail_tab():
    """ZebrafishEmbryoAnalyzerMainWidget.cleanup() must clean transient detail state."""
    body = _method_source(_WIDGET_PY, "ZebrafishEmbryoAnalyzerMainWidget", "cleanup")
    assert "self._detail.cleanup" in body


def test_detail_tab_has_cleanup():
    """DetailTab must define cleanup() for transient visual state."""
    assert _has_method(_DETAIL_PY, "DetailTab", "cleanup"), (
        "DetailTab must define cleanup()"
    )


def test_detail_tab_has_reset_method():
    """DetailTab must define a reset() method for scene-close visual cleanup."""
    assert _has_method(_DETAIL_PY, "DetailTab", "reset"), (
        "DetailTab must define a reset() method"
    )


def test_detail_tab_cleanup_calls_invalidate_cache():
    """DetailTab.cleanup() must call invalidate_cache() to discard stale pixmaps."""
    body = _method_source(_DETAIL_PY, "DetailTab", "cleanup")
    assert "invalidate_cache" in body


def test_detail_tab_reset_calls_invalidate_cache():
    """DetailTab.reset() must call invalidate_cache() to discard stale pixmaps."""
    body = _method_source(_DETAIL_PY, "DetailTab", "reset")
    assert "invalidate_cache" in body, (
        "DetailTab.reset() must call self.invalidate_cache()"
    )


def test_detail_tab_reset_has_no_worker_timer_logic():
    """DetailTab.reset() must not reference removed worker/poll timer state."""
    body = _method_source(_DETAIL_PY, "DetailTab", "reset")
    assert "_poll_timer" not in body
    assert "_jobs" not in body
    assert "_pending" not in body


def test_setup_no_prewarm_timer():
    """setup() must not create a prewarm import timer."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "setup")
    assert "_prewarm_timer" not in body, (
        "setup() must not assign self._prewarm_timer"
    )


def test_cleanup_no_prewarm_timer():
    """cleanup() must not reference _prewarm_timer."""
    body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", "cleanup")
    assert "_prewarm_timer" not in body, (
        "cleanup() must not reference _prewarm_timer — it was removed in G1 Redux"
    )


# ---------------------------------------------------------------------------
# Behavioral tests (subprocess with stubs)
# ---------------------------------------------------------------------------

def test_reset_for_scene_close_clears_results_and_paths():
    """reset_for_scene_close() clears _results, _image_paths and _excluded."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        # Bypass __init__ (needs Qt layout); set all referenced attributes manually.
        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._results     = [{"filename": "a.png"}, {"filename": "b.png"}]
        w._image_paths = ["/a.png", "/b.png"]
        w._excluded    = {"b.png"}
        w._queue_list  = MagicMock()
        w._detail      = MagicMock()
        w._gallery     = MagicMock()
        w._results_tab = MagicMock()
        w._exclude_tab = MagicMock()
        w._run_stack   = MagicMock()
        w._scale_status = MagicMock()
        w._bar_um_edit  = MagicMock()
        w._active_downloader = None

        w.reset_for_scene_close()

        assert w._results     == [],    f"_results: {w._results!r}"
        assert w._image_paths == [],    f"_image_paths: {w._image_paths!r}"
        assert w._excluded    == set(), f"_excluded: {w._excluded!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_reset_for_scene_close_calls_detail_reset_and_clears_gallery():
    """reset_for_scene_close() calls detail.reset() and gallery.populate([])."""
    r = _run("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._results     = [{"filename": "a.png"}]
        w._image_paths = ["/a.png"]
        w._excluded    = set()
        w._queue_list  = MagicMock()
        w._detail      = MagicMock()
        w._gallery     = MagicMock()
        w._results_tab = MagicMock()
        w._exclude_tab = MagicMock()
        w._run_stack   = MagicMock()
        w._scale_status = MagicMock()
        w._bar_um_edit  = MagicMock()
        w._active_downloader = None

        w.reset_for_scene_close()

        w._detail.reset.assert_called_once()

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
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
        w._results = [{"filename": "a.png"}]
        w._detail  = MagicMock()
        w._run_stack = MagicMock()
        w._active_downloader = None

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
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        # Create without calling __init__ or setup()
        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._main = None
        w._parameterNode = None
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
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._main = None
        w._parameterNode = None
        # Simulate three observers registered (StartClose, EndClose, EndImport)
        w._obs  = [1001, 1002, 1003]

        w.cleanup()

        assert w._obs == [], f"Observers not cleared: {w._obs!r}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_no_prewarm_timer_on_cleanup():
    """cleanup() must not stop a _prewarm_timer."""
    # Duplicate of test_cleanup_no_prewarm_timer but uses a subprocess for defence-in-depth.
    # Use _MAIN_PY constant (interpolated before subprocess launch) so __file__ is not needed.
    r = _run(f"""
        import ast
        src = open(r"{_MAIN_PY}", encoding="utf-8").read()
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        cleanup_body = ""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "cleanup" and hasattr(node, "end_lineno"):
                    cleanup_body = "".join(lines[node.lineno - 1:node.end_lineno])
                    break
        assert "_prewarm_timer" not in cleanup_body, (
            "cleanup() must not reference _prewarm_timer after G1 Redux"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_observer_registration_is_idempotent():
    """_register_scene_observers() must not add duplicate observers on repeated calls."""
    r = _run("""
        import slicer
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._obs = []
        w._sceneObserversRegistered = False

        w._register_scene_observers()
        count_after_first = len(w._obs)

        w._register_scene_observers()
        count_after_second = len(w._obs)

        # StartCloseEvent + EndCloseEvent + EndImportEvent = 3 observers
        assert count_after_first == 3, f"Expected 3 observers after first call, got {count_after_first}"
        assert count_after_second == 3, (
            f"Second call must not add duplicates; got {count_after_second} observers"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cleanup_resets_observer_registration_flag():
    """cleanup() resets _sceneObserversRegistered so a subsequent setup can re-register."""
    r = _run("""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerWidget
        import slicer

        w = object.__new__(ZebrafishEmbryoAnalyzerWidget)
        w._main = None
        w._parameterNode = None
        w._obs  = [1001, 1002, 1003]
        w._sceneObserversRegistered = True

        w.cleanup()

        assert w._sceneObserversRegistered is False, (
            "_sceneObserversRegistered must be False after cleanup"
        )
        # Re-registration must work exactly once after a cleanup
        w._register_scene_observers()
        # StartCloseEvent + EndCloseEvent + EndImportEvent = 3 observers
        assert len(w._obs) == 3, (
            f"Re-registration should add 3 observers; got {len(w._obs)}"
        )
        assert w._sceneObserversRegistered is True
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_cleanup_invalidates_cache():
    """DetailTab.cleanup() calls invalidate_cache() which clears the pixmap cache."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache = {"a": "pixmap", "b": "pixmap2"}

        d.cleanup()

        assert d._cache == {}, f"Cache not cleared by cleanup: {d._cache}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_cleanup_is_idempotent():
    """DetailTab.cleanup() is safe to call twice — second call must not raise."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache = {}

        d.cleanup()
        d.cleanup()   # second call must not raise
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_reset_shows_placeholder():
    """DetailTab.reset() must call show_placeholder() on the view — clears visible image."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache   = {"old": "pixmap"}
        d._results = [{"filename": "fish.png"}]
        d._current_idx = 0
        d._full_pixmap = MagicMock()
        d._manual_mode = True
        d._manual_points = [(1, 2)]
        mock_view = MagicMock()
        d._view = mock_view
        d._metrics_label = MagicMock()
        d._nav_label = MagicMock()
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None

        d.reset()

        mock_view.show_placeholder.assert_called_once()
        call_text = mock_view.show_placeholder.call_args[0][0]
        assert isinstance(call_text, str) and len(call_text) > 0, (
            f"show_placeholder must be called with a non-empty string, got {call_text!r}"
        )
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_reset_clears_texts_and_labels():
    """DetailTab.reset() must clear metrics_label and nav_label."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache = {}
        d._results = []
        d._current_idx = 0
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        d._view = MagicMock()
        metrics = MagicMock()
        nav = MagicMock()
        d._metrics_label = metrics
        d._nav_label = nav
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None

        d.reset()

        metrics.setText.assert_called_with("")
        nav.setText.assert_called_with("")
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_reset_disables_navigation_and_clears_index():
    """DetailTab.reset() disables both nav buttons and resets _current_idx to 0."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache = {}
        d._results = [{"filename": "a.png"}, {"filename": "b.png"}]
        d._current_idx = 1   # non-zero to verify reset
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        d._view = MagicMock()
        d._metrics_label = MagicMock()
        d._nav_label = MagicMock()
        btn_prev = MagicMock()
        btn_next = MagicMock()
        d._btn_prev = btn_prev
        d._btn_next = btn_next
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None

        d.reset()

        assert d._current_idx == 0, f"_current_idx must be 0 after reset; got {d._current_idx}"
        btn_prev.setEnabled.assert_called_with(False)
        btn_next.setEnabled.assert_called_with(False)
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_reset_invalidates_cache_without_worker_state():
    """DetailTab.reset() clears cache without requiring worker bookkeeping."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache   = {0: "pixmap_a", 1: "pixmap_b"}
        d._results = []
        d._current_idx = 0
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        d._view = MagicMock()
        d._metrics_label = MagicMock()
        d._nav_label = MagicMock()
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None

        d.reset()

        assert d._cache   == {}, f"Cache not cleared: {d._cache}"
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_has_no_poll_pending_worker_method():
    """DetailTab must not retain the removed worker polling method."""
    assert not _has_method(_DETAIL_PY, "DetailTab", "_poll_pending")


def test_detail_tab_show_result_builds_selected_pixmap_only():
    """show_result() builds the selected pixmap and does not require neighbour preload."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache   = {}
        d._results = []
        d._current_idx = 0
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        d._view = MagicMock()
        d._metrics_label = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._btn_revert_auto = MagicMock()
        d._btn_manual_adjust = MagicMock()
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._nav_label = MagicMock()
        d._pending_reset_zoom = True
        d._chk_exclude = MagicMock()
        d._current_filename = None

        import ZebrafishEmbryoAnalyzerLib.detail_tab as _dt
        _dt._build_rgb_array = lambda result: result["rgb"]
        fake_pixmap = MagicMock()
        _dt._numpy_to_qpixmap = lambda arr: fake_pixmap

        result = {"filename": "fish.png", "rgb": object(), "length": 1.0}
        d.show_result(0, [result])

        assert d._cache[0] is fake_pixmap
        assert d._full_pixmap is fake_pixmap
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_reset_does_not_reference_poll_timer():
    """DetailTab.reset() does not use the removed poll timer."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        d = object.__new__(DetailTab)
        d._cache = {}
        d._results = []
        d._current_idx = 0
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        d._view = MagicMock()
        d._metrics_label = MagicMock()
        d._nav_label = MagicMock()
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None
        mock_timer = MagicMock()
        d._poll_timer = mock_timer

        d.reset()

        mock_timer.stop.assert_not_called()
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_tab_accepts_new_results_after_reset():
    """After reset(), show_result() must update internal state with fresh results."""
    r = _run_detail("""
        from unittest.mock import MagicMock
        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab

        from unittest.mock import patch as _patch
        d = object.__new__(DetailTab)
        d._cache = {}
        d._results = []
        d._current_idx = 0
        d._full_pixmap = None
        d._manual_mode = False
        d._manual_points = []
        mock_view = MagicMock()
        d._view = mock_view
        d._metrics_label = MagicMock()
        d._nav_label = MagicMock()
        d._btn_prev = MagicMock()
        d._btn_next = MagicMock()
        d._manual_row_widget = MagicMock()
        d._manual_status = MagicMock()
        d._on_navigate = None
        d._pending_reset_zoom = True
        d._params_getter = None
        d._logic = MagicMock()
        d._btn_revert_auto = MagicMock()
        d._btn_manual_adjust = MagicMock()
        d._chk_exclude = MagicMock()
        d._current_filename = None

        # Reset first, then call show_result with fresh data.
        d.reset()

        fresh_results = [{"filename": "new_fish.png", "length": 1.2, "curvature": "straight"}]
        # Patch _ensure_cached so no real overlay is built.
        d._ensure_cached = MagicMock()
        # Pre-populate cache so show_result can set _full_pixmap without I/O.
        fake_pixmap = MagicMock()
        d._cache[0] = fake_pixmap
        d.show_result(0, fresh_results)

        assert d._results is fresh_results, "_results must be updated to fresh list"
        assert d._current_idx == 0
        # show_result now assigns pixmap directly from cache (synchronous path).
        assert d._full_pixmap is fake_pixmap
        print("OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_no_new_mrml_node_references():
    """D1 must not introduce MRML node references in setup, enter, exit, or cleanup."""
    for method in ("setup", "enter", "exit", "cleanup", "_on_scene_start_close",
                   "_on_scene_end_close", "_register_scene_observers"):
        if not _has_method(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", method):
            continue
        body = _method_source(_MAIN_PY, "ZebrafishEmbryoAnalyzerWidget", method)
        mrml_refs = re.findall(r'\bvtkMRML\w+\b', body)
        assert not mrml_refs, (
            f"{method}() introduces MRML node references: {mrml_refs} — D1 must not add MRML nodes"
        )


# ---------------------------------------------------------------------------
# H1 — shell layout save/restore (source-level and behavioral)
# ---------------------------------------------------------------------------

def test_widget_init_has_no_set_layout_call():
    """__init__ must not call setLayout — that is done in apply_shell_layout."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzerLib" / "widget.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    text = ast.unparse(child)
                    assert "setLayout" not in text, (
                        f"__init__ must not call setLayout (line {child.lineno}): {text}"
                    )


def test_apply_shell_layout_method_exists():
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzerLib" / "widget.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "apply_shell_layout" in names
    assert "restore_shell_layout" in names


def test_restore_shell_layout_safe_without_prior_apply():
    """restore_shell_layout must not raise if apply_shell_layout was never called."""
    import sys
    import types
    from unittest.mock import MagicMock
    import importlib

    # Build a minimal stub environment for widget.py
    qt_mock = MagicMock()
    ctk_mock = MagicMock()
    slicer_mock = MagicMock()
    slicer_mock.util.mainWindow.return_value = None

    saved = {}
    for mod in ("qt", "ctk", "slicer"):
        saved[mod] = sys.modules.get(mod)
    sys.modules["qt"] = qt_mock
    sys.modules["ctk"] = ctk_mock
    sys.modules["slicer"] = slicer_mock

    try:
        import ZebrafishEmbryoAnalyzerLib.widget as _wm
        _wm = importlib.reload(_wm)
        w = object.__new__(_wm.ZebrafishEmbryoAnalyzerMainWidget)
        w._saved_layout_id = None
        w._saved_central_visible = None
        w._saved_pydock_floating = None
        w._saved_pydock_dock_area = None
        w._saved_dataprobe_collapsed = None
        # Must not raise
        w.restore_shell_layout()
    finally:
        for mod, val in saved.items():
            if val is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = val


def test_enter_calls_apply_shell_layout():
    """ZebrafishEmbryoAnalyzerWidget.enter() must call apply_shell_layout on _main."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enter":
            body_text = ast.unparse(node)
            assert "apply_shell_layout" in body_text, (
                "enter() must call apply_shell_layout"
            )
            return
    assert False, "enter() method not found"


def test_exit_calls_restore_shell_layout():
    """ZebrafishEmbryoAnalyzerWidget.exit() must call restore_shell_layout on _main."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "exit":
            body_text = ast.unparse(node)
            assert "restore_shell_layout" in body_text, (
                "exit() must call restore_shell_layout"
            )
            return
    assert False, "exit() method not found"


def test_cleanup_calls_restore_shell_layout():
    """cleanup() must call restore_shell_layout() so reload does not corrupt saved state."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent / "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cleanup":
            body_text = ast.unparse(node)
            assert "restore_shell_layout" in body_text, (
                "cleanup() must call restore_shell_layout() to handle module reload correctly"
            )
            return
    assert False, "cleanup() method not found in ZebrafishEmbryoAnalyzer.py"


def test_apply_shell_layout_is_idempotent():
    """Second call to apply_shell_layout() must be a no-op when already applied."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzerLib" / "widget.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "apply_shell_layout":
            body_text = ast.unparse(node)
            # The guard must reference _saved_layout_id and return early
            assert "_saved_layout_id" in body_text
            assert "return" in body_text
            return
    assert False, "apply_shell_layout() method not found"


def test_setup_calls_apply_shell_layout():
    """setup() must call apply_shell_layout() to handle the reload case."""
    import ast
    from pathlib import Path
    source = (Path(__file__).parent.parent /
              "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            body_text = ast.unparse(node)
            assert "apply_shell_layout" in body_text, (
                "setup() must call apply_shell_layout() to handle module reload"
            )
            return
    assert False, "setup() method not found"
