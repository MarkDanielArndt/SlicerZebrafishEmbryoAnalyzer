"""Safety tests for G1 no-thread and no-hidden-preload behavior."""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).parent.parent
PRODUCTION_ROOT = ROOT / "ZebrafishAnalysis"
WIDGET_PATH = PRODUCTION_ROOT / "ZebrafishAnalysisLib" / "widget.py"
MAIN_PATH = PRODUCTION_ROOT / "ZebrafishAnalysis.py"


@pytest.fixture
def widget_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "qt", MagicMock())
    monkeypatch.setitem(sys.modules, "ctk", MagicMock())
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    import importlib
    import ZebrafishAnalysisLib.widget as module
    return importlib.reload(module)


def _method_names(path):
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _calls_in_file(path):
    tree = ast.parse(path.read_text())
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_no_python_thread_creation_in_production_code():
    """Production extension code must not create Python background threads."""
    violations = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imported_thread_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "threading":
                for alias in node.names:
                    if alias.name == "Thread":
                        imported_thread_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "Thread"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "threading"
                ):
                    violations.append((path, node.lineno, "threading.Thread"))
                elif isinstance(func, ast.Name) and func.id in imported_thread_names:
                    violations.append((path, node.lineno, "Thread"))
    assert violations == []


def test_startup_prewarm_imports_removed():
    assert "_prewarm_imports" not in _method_names(MAIN_PATH)


def test_widget_model_preload_methods_removed():
    names = _method_names(WIDGET_PATH)
    assert "_start_preload" not in names
    assert "_preload_cached_models" not in names


def test_no_startup_model_preload_timer_or_signal():
    calls = _calls_in_file(WIDGET_PATH)
    forbidden = []
    for call in calls:
        text = ast.unparse(call)
        if "_preload_cached_models" in text or "_start_preload" in text:
            forbidden.append((call.lineno, text))
        if "singleShot" in text and "500" in text:
            forbidden.append((call.lineno, text))
    assert forbidden == []


def test_model_selection_only_updates_parameter_node():
    source = WIDGET_PATH.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    connect_body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_connect_signals":
            connect_body = "".join(lines[node.lineno - 1:node.end_lineno])
            break
    assert connect_body is not None
    assert "_notify_settings_changed" in connect_body
    assert "preload" not in connect_body.lower()


def test_run_analysis_starts_download_before_analysis_when_models_missing(widget_module):
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._run_token = 0
    w._image_paths = ["/tmp/fish.png"]
    w._model_combo = MagicMock()
    w._model_combo.currentData = "general"
    w._chk_length = MagicMock()
    w._chk_curvature = MagicMock()
    w._chk_ratio = MagicMock()
    w._chk_eyes = MagicMock()
    w._chk_hitl = MagicMock()
    for chk in (w._chk_length, w._chk_curvature, w._chk_ratio, w._chk_eyes, w._chk_hitl):
        chk.isChecked.return_value = True
    w._threshold_slider = MagicMock()
    w._threshold_slider.value = 0.85
    w._um_per_px = MagicMock()
    w._um_per_px.value = 22.99
    w._prompt_download_models = MagicMock(return_value=True)
    w._missing_required_models = MagicMock(return_value=[{"label": "Body"}])
    w._start_model_download = MagicMock()
    w._run_analysis_after_models_ready = MagicMock()

    w._on_run()

    w._start_model_download.assert_called_once()
    w._run_analysis_after_models_ready.assert_not_called()


def test_run_analysis_loads_models_only_when_cache_ready(widget_module):
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._run_token = 0
    w._image_paths = ["/tmp/fish.png"]
    w._model_combo = MagicMock()
    w._model_combo.currentData = "general"
    w._chk_length = MagicMock()
    w._chk_curvature = MagicMock()
    w._chk_ratio = MagicMock()
    w._chk_eyes = MagicMock()
    w._chk_hitl = MagicMock()
    for chk in (w._chk_length, w._chk_curvature, w._chk_ratio, w._chk_eyes, w._chk_hitl):
        chk.isChecked.return_value = True
    w._threshold_slider = MagicMock()
    w._threshold_slider.value = 0.85
    w._um_per_px = MagicMock()
    w._um_per_px.value = 22.99
    w._missing_required_models = MagicMock(return_value=[])
    w._run_analysis_after_models_ready = MagicMock()

    w._on_run()

    w._run_analysis_after_models_ready.assert_called_once()


def test_downloader_success_rechecks_cache_before_analysis(widget_module):
    import qt as _qt
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._run_analysis_after_models_ready = MagicMock()
    missing = [{"label": "Body"}]
    params = {"model_id": "general"}

    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(True, "succeeded", None, controller)
        return controller

    with patch("ZebrafishAnalysisLib.model_downloader.start_model_download", fake_start):
        w._start_model_download(missing, "general", params, token=1)

    w._missing_required_models.assert_called_with("general")
    # Analysis is now deferred via QTimer.singleShot, not called directly.
    w._run_analysis_after_models_ready.assert_not_called()
    _qt.QTimer.singleShot.assert_called_once()
    call_args = _qt.QTimer.singleShot.call_args
    assert call_args[0][0] == 0
    deferred = call_args[0][1]
    assert callable(deferred)
    # Call the deferred — token matches so analysis runs.
    deferred()
    w._run_analysis_after_models_ready.assert_called_once_with("general", params, 1)


def test_downloader_failure_does_not_start_analysis(widget_module):
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._run_analysis_after_models_ready = MagicMock()
    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(False, "failed", "offline", controller)
        return controller

    with patch("ZebrafishAnalysisLib.model_downloader.start_model_download", fake_start):
        w._start_model_download([{"label": "Body"}], "general", {}, token=1)

    w._run_analysis_after_models_ready.assert_not_called()


# ---------------------------------------------------------------------------
# New G1 token / deferred-analysis tests
# ---------------------------------------------------------------------------

def _make_download_widget(widget_module, token=1):
    """Return a minimal widget shell wired for _start_model_download tests."""
    import qt as _qt
    _qt.QTimer.singleShot.reset_mock()
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = token
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._missing_required_models = MagicMock(return_value=[])
    w._run_analysis_after_models_ready = MagicMock()
    return w


def _fire_download_success(widget_module, w, token=1):
    """Call _start_model_download with a fake downloader that immediately fires success."""
    controller = MagicMock()

    def fake_start(entries, callback, parent=None):
        w._active_downloader = controller
        callback(True, "succeeded", None, controller)
        return controller

    with patch("ZebrafishAnalysisLib.model_downloader.start_model_download", fake_start):
        w._start_model_download([{"label": "Body"}], "general", {"model_id": "general"}, token=token)

    return controller


def test_download_success_schedules_deferred_continuation_not_direct_call(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    # QTimer.singleShot must have been called exactly once with delay=0 and a callable.
    _qt.QTimer.singleShot.assert_called_once()
    args = _qt.QTimer.singleShot.call_args[0]
    assert args[0] == 0
    assert callable(args[1])

    # Analysis must NOT have been called synchronously inside _finished.
    w._run_analysis_after_models_ready.assert_not_called()


def test_download_success_does_not_call_analysis_inside_finished_callback(widget_module):
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)
    assert w._run_analysis_after_models_ready.call_count == 0


def test_deferred_continuation_calls_analysis_when_token_matches(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    deferred()

    w._run_analysis_after_models_ready.assert_called_once_with(
        "general", {"model_id": "general"}, 1
    )


def test_disposed_before_deferred_continuation_prevents_analysis(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    w._disposed = True
    deferred()

    w._run_analysis_after_models_ready.assert_not_called()
    # UI must be reset to idle.
    w._run_stack.setCurrentIndex.assert_called_with(0)


def test_stale_token_before_deferred_continuation_prevents_analysis(widget_module):
    import qt as _qt
    w = _make_download_widget(widget_module, token=1)
    _fire_download_success(widget_module, w, token=1)

    deferred = _qt.QTimer.singleShot.call_args[0][1]
    # Simulate cancel / newer run invalidating the token.
    w._run_token = 2
    deferred()

    w._run_analysis_after_models_ready.assert_not_called()
    w._run_stack.setCurrentIndex.assert_called_with(0)


def test_start_model_download_exception_restores_run_ui(widget_module):
    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._active_downloader = None
    w._disposed = False
    w._run_token = 1
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()

    def raising_start(entries, callback, parent=None):
        raise RuntimeError("network init failed")

    with patch("ZebrafishAnalysisLib.model_downloader.start_model_download", raising_start):
        w._start_model_download([{"label": "Body"}], "general", {}, token=1)

    w._run_stack.setCurrentIndex.assert_called_with(0)
    assert w._active_downloader is None


def test_no_processevents_in_download_to_analysis_path():
    source = WIDGET_PATH.read_text()
    tree = ast.parse(source)

    target_functions = {"_start_model_download", "_run_analysis_after_models_ready"}
    found_violations = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in target_functions:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    text = ast.unparse(child)
                    if "processEvents" in text:
                        found_violations.append((node.name, child.lineno, text))

    assert found_violations == [], f"processEvents found in: {found_violations}"


def test_inference_progress_callback_unchanged(widget_module):
    from ZebrafishAnalysisLib.errors import AnalysisInputError

    w = object.__new__(widget_module.ZebrafishAnalysisMainWidget)
    w._run_token = 1
    w._image_paths = ["/img/a.png", "/img/b.png", "/img/c.png"]
    w._run_progress = MagicMock()
    w._run_stack = MagicMock()
    w._chk_curvature = MagicMock()
    w._chk_curvature.isChecked.return_value = False
    w._chk_eyes = MagicMock()
    w._chk_eyes.isChecked.return_value = False

    progress_values = []

    def fake_run_analysis(paths, params, callback):
        for i in range(1, 4):
            callback(i, 3)
            progress_values.append(i)
        return [{"filename": "a.png"}, {"filename": "b.png"}, {"filename": "c.png"}]

    logic = MagicMock()
    logic.preload_models.return_value = None
    logic.run_analysis.side_effect = fake_run_analysis
    w._logic = logic

    # Stub out post-analysis methods to avoid attribute errors.
    w._on_results_ready = MagicMock()
    w._try_update_mrml_table = MagicMock()

    w._run_analysis_after_models_ready("general", {}, token=1)

    assert progress_values == [1, 2, 3]
    # setValue called with each per-image count.
    set_value_calls = [c[0][0] for c in w._run_progress.setValue.call_args_list]
    assert 1 in set_value_calls
    assert 2 in set_value_calls
    assert 3 in set_value_calls
