"""
Tests for H2: Clarify analysis progress and errors.

Covers:
- Run button disabled when no images loaded (_set_queue + _refresh_run_button)
- Run button re-enabled when images are added, regardless of dependency state
- Error categorization by exit_code on InferenceController
- Traceback suppression: raw tracebacks go to log, not UI
- InferenceController.exit_code stored on _on_finished
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Widget fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def widget_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "qt", MagicMock())
    monkeypatch.setitem(sys.modules, "ctk", MagicMock())
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    monkeypatch.setitem(sys.modules, "slicer", slicer)
    import importlib
    import ZebrafishEmbryoAnalyzerLib.widget as module
    return importlib.reload(module)


def _make_widget(widget_module, deps_ok=True):
    """Return a minimal widget shell for button/error tests."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._deps_ok = deps_ok
    w._image_paths = []
    w._btn_run = MagicMock()
    return w


# ---------------------------------------------------------------------------
# Run button state
# ---------------------------------------------------------------------------

def test_refresh_run_button_disabled_when_no_images(widget_module):
    w = _make_widget(widget_module, deps_ok=True)
    w._image_paths = []
    w._refresh_run_button()
    w._btn_run.setEnabled.assert_called_with(False)


def test_refresh_run_button_enabled_when_images_and_deps_ok(widget_module):
    w = _make_widget(widget_module, deps_ok=True)
    w._image_paths = ["/tmp/fish.png"]
    w._refresh_run_button()
    w._btn_run.setEnabled.assert_called_with(True)


def test_refresh_run_button_enabled_even_when_deps_missing(widget_module):
    """Missing packages must not disable Run — they are installed when it is pressed,
    so disabling the button would leave no way to trigger the installation."""
    w = _make_widget(widget_module, deps_ok=False)
    w._image_paths = ["/tmp/fish.png"]
    w._refresh_run_button()
    w._btn_run.setEnabled.assert_called_with(True)


def test_run_button_tooltip_announces_pending_install(widget_module):
    w = _make_widget(widget_module, deps_ok=False)
    w._image_paths = ["/tmp/fish.png"]
    w._refresh_run_button()
    tooltip = w._btn_run.setToolTip.call_args.args[0]
    assert "install" in tooltip.lower()


def test_set_queue_empty_disables_run_button(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._deps_ok = True
    w._image_paths = ["/tmp/fish.png"]  # start with images
    w._btn_run = MagicMock()
    w._queue_list = MagicMock()
    w._results = []
    w._excluded = set()
    w._detail = MagicMock()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._tabs = MagicMock()
    w._um_per_px = MagicMock()
    w._load_originals = MagicMock()
    # No active runner
    w._active_runner = None

    w._set_queue([])

    # Button must be disabled after clearing queue
    w._btn_run.setEnabled.assert_called_with(False)


def test_set_queue_with_images_enables_run_button(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._deps_ok = True
    w._image_paths = []
    w._btn_run = MagicMock()
    w._queue_list = MagicMock()
    w._results = []
    w._excluded = set()
    w._detail = MagicMock()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._tabs = MagicMock()
    w._um_per_px = MagicMock()
    w._load_originals = MagicMock()
    w._active_runner = None

    w._set_queue(["/tmp/a.png", "/tmp/b.png"])

    w._btn_run.setEnabled.assert_called_with(True)


def test_run_button_disabled_on_construction_with_no_images(widget_module):
    """Run button must be disabled immediately after widget construction with no images."""
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._deps_ok = True
    w._image_paths = []
    w._btn_run = MagicMock()
    w._refresh_run_button()
    w._btn_run.setEnabled.assert_called_with(False)


def test_set_queue_enables_button_regardless_of_deps(widget_module):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._run_token = 0
    w._deps_ok = False
    w._image_paths = []
    w._btn_run = MagicMock()
    w._queue_list = MagicMock()
    w._results = []
    w._excluded = set()
    w._detail = MagicMock()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._tabs = MagicMock()
    w._um_per_px = MagicMock()
    w._load_originals = MagicMock()
    w._active_runner = None

    w._set_queue(["/tmp/a.png"])

    # Images are queued, so Run is available; the packages are installed on demand.
    w._btn_run.setEnabled.assert_called_with(True)


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

def test_categorize_exit1_shows_first_line(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    result = w._categorize_inference_error("Value error: bad input\nTraceback ...", ctrl)
    # Should use first line only — Traceback guard fires first
    # Actually: "Traceback" is NOT in the first line "Value error: bad input"
    # but it IS in the full message. Let's check: "Traceback" IS in msg.
    # So traceback guard fires → generic message
    assert result == "Analysis failed. Check the application log for details."


def test_categorize_exit1_no_traceback(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    result = w._categorize_inference_error("Segmentation failed: out of memory", ctrl)
    assert result == "Analysis failed: Segmentation failed: out of memory"


def test_categorize_exit1_multiline_no_traceback(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    result = w._categorize_inference_error("First line\nSecond line", ctrl)
    assert result == "Analysis failed: First line"


def test_categorize_exit2(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 2
    result = w._categorize_inference_error("model missing", ctrl)
    assert result == "Required models are not loaded. Run the analysis again to trigger a download."


def test_categorize_exit3(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 3
    result = w._categorize_inference_error("bad request", ctrl)
    assert result == "Internal error: bad analysis request. Check the application log."


def test_categorize_exit4(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 4
    result = w._categorize_inference_error("write failed", ctrl)
    assert result == "Internal error: could not write temporary results. Check disk space."


def test_categorize_unknown_exit_code(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 99
    result = w._categorize_inference_error("something went wrong", ctrl)
    assert result == "Analysis failed. Check the application log."


def test_categorize_none_exit_code(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = None
    result = w._categorize_inference_error("something went wrong", ctrl)
    assert result == "Analysis failed. Check the application log."


# ---------------------------------------------------------------------------
# Traceback suppression
# ---------------------------------------------------------------------------

def test_traceback_message_not_shown_in_ui(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    tb_message = "Traceback (most recent call last):\n  File foo.py, line 1\nValueError: oops"
    result = w._categorize_inference_error(tb_message, ctrl)
    assert "Traceback" not in result
    assert result == "Analysis failed. Check the application log for details."


def test_traceback_message_logged_as_warning(widget_module, caplog):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    tb_message = "Traceback (most recent call last):\n  File foo.py, line 1\nValueError: oops"
    with caplog.at_level(logging.WARNING, logger="root"):
        w._categorize_inference_error(tb_message, ctrl)
    assert any("Traceback" in r.message for r in caplog.records)


def test_traceback_in_partial_message_still_suppressed(widget_module):
    w = _make_widget(widget_module)
    ctrl = MagicMock()
    ctrl.exit_code = 1
    # "Traceback" appears mid-message
    result = w._categorize_inference_error("Error occurred. Traceback was logged.", ctrl)
    assert result == "Analysis failed. Check the application log for details."


# ---------------------------------------------------------------------------
# InferenceController.exit_code stored on _on_finished
# ---------------------------------------------------------------------------

def test_inference_controller_stores_exit_code_on_failure():
    """exit_code attribute is set in _on_finished before _finish_once."""
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    sys.path.insert(0, str(Path(__file__).parent.parent / "ZebrafishEmbryoAnalyzer"))

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController

    finished_calls = []

    def on_finished(success, state, message, ctrl):
        finished_calls.append((success, state, ctrl.exit_code))

    fake_qt = MagicMock()
    ctrl = InferenceController(
        image_paths=[],
        model_id="general",
        params={},
        originals=[],
        on_finished=on_finished,
        qt_module=fake_qt,
    )
    # Set up minimal state so _on_finished can run without a real tmp dir
    ctrl.state = "running"
    ctrl._tmp_dir = None
    ctrl._result_json_path = None
    ctrl._process = None
    ctrl._finished_called = False

    ctrl._on_finished(exit_code=3)

    assert len(finished_calls) == 1
    success, state, stored_exit = finished_calls[0]
    assert not success
    assert state == "failed"
    assert stored_exit == 3


def test_inference_controller_exit_code_default_is_none():
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    sys.path.insert(0, str(Path(__file__).parent.parent / "ZebrafishEmbryoAnalyzer"))

    from ZebrafishEmbryoAnalyzerLib.inference_runner import InferenceController

    fake_qt = MagicMock()
    ctrl = InferenceController(
        image_paths=[],
        model_id="general",
        params={},
        originals=[],
        on_finished=lambda *a: None,
        qt_module=fake_qt,
    )
    assert ctrl.exit_code is None
