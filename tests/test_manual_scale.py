"""
Manual scale-bar calibration: logic helper, widget handlers and DetailTab mode.

The logic tests are pure (synthetic image in tmp_path). The widget tests bypass
__init__ with MagicMocks (no Qt/Slicer); the DetailTab tests run in a subprocess
with the qt stub from test_lifecycle.
"""

import sys
import textwrap
from unittest.mock import MagicMock

import numpy as np
import pytest

from .test_lifecycle import _run_detail


@pytest.fixture
def image_path(tmp_path):
    import cv2
    img = np.zeros((100, 200, 3), dtype=np.uint8)   # h=100, w=200
    p = tmp_path / "bar.png"
    assert cv2.imwrite(str(p), img)
    return str(p)


# ---------------------------------------------------------------------------
# calibrate_scale_from_points
# ---------------------------------------------------------------------------

def test_horizontal_bar_gives_expected_scale(image_path):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(image_path, (50, 20), (50, 120), 500.0)
    assert r["success"] is True
    assert r["bar_length_px"] == pytest.approx(100.0)
    assert r["scale_um_per_px"] == pytest.approx(5.0)


def test_row_col_converted_to_xy_for_diagonal_pair(image_path):
    # (row, col): 30 rows and 40 cols apart -> 50 px. A swapped x/y would still give
    # 50, so pin the conversion by spying on what reaches the library function.
    from unittest.mock import patch
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    import ZebrafishEmbryoAnalyzerCore.scalebar as sb
    import zebrafish_measure.scalebar as zm
    real = zm.calibrate_from_endpoints
    spy = MagicMock(side_effect=real)
    with patch.object(zm, "calibrate_from_endpoints", spy):
        r = calibrate_scale_from_points(image_path, (10, 20), (40, 60), 100.0)
    (pt1, pt2, shape), kw = spy.call_args
    assert pt1 == (20.0, 10.0) and pt2 == (60.0, 40.0)   # (x, y) = (col, row)
    assert shape == (100, 200)                            # (h, w)
    assert kw["label_um"] == 100.0
    assert r["bar_length_px"] == pytest.approx(50.0)
    assert r["scale_um_per_px"] == pytest.approx(2.0)


@pytest.mark.parametrize("p2", [(50, 20), (50.4, 20.4)])
def test_too_close_points_fail(image_path, p2):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(image_path, (50, 20), p2, 500.0)
    assert r["success"] is False and r["scale_um_per_px"] is None
    assert r["message"]


@pytest.mark.parametrize("bar", [None, "", "abc", 0, -5, float("nan"), float("inf")])
def test_bad_bar_length_fails(image_path, bar):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(image_path, (50, 20), (50, 120), bar)
    assert r["success"] is False and r["message"]


@pytest.mark.parametrize("bar", [0.01, 1e6])   # 0.0001 and 10000 um/px
def test_out_of_range_scale_fails(image_path, bar):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(image_path, (50, 20), (50, 120), bar)
    assert r["success"] is False
    assert "range" in r["message"]


def test_unreadable_image_fails(tmp_path):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(str(tmp_path / "missing.png"), (1, 1), (1, 50), 100.0)
    assert r["success"] is False
    assert "read" in r["message"].lower()


def test_malformed_points_do_not_raise(image_path):
    from ZebrafishEmbryoAnalyzerLib.logic import calibrate_scale_from_points
    r = calibrate_scale_from_points(image_path, None, (1, 2), 100.0)
    assert r["success"] is False


# ---------------------------------------------------------------------------
# Widget handlers
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


def _widget(widget_module, points=(), text="500"):
    w = object.__new__(widget_module.ZebrafishEmbryoAnalyzerMainWidget)
    w._logic = MagicMock()
    w._image_paths = ["/first.png", "/second.png"]
    w._scale_points = list(points)
    w._bar_um_edit = MagicMock()
    w._bar_um_edit.text = text
    w._scale_status = MagicMock()
    w._um_per_px = MagicMock()
    w._detail = MagicMock()
    w._detail._calib_mode = False
    w._tabs = MagicMock()
    return w


def test_apply_with_two_points_uses_manual_calibration(widget_module):
    w = _widget(widget_module, points=[(10, 20), (10, 120)])
    w._logic.calibrate_scale_from_points.return_value = {
        "success": True, "scale_um_per_px": 5.0, "bar_length_px": 100.0, "message": ""}
    w._on_apply_scale()
    w._logic.calibrate_scale_from_points.assert_called_once_with(
        "/first.png", (10, 20), (10, 120), 500.0)
    w._logic.detect_scalebar.assert_not_called()
    assert w._um_per_px.value == 5.0
    msg = w._scale_status.setText.call_args.args[0]
    assert "100.0 px" in msg and "500" in msg and "5.0000" in msg


def test_apply_with_two_points_failure_keeps_spinbox(widget_module):
    w = _widget(widget_module, points=[(10, 20), (10, 120)])
    w._um_per_px.value = 22.99
    w._logic.calibrate_scale_from_points.return_value = {
        "success": False, "scale_um_per_px": None, "bar_length_px": None,
        "message": "out of range"}
    w._on_apply_scale()
    assert w._um_per_px.value == 22.99
    w._scale_status.setText.assert_called_with("out of range")


def test_apply_without_points_falls_back_to_auto_detect(widget_module):
    w = _widget(widget_module)
    w._detail._calib_mode = False
    w._logic.detect_scalebar.return_value = {
        "success": True, "scale_um_per_px": 3.0}
    w._on_apply_scale()
    w._logic.detect_scalebar.assert_called_once_with("/first.png", label_um=500.0)
    w._logic.calibrate_scale_from_points.assert_not_called()
    assert w._um_per_px.value == 3.0
    w._scale_status.setStyleSheet.assert_called_with("color: #4CAF50;")


def test_apply_with_one_point_does_not_auto_detect(widget_module):
    w = _widget(widget_module, points=[(10, 20)])
    w._detail._calib_mode = False
    w._on_apply_scale()
    w._logic.detect_scalebar.assert_not_called()
    w._logic.calibrate_scale_from_points.assert_not_called()
    w._scale_status.setText.assert_called_with("Click both ends of the scale bar first.")


def test_apply_in_calibration_mode_does_not_auto_detect(widget_module):
    w = _widget(widget_module)
    w._detail._calib_mode = True
    w._on_apply_scale()
    w._logic.detect_scalebar.assert_not_called()
    w._scale_status.setText.assert_called_with("Click both ends of the scale bar first.")


def test_apply_two_points_empty_length_prompts(widget_module):
    w = _widget(widget_module, points=[(1, 2), (3, 4)], text="")
    w._detail._calib_mode = True
    w._on_apply_scale()
    w._scale_status.setText.assert_called_with("Enter the physical bar length (µm).")
    w._logic.calibrate_scale_from_points.assert_not_called()


def test_points_cleared_sets_cancelled_status(widget_module):
    w = _widget(widget_module)
    w._on_scale_points_changed([(1, 1)], None)
    w._on_scale_points_changed([], None)
    w._scale_status.setText.assert_called_with("Manual calibration cancelled.")


def test_points_cleared_keeps_apply_result(widget_module):
    w = _widget(widget_module, points=[(10, 20), (10, 120)])
    w._detail._calib_mode = True
    w._on_scale_points_changed([(10, 20), (10, 120)], 100.0)
    w._logic.calibrate_scale_from_points.return_value = {
        "success": True, "scale_um_per_px": 5.0, "bar_length_px": 100.0, "message": ""}
    w._on_apply_scale()
    w._scale_status.setText.reset_mock()
    w._on_scale_points_changed([], None)
    w._scale_status.setText.assert_not_called()


def test_apply_auto_detect_failure_shows_message(widget_module):
    w = _widget(widget_module)
    w._logic.detect_scalebar.return_value = {"success": False, "message": "nothing found"}
    w._on_apply_scale()
    w._scale_status.setText.assert_called_with("nothing found")


def test_apply_invalid_float_even_with_points(widget_module):
    w = _widget(widget_module, points=[(1, 2), (3, 4)], text="abc")
    w._on_apply_scale()
    assert "Invalid value" in w._scale_status.setText.call_args.args[0]
    w._logic.calibrate_scale_from_points.assert_not_called()
    w._logic.detect_scalebar.assert_not_called()


def test_points_callback_shows_pixel_distance(widget_module):
    w = _widget(widget_module)
    w._on_scale_points_changed([(0, 0), (0, 100)], 100.0)
    assert w._scale_points == [(0, 0), (0, 100)]
    assert "100.0 px" in w._scale_status.setText.call_args.args[0]


def test_manual_button_requires_images(widget_module):
    w = _widget(widget_module)
    w._image_paths = []
    w._on_manual_scale()
    w._scale_status.setText.assert_called_with("Load images first.")
    w._detail.start_scale_calibration.assert_not_called()


def test_manual_button_starts_calibration_on_first_image(widget_module, image_path):
    w = _widget(widget_module)
    w._image_paths = [image_path, "/second.png"]
    w._scale_points = [(1, 1), (2, 2)]
    w.ensure_dependencies = MagicMock(return_value=True)
    w._on_manual_scale()
    rgb = w._detail.start_scale_calibration.call_args.args[0]
    assert rgb.shape == (100, 200, 3)
    w._tabs.setCurrentIndex.assert_called_with(1)
    assert w._scale_points == []
    assert w._scale_status.setText.call_args.args[0] == \
        "Click the start and end of the scale bar"


def test_reset_for_scene_close_clears_scale_points(widget_module):
    w = _widget(widget_module, points=[(1, 2), (3, 4)])
    w._cancel_workers = MagicMock()
    w._gallery = MagicMock()
    w._results_tab = MagicMock()
    w._run_stack = MagicMock()
    w.reset_for_scene_close()
    assert w._scale_points == []
    w._detail.reset.assert_called_once()


def test_set_queue_clears_scale_points(widget_module):
    w = _widget(widget_module, points=[(1, 2), (3, 4)])
    w._run_token = 0
    w._deps_ok = True
    w._btn_run = MagicMock()
    w._results = []
    w._excluded = set()
    w._results_tab = MagicMock()
    w._gallery = MagicMock()
    w._load_originals = MagicMock()
    w._active_runner = None
    w._set_queue([])
    assert w._scale_points == []


# ---------------------------------------------------------------------------
# DetailTab calibration mode (subprocess, qt stub)
# ---------------------------------------------------------------------------

_DETAIL_SETUP = """
    import numpy as np
    from unittest.mock import MagicMock
    from ZebrafishEmbryoAnalyzerLib import detail_tab
    from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab
    detail_tab._numpy_to_qpixmap = lambda a: MagicMock()
    detail_tab.qt.QColor = MagicMock()

    d = object.__new__(DetailTab)
    d._view = MagicMock()
    for n in ("_metrics_label", "_nav_label", "_btn_prev", "_btn_next",
              "_manual_row_widget", "_manual_status", "_chk_exclude",
              "_btn_revert_auto", "_btn_manual_adjust"):
        setattr(d, n, MagicMock())
    d._cache = {}
    d._results = []
    d._current_idx = 0
    d._manual_mode = False
    d._manual_points = []
    d._calib_mode = False
    d._calib_points = []
    d._current_filename = None
    d._full_pixmap = None
    events = []
    d._on_calibration_change = lambda pts, dist: events.append((pts, dist))
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
"""


def _detail(body):
    r = _run_detail(textwrap.dedent(_DETAIL_SETUP) + textwrap.dedent(body))
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_detail_calibration_collects_two_taps_and_reports_distance():
    _detail("""
        d.start_scale_calibration(rgb)
        d._view.set_manual_mode.assert_called_with(True)
        d._on_image_tap(1, 2)
        d._on_image_tap(4, 6)
        assert events[-2] == ([(1, 2)], None), events
        assert events[-1] == ([(1, 2), (4, 6)], 5.0), events
        assert d._manual_points == []
        # third tap starts over
        d._on_image_tap(7, 7)
        assert events[-1] == ([(7, 7)], None), events
        print("OK")
    """)


def test_detail_manual_adjust_cancels_calibration():
    _detail("""
        d.start_scale_calibration(rgb)
        d._on_image_tap(1, 2)
        d._results = [{"filename": "a.png"}]
        d._on_manual_adjust_clicked()
        assert d._calib_mode is False and d._calib_points == []
        assert d._manual_mode is True
        assert events[-1] == ([], None)
        d._on_image_tap(3, 3)          # goes to the manual length tool
        assert d._manual_points == [(3, 3)] and d._calib_points == []
        print("OK")
    """)


def test_detail_calibration_cancels_manual_mode_and_reset_clears():
    _detail("""
        d._manual_mode = True
        d._manual_points = [(1, 1)]
        d.start_scale_calibration(rgb)
        assert d._manual_mode is False and d._manual_points == []
        assert d._calib_mode is True
        d._on_image_tap(2, 2)
        d.reset()
        assert d._calib_mode is False and d._calib_points == []
        assert events[-1] == ([], None)
        print("OK")
    """)


def test_detail_show_result_and_raw_image_end_calibration():
    _detail("""
        d.start_scale_calibration(rgb)
        d._on_image_tap(2, 2)
        d.show_raw_image(rgb, "x")
        assert d._calib_mode is False and events[-1] == ([], None)
        d.start_scale_calibration(rgb)
        d._on_image_tap(2, 2)
        d.show_result(0, [{"filename": "a.png"}])
        assert d._calib_mode is False and events[-1] == ([], None)
        assert d._current_filename == "a.png"
        d._chk_exclude.setEnabled.assert_called_with(True)
        d.start_scale_calibration(rgb)
        d.cancel_scale_calibration()
        assert d._calib_mode is False
        d._view.clear_dots.assert_called()
        print("OK")
    """)


def test_detail_start_calibration_disables_exclude_and_clears_filename():
    _detail("""
        d._current_filename = "a.png"
        d.start_scale_calibration(rgb)
        d._chk_exclude.setEnabled.assert_called_with(False)
        d._manual_row_widget.setVisible.assert_called_with(False)
        assert d._current_filename is None
        print("OK")
    """)
