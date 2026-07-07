"""Tests for explicit model readiness and download prompting."""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def _stub_slicer_env(testing_enabled=False, user_answer=0x10000):
    qt_mock = MagicMock()
    qt_mock.QMessageBox.Yes = 0x4000
    qt_mock.QMessageBox.No = 0x10000
    msg_instance = MagicMock()
    msg_instance.exec_.return_value = user_answer
    qt_mock.QMessageBox.return_value = msg_instance

    slicer_mock = MagicMock()
    slicer_mock.app.testingEnabled.return_value = testing_enabled

    saved = {k: sys.modules[k] for k in ("slicer", "qt", "ctk") if k in sys.modules}
    sys.modules["slicer"] = slicer_mock
    sys.modules["qt"] = qt_mock
    sys.modules["ctk"] = MagicMock()
    sys.modules.pop("ZebrafishEmbryoAnalyzerLib.widget", None)
    try:
        yield qt_mock, msg_instance, slicer_mock
    finally:
        for k in ("slicer", "qt", "ctk", "ZebrafishEmbryoAnalyzerLib.widget"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def _make_checkbox(checked=False):
    chk = MagicMock()
    chk.isChecked.return_value = checked
    return chk


def _make_widget(eyes_checked=False, curvature_checked=True):
    from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget

    widget = object.__new__(ZebrafishEmbryoAnalyzerMainWidget)
    widget._chk_eyes = _make_checkbox(checked=eyes_checked)
    widget._chk_curvature = _make_checkbox(checked=curvature_checked)
    return widget


def test_missing_models_uses_get_missing_models():
    with _stub_slicer_env():
        import ZebrafishEmbryoAnalyzerLib.model_manifest as manifest

        get_missing = MagicMock(return_value=[])
        with patch.object(manifest, "get_missing_models", get_missing):
            widget = _make_widget()
            assert widget._missing_required_models("general") == []

        called_dict = get_missing.call_args[0][0]
        assert "body" in called_dict
        assert "curvature" in called_dict
        assert "eye" not in called_dict


def test_prompt_suppressed_in_testing_mode():
    with _stub_slicer_env(testing_enabled=True, user_answer=0x4000) as (
        qt_mock,
        msg,
        slicer_mock,
    ):
        widget = _make_widget()
        assert widget._prompt_download_models([{"label": "Body"}]) is False
        qt_mock.QMessageBox.assert_not_called()


def test_prompt_user_declines_returns_false():
    with _stub_slicer_env(user_answer=0x10000) as (qt_mock, msg, slicer_mock):
        widget = _make_widget()
        assert widget._prompt_download_models([{"label": "Body", "size_bytes": 1000}]) is False
        qt_mock.QMessageBox.assert_called_once()


def test_prompt_user_accepts_returns_true_without_downloading():
    with _stub_slicer_env(user_answer=0x4000) as (qt_mock, msg, slicer_mock):
        widget = _make_widget()
        assert widget._prompt_download_models([{"label": "Body", "size_bytes": 1000}]) is True
        qt_mock.QMessageBox.assert_called_once()


def test_eyes_unchecked_eye_model_not_required():
    with _stub_slicer_env():
        widget = _make_widget(eyes_checked=False)
        required = widget._required_model_entries("general")
        assert "eye" not in required
        assert "body" in required
        assert "curvature" in required


def test_eyes_checked_eye_model_required():
    with _stub_slicer_env():
        widget = _make_widget(eyes_checked=True)
        required = widget._required_model_entries("general")
        assert "eye" in required
        assert "body" in required
        assert "curvature" in required


def test_curvature_unchecked_not_required():
    with _stub_slicer_env():
        widget = _make_widget(curvature_checked=False)
        required = widget._required_model_entries("general")
        assert "curvature" not in required
        assert "body" in required


def test_curvature_checked_is_required():
    with _stub_slicer_env():
        widget = _make_widget(curvature_checked=True)
        required = widget._required_model_entries("general")
        assert "curvature" in required
        assert "body" in required
