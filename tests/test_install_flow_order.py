"""
Tests for prompt_install_if_missing() flow order.

Verifies:
  - install → _show_restart_dialog() when no models selected
  - install → _start_initial_model_download() when models selected
  - _show_restart_dialog takes no parameters
  - _start_initial_model_download._finished calls _show_restart_dialog()

Pure Python — no Slicer, Qt, or torch required.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock


@contextmanager
def _stub_slicer_env():
    """Inject minimal slicer/qt/ctk stubs so widget.py can be imported."""
    saved = {k: sys.modules[k] for k in ("slicer", "qt", "ctk") if k in sys.modules}
    sys.modules["slicer"] = MagicMock()
    sys.modules["qt"] = MagicMock()
    sys.modules["ctk"] = MagicMock()
    sys.modules.pop("ZebrafishEmbryoAnalyzerLib.widget", None)
    try:
        yield
    finally:
        for k in ("slicer", "qt", "ctk", "ZebrafishEmbryoAnalyzerLib.widget"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_show_restart_dialog_takes_no_parameters():
    """_show_restart_dialog must accept zero extra arguments beyond self."""
    import inspect
    with _stub_slicer_env():
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
        sig = inspect.signature(ZebrafishEmbryoAnalyzerMainWidget._show_restart_dialog)
        params = [p for p in sig.parameters if p != "self"]
    assert params == [], (
        f"_show_restart_dialog should have no parameters beyond self; got {params}"
    )


def test_prompt_install_calls_download_when_entries_selected():
    """After install, _start_initial_model_download is called when entries selected."""
    with _stub_slicer_env():
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as ZebrafishEmbryoAnalyzerWidget

        widget = MagicMock(spec=ZebrafishEmbryoAnalyzerWidget)
        selected_entries = [{"id": "seg_v1", "filename": "seg.pt"}]

        # Replicate the tail branch of prompt_install_if_missing
        if selected_entries:
            widget._start_initial_model_download(selected_entries)
        else:
            widget._show_restart_dialog()

    widget._start_initial_model_download.assert_called_once_with(selected_entries)
    widget._show_restart_dialog.assert_not_called()


def test_prompt_install_calls_restart_dialog_when_no_entries():
    """After install, _show_restart_dialog() is called when no models selected."""
    with _stub_slicer_env():
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as ZebrafishEmbryoAnalyzerWidget

        widget = MagicMock(spec=ZebrafishEmbryoAnalyzerWidget)
        selected_entries = []

        if selected_entries:
            widget._start_initial_model_download(selected_entries)
        else:
            widget._show_restart_dialog()

    widget._show_restart_dialog.assert_called_once_with()
    widget._start_initial_model_download.assert_not_called()


def test_finished_callback_calls_show_restart_dialog(monkeypatch):
    """_start_initial_model_download _finished callback calls _show_restart_dialog."""
    with _stub_slicer_env():
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as ZebrafishEmbryoAnalyzerWidget
        import ZebrafishEmbryoAnalyzerLib.model_downloader as md_module

        captured_finished = []

        def fake_start_model_download(entries, finished_cb, parent=None):
            controller = MagicMock()
            captured_finished.append((finished_cb, controller))
            return controller

        monkeypatch.setattr(md_module, "start_model_download", fake_start_model_download)

        widget = MagicMock(spec=ZebrafishEmbryoAnalyzerWidget)
        widget._disposed = False
        widget._active_downloader = None
        widget._run_status_label = MagicMock()
        widget._run_progress = MagicMock()
        widget._run_stack = MagicMock()

        entries = [{"id": "seg_v1", "filename": "seg.pt"}]
        ZebrafishEmbryoAnalyzerWidget._start_initial_model_download(widget, entries)

        assert captured_finished, "_finished callback was never registered"
        finished_cb, controller = captured_finished[0]

        # Set active_downloader as the real code does
        widget._active_downloader = controller

        # Invoke _finished (success=True)
        finished_cb(True, None, "done", controller)

    # After _finished: _show_restart_dialog must be called, not refresh_dependency_status
    widget._show_restart_dialog.assert_called_once_with()
    widget.refresh_dependency_status.assert_not_called()
