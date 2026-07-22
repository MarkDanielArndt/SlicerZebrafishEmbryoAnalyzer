"""
Tests for _load_originals — reading the selected images and building their thumbnails.

This runs on the main thread, so it has to hand control back to the Qt event loop between
images. Without that the application freezes until the last file is read; on the very first
load after installing the packages that is long enough for the system busy cursor to appear.

Processing events makes a second load possible while the first is still running, which is
why the re-entrancy guard is covered here too.

Pure Python — no Slicer, Qt, OpenCV or torch required.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np


@contextmanager
def _stub_env():
    saved = {k: sys.modules[k] for k in ("slicer", "qt", "ctk", "cv2") if k in sys.modules}
    slicer = MagicMock()
    cv2 = MagicMock()
    cv2.imread.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.cvtColor.side_effect = lambda img, code: img
    cv2.resize.side_effect = lambda img, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)

    sys.modules["slicer"] = slicer
    sys.modules["qt"] = MagicMock()
    sys.modules["ctk"] = MagicMock()
    sys.modules["cv2"] = cv2
    sys.modules.pop("ZebrafishEmbryoAnalyzerLib.widget", None)
    try:
        yield slicer, cv2
    finally:
        for k in ("slicer", "qt", "ctk", "cv2", "ZebrafishEmbryoAnalyzerLib.widget"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def _widget(cls, stubs):
    w = object.__new__(cls)
    w._results = stubs
    w._gallery = MagicMock()
    w._btn_folder = MagicMock()
    w._btn_folder.text = "Load Folder…"
    w._btn_files = MagicMock()
    w._btn_files.text = "Load Images…"
    return w


def _stubs(n):
    return [{"filename": f"f{i}.png", "original": None} for i in range(n)]


def test_event_loop_gets_a_turn_between_images():
    """Without this the whole application is frozen until the last file is read."""
    with _stub_env() as (slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(3)
        w = _widget(W, stubs)

        W._load_originals(w, ["a.png", "b.png", "c.png"], stubs)

        assert slicer.app.processEvents.call_count >= 3


def test_every_image_is_loaded_and_thumbnailed():
    with _stub_env() as (_slicer, cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(3)
        w = _widget(W, stubs)

        W._load_originals(w, ["a.png", "b.png", "c.png"], stubs)

        assert cv2.imread.call_count == 3
        assert all(s["original"] is not None for s in stubs)
        assert w._gallery.update_thumb_prebuilt.call_count == 3


def test_unreadable_image_does_not_abort_the_rest():
    """cv2.imread returns None for a file it cannot decode — the remaining images must
    still load."""
    with _stub_env() as (_slicer, cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(3)
        w = _widget(W, stubs)
        cv2.imread.side_effect = [None, np.zeros((10, 10, 3), dtype=np.uint8),
                                  np.zeros((10, 10, 3), dtype=np.uint8)]

        W._load_originals(w, ["broken.png", "b.png", "c.png"], stubs)

        assert stubs[0]["original"] is None
        assert stubs[1]["original"] is not None
        assert stubs[2]["original"] is not None


def test_a_second_load_aborts_the_first():
    """Processing events lets the user start another load mid-run. The first one must stop
    instead of writing into the replaced result list."""
    with _stub_env() as (slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(5)
        w = _widget(W, stubs)

        # Simulate the user triggering a new load during the first processEvents call.
        def replace_results():
            w._results = _stubs(2)
        slicer.app.processEvents.side_effect = replace_results

        W._load_originals(w, [f"{i}.png" for i in range(5)], stubs)

        # First image was written before the swap; the loop then bailed out.
        assert stubs[0]["original"] is not None
        assert all(s["original"] is None for s in stubs[1:])


def test_progress_is_reported_on_the_button_that_started_the_load():
    """Feedback belongs where the user clicked. The restored label is also the signal that
    the load has finished — on a large folder there is otherwise no way to tell a finished
    load from a stalled one."""
    with _stub_env() as (_slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(3)
        w = _widget(W, stubs)

        W._load_originals(w, ["a.png", "b.png", "c.png"], stubs, w._btn_folder)

        texts = [c.args[0] for c in w._btn_folder.setText.call_args_list]
        assert texts[:3] == ["Loading… 1/3", "Loading… 2/3", "Loading… 3/3"]
        assert texts[-1] == "Load Folder…"          # original label restored
        w._btn_files.setText.assert_called_once_with("Load Images…")


def test_both_buttons_are_disabled_while_loading_and_restored_after():
    with _stub_env() as (_slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(2)
        w = _widget(W, stubs)

        W._load_originals(w, ["a.png", "b.png"], stubs, w._btn_files)

        for btn in (w._btn_folder, w._btn_files):
            assert [c.args[0] for c in btn.setEnabled.call_args_list] == [False, True]


def test_buttons_are_restored_even_when_the_load_is_aborted():
    with _stub_env() as (slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        stubs = _stubs(5)
        w = _widget(W, stubs)
        slicer.app.processEvents.side_effect = lambda: setattr(w, "_results", _stubs(2))

        W._load_originals(w, [f"{i}.png" for i in range(5)], stubs, w._btn_folder)

        assert w._btn_folder.setText.call_args_list[-1].args[0] == "Load Folder…"
        assert w._btn_folder.setEnabled.call_args_list[-1].args[0] is True


def test_empty_selection_touches_no_button():
    with _stub_env() as (_slicer, _cv2):
        from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget as W
        w = _widget(W, [])
        W._load_originals(w, [], [], w._btn_folder)
        w._btn_folder.setEnabled.assert_not_called()
