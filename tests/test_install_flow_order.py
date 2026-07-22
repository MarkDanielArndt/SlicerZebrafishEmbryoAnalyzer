"""
Tests for ensure_dependencies() — the on-demand dependency install flow.

Verifies:
  - opening the module never triggers an install (no call site in enter())
  - nothing missing → the caller proceeds
  - user declines → the caller does not proceed and nothing is installed
  - install needing no restart → the caller carries on in the same session
  - install replacing something already imported → restart dialog, caller stops
  - a failed install is reported and does not proceed
  - the confirmation and the restart use Slicer's standard dialogs
  - a failing pip install is left to Slicer's own dialog, which carries the full log

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


def _widget_class():
    from ZebrafishEmbryoAnalyzerLib.widget import ZebrafishEmbryoAnalyzerMainWidget
    return ZebrafishEmbryoAnalyzerMainWidget


def _shell(cls):
    w = object.__new__(cls)
    w._show_restart_dialog = MagicMock()
    w.refresh_dependency_status = MagicMock()
    return w


def test_show_restart_dialog_takes_no_parameters():
    """_show_restart_dialog must accept zero extra arguments beyond self."""
    import inspect
    with _stub_slicer_env():
        sig = inspect.signature(_widget_class()._show_restart_dialog)
        params = [p for p in sig.parameters if p != "self"]
    assert params == [], (
        f"_show_restart_dialog should have no parameters beyond self; got {params}"
    )


def test_module_entry_notifies_but_never_installs():
    """Opening the module refreshes the in-panel notice so the user learns about a
    pending install early, but must not open a dialog or install anything."""
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzer.py").read_text()
    enter_body = src.split("def enter(self)")[1].split("def exit(self)")[0]
    # Strip comments — enter() explains in prose why it does not call the installer.
    code = "\n".join(line.split("#")[0] for line in enter_body.splitlines())
    assert "ensure_dependencies(" not in code
    assert "refresh_dependency_status(" in code
    assert "prompt_install_if_missing" not in src


def test_proceeds_when_nothing_missing(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": []})
        sys.modules["slicer"].app.testingEnabled.return_value = False

        assert cls.ensure_dependencies(w, "analysis") is True
        w._show_restart_dialog.assert_not_called()


def test_declining_stops_the_caller(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = False

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        install = MagicMock()
        monkeypatch.setattr(di, "install_packages", install)

        assert cls.ensure_dependencies(w, "analysis") is False
        install.assert_not_called()
        w._show_restart_dialog.assert_not_called()


def test_install_that_needs_no_restart_lets_the_caller_continue(monkeypatch):
    """A freshly installed package that this session never imported is usable straight
    away, so the action that triggered the install should simply carry on."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = True

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        monkeypatch.setattr(di, "install_packages", MagicMock(return_value="ready"))

        assert cls.ensure_dependencies(w, "analysis") is True
        w._show_restart_dialog.assert_not_called()


def test_install_that_needs_a_restart_stops_the_caller(monkeypatch):
    """Only when something already held in memory was replaced — numpy in practice —
    must the action stop and the restart be offered."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = True

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        monkeypatch.setattr(di, "install_packages", MagicMock(return_value="restart"))

        assert cls.ensure_dependencies(w, "analysis") is False
        w._show_restart_dialog.assert_called_once_with()


def test_failed_install_is_reported(monkeypatch):
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = True

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(di, "get_missing_packages",
                            lambda purpose="analysis": {"torch": [], "general": ["timm"]})
        monkeypatch.setattr(di, "install_packages",
                            MagicMock(side_effect=RuntimeError("boom")))

        assert cls.ensure_dependencies(w, "analysis") is False
        slicer.util.errorDisplay.assert_called_once()
        w._show_restart_dialog.assert_not_called()


def test_confirmation_uses_the_standard_dialog_with_package_details(monkeypatch):
    """Tier 3 asks for minimal popups and no unnecessary custom GUI — the package list
    belongs in the standard dialog's detail area, not in a hand-built QDialog."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = _shell(cls)
        slicer = sys.modules["slicer"]
        slicer.app.testingEnabled.return_value = False
        slicer.util.confirmOkCancelDisplay.return_value = False

        import ZebrafishEmbryoAnalyzerLib.dependency_installer as di
        monkeypatch.setattr(
            di, "get_missing_packages",
            lambda purpose="analysis": {"torch": ["torch"], "general": ["timm"]})

        cls.ensure_dependencies(w, "analysis")

        slicer.util.confirmOkCancelDisplay.assert_called_once()
        detail = slicer.util.confirmOkCancelDisplay.call_args.kwargs["detailedText"]
        assert "timm" in detail
        assert "torch" in detail


def test_restart_uses_the_standard_dialog():
    """Both reference extensions confirm the restart with slicer.util.confirmOkCancelDisplay
    and then call slicer.util.restart(); no hand-built dialog."""
    with _stub_slicer_env():
        cls = _widget_class()
        w = object.__new__(cls)
        w._status_log = None
        slicer = sys.modules["slicer"]
        slicer.util.confirmOkCancelDisplay.return_value = True

        cls._show_restart_dialog(w)

        slicer.util.confirmOkCancelDisplay.assert_called_once()
        slicer.util.restart.assert_called_once()


def test_declining_the_restart_refreshes_status_instead():
    with _stub_slicer_env():
        cls = _widget_class()
        w = object.__new__(cls)
        w._status_log = None
        w.refresh_dependency_status = MagicMock()
        slicer = sys.modules["slicer"]
        slicer.util.confirmOkCancelDisplay.return_value = False

        cls._show_restart_dialog(w)

        slicer.util.restart.assert_not_called()
        w.refresh_dependency_status.assert_called_once()


def test_no_custom_restart_dialog_remains():
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerLib/widget.py").read_text()
    assert "Restart Required" not in src
    assert "Restart Now" not in src


def test_no_custom_setup_dialog_remains():
    """The hand-built setup dialog and its model checkboxes are gone; models download
    on demand at the point they are first needed."""
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerLib/widget.py").read_text()
    assert "ZebrafishEmbryoAnalyzer — Setup" not in src
    assert "_start_initial_model_download" not in src
    assert "_install_declined" not in src


def test_image_loading_checks_dependencies_first():
    """Loading a folder crashed with ModuleNotFoundError: cv2 before this guard existed —
    the gallery renders every thumbnail through overlay.py, which imports cv2."""
    from pathlib import Path
    src = Path("ZebrafishEmbryoAnalyzer/ZebrafishEmbryoAnalyzerLib/widget.py").read_text()
    for handler in ("_on_load_folder", "_on_load_files"):
        body = src.split(f"def {handler}(self)")[1].split("\n    def ")[0]
        assert 'ensure_dependencies("images")' in body, f"{handler} loads images unguarded"
        # The cancel check must come first — while a load runs the button is an abort,
        # and asking to install packages at that moment would make no sense.
        assert body.index("_cancel_load_if_running") < body.index("ensure_dependencies")
