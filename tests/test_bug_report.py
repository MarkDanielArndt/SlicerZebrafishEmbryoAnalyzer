import sys
from unittest.mock import patch


def test_build_report_contains_expected_sections():
    # Other test modules stub sys.modules["slicer"] at import time and don't
    # always clean up, so this must not assume anything about its shape —
    # build_report() must never raise regardless of what "slicer" is.
    from ZebrafishEmbryoAnalyzerLib.bug_report import build_report
    report = build_report()
    assert "## Environment" in report
    assert "## Package versions" in report
    assert "## Steps to reproduce" in report
    assert "- numpy:" in report


def test_build_report_falls_back_outside_slicer(monkeypatch):
    # Force the real ImportError path even if an earlier test left a stub
    # "slicer" module behind in sys.modules.
    monkeypatch.delitem(sys.modules, "slicer", raising=False)
    from ZebrafishEmbryoAnalyzerLib.bug_report import build_report
    report = build_report()
    assert "not running inside Slicer" in report


def test_extension_commit_falls_back_when_git_unavailable():
    from ZebrafishEmbryoAnalyzerLib import bug_report

    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    with patch("subprocess.run", side_effect=_boom):
        assert bug_report._extension_commit() == "unknown (not a git checkout)"


def test_extension_commit_returns_short_sha_in_this_repo():
    from ZebrafishEmbryoAnalyzerLib.bug_report import _extension_commit
    commit = _extension_commit()
    assert commit != "unknown (not a git checkout)"
    assert len(commit) >= 7
