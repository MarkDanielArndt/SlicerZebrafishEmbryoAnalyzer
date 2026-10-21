"""Behaviour of the self-test runner: it must report every test and still fail loudly.

Collecting failures instead of aborting is only safe as long as runTest() re-raises
at the end. Without that, unittest would count the swallowed exception as a pass and
CTest would report a green build for a broken extension.
"""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MODULE_DIR = Path(__file__).parent.parent / "ZebrafishEmbryoAnalyzer"


class _FakeTestBase(unittest.TestCase):
    """Stand-in for ScriptedLoadableModuleTest: real TestCase, silent popups."""

    def delayDisplay(self, message, requestedDelay=None, msec=None):
        pass


@pytest.fixture
def analyzer():
    """Import ZebrafishEmbryoAnalyzer.py with the Slicer API faked out.

    sys.modules is snapshotted and restored wholesale rather than patched entry
    by entry: importing the module file runs _evict_reload_modules() at import time,
    which drops ZebrafishEmbryoAnalyzerLib entries that monkeypatch never recorded
    and therefore cannot put back. Leaving that eviction in place made unrelated
    tests fail depending on file order.
    """
    saved_modules = dict(sys.modules)
    saved_path = list(sys.path)

    slicer = MagicMock()
    slm = types.ModuleType("slicer.ScriptedLoadableModule")
    slm.ScriptedLoadableModule = type("ScriptedLoadableModule", (), {})
    slm.ScriptedLoadableModuleWidget = type("ScriptedLoadableModuleWidget", (), {})
    slm.ScriptedLoadableModuleLogic = type("ScriptedLoadableModuleLogic", (), {})
    slm.ScriptedLoadableModuleTest = _FakeTestBase
    util = types.ModuleType("slicer.util")
    util.VTKObservationMixin = type("VTKObservationMixin", (), {})
    slicer.ScriptedLoadableModule = slm
    slicer.util = util

    sys.path.insert(0, str(MODULE_DIR))
    sys.modules["vtk"] = MagicMock()
    sys.modules["slicer"] = slicer
    sys.modules["slicer.ScriptedLoadableModule"] = slm
    sys.modules["slicer.util"] = util
    sys.modules.pop("ZebrafishEmbryoAnalyzer", None)

    try:
        yield importlib.import_module("ZebrafishEmbryoAnalyzer")
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _stub_tests(case_class, monkeypatch, failing=()):
    """Replace the real tests with stubs; return the list they append to when run."""
    executed = []

    def make(name):
        def stub(self):
            executed.append(name)
            if name in failing:
                raise ValueError(f"boom in {name}")
        return stub

    for name in case_class.TEST_NAMES:
        monkeypatch.setattr(case_class, name, make(name))
    return executed


def test_all_green_reports_full_count(analyzer, monkeypatch, capsys):
    case_class = analyzer.ZebrafishEmbryoAnalyzerTest
    executed = _stub_tests(case_class, monkeypatch)

    case_class().runTest()

    total = len(case_class.TEST_NAMES)
    assert executed == list(case_class.TEST_NAMES)
    assert f"{total}/{total} passed" in capsys.readouterr().out


def test_failure_is_reported_and_reraised(analyzer, monkeypatch, capsys):
    case_class = analyzer.ZebrafishEmbryoAnalyzerTest
    broken = "test_mrml_table_node_reuse"
    executed = _stub_tests(case_class, monkeypatch, failing=(broken,))

    with pytest.raises(AssertionError, match=broken):
        case_class().runTest()

    # A failure must not cut the run short — every other test still reports.
    total = len(case_class.TEST_NAMES)
    assert executed == list(case_class.TEST_NAMES)
    out = capsys.readouterr().out
    assert f"{total - 1}/{total} passed" in out
    assert f"FAIL  {broken}" in out
    assert "ValueError: boom in" in out
