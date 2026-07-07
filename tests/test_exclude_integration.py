"""
Behavioral tests for exclude functionality integrated into ResultsTab and DetailTab.

These tests run without Slicer/Qt by using lightweight stubs.
"""

import sys
import types
import pytest
from pathlib import Path

# Mirror the runtime sys.path setup used by conftest/other tests
_MODULE_DIR = Path(__file__).resolve().parent.parent / "ZebrafishEmbryoAnalyzer"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


# ---------------------------------------------------------------------------
# Qt stub — minimal surface used by results_tab and detail_tab
# ---------------------------------------------------------------------------

class _Signal:
    def __init__(self, owner):
        self._owner = owner
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, value):
        if not self._owner._signals_blocked:
            for cb in self._callbacks:
                cb(value)


class _CheckBox:
    def __init__(self):
        self._checked = False
        self._enabled = True
        self._signals_blocked = False
        self.toggled = _Signal(self)

    def setChecked(self, v):
        old = self._checked
        self._checked = bool(v)
        if old != self._checked:
            self.toggled.emit(self._checked)

    def isChecked(self):
        return self._checked

    def setEnabled(self, v):
        self._enabled = bool(v)

    def isEnabled(self):
        return self._enabled

    def blockSignals(self, v):
        self._signals_blocked = bool(v)


class _TableWidgetItem:
    def __init__(self, text=""):
        self.text = text


class _TableWidget:
    def __init__(self, rows, cols):
        self.rowCount = rows
        self._cols = cols
        self._items = {}
        self._widgets = {}
        self._header = _Header()

    def setHorizontalHeaderLabels(self, labels):
        pass

    def horizontalHeader(self):
        return self._header

    def setItem(self, row, col, item):
        self._items[(row, col)] = item

    def setCellWidget(self, row, col, widget):
        self._widgets[(row, col)] = widget

    def cellWidget(self, row, col):
        return self._widgets.get((row, col))

    editTriggers = None
    selectionBehavior = None


class _Header:
    def setSectionResizeMode(self, col, mode):
        pass


class _VBoxLayout:
    def __init__(self, parent=None):
        pass

    def addWidget(self, w, *args):
        pass

    def addLayout(self, l, *args):
        pass


class _QWidget:
    def __init__(self, *args):
        pass

    def setFocusPolicy(self, p):
        pass


_qt_stub = types.ModuleType("qt")
_qt_stub.QWidget = _QWidget
_qt_stub.QTableWidget = _TableWidget
_qt_stub.QTableWidgetItem = _TableWidgetItem
_qt_stub.QCheckBox = _CheckBox
_qt_stub.QVBoxLayout = _VBoxLayout
_qt_stub.QHBoxLayout = _VBoxLayout
_qt_stub.QHeaderView = types.SimpleNamespace(Stretch=0)
_qt_stub.QAbstractItemView = types.SimpleNamespace(
    NoEditTriggers=0, SelectRows=0
)
_qt_stub.QLabel = lambda *a, **kw: types.SimpleNamespace(
    setText=lambda *a: None,
    setWordWrap=lambda *a: None,
    setStyleSheet=lambda *a: None,
    setAlignment=lambda *a: None,
)
_qt_stub.QPushButton = lambda *a, **kw: types.SimpleNamespace(
    clicked=types.SimpleNamespace(connect=lambda *a: None),
    setFixedWidth=lambda *a: None,
    setFixedHeight=lambda *a: None,
    setStyleSheet=lambda *a: None,
    setEnabled=lambda *a: None,
    setVisible=lambda *a: None,
    setText=lambda *a: None,
)
_qt_stub.Qt = types.SimpleNamespace(AlignCenter=0, StrongFocus=0)
_qt_stub.QTimer = types.SimpleNamespace(singleShot=lambda *a: None)

sys.modules.setdefault("qt", _qt_stub)

# Stub out heavy imports pulled in by detail_tab
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
for _mod in ("ZebrafishEmbryoAnalyzerLib.zoom_view", "ZebrafishEmbryoAnalyzerLib.overlay"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# Now import the modules under test (without Slicer)
from ZebrafishEmbryoAnalyzerLib import results_tab  # noqa: E402


# ---------------------------------------------------------------------------
# ResultsTab tests
# ---------------------------------------------------------------------------

def _make_result(filename, error=None):
    return {"filename": filename, "length": 100.0, "curvature": "straight",
            "ratio": 1.0, "eye_area": None, "eye_diameter": None, "error": error}


class TestResultsTabExcludeColumn:
    def test_exclude_column_header(self):
        tab = results_tab.ResultsTab()
        # _EXCL_COL should be the last column
        assert results_tab._EXCL_COL == len(results_tab.COLUMNS)

    def test_populate_no_errors_unchecked(self):
        calls = []
        tab = results_tab.ResultsTab(on_exclude_change=lambda fn, ch: calls.append((fn, ch)))
        tab.populate([_make_result("a.png")])
        assert tab._rows[0][0] == "a.png"
        assert tab._rows[0][1].isChecked() is False

    def test_populate_error_row_auto_checked(self):
        tab = results_tab.ResultsTab()
        tab.populate([_make_result("bad.png", error="timeout")])
        assert tab._rows[0][1].isChecked() is True

    def test_populate_previously_excluded(self):
        tab = results_tab.ResultsTab()
        tab.populate([_make_result("ok.png")], excluded={"ok.png"})
        assert tab._rows[0][1].isChecked() is True

    def test_get_excluded_returns_checked(self):
        tab = results_tab.ResultsTab()
        tab.populate([_make_result("a.png"), _make_result("b.png", error="err")])
        result = tab.get_excluded()
        assert result == {"b.png"}

    def test_sync_exclude_updates_without_callback(self):
        calls = []
        tab = results_tab.ResultsTab(on_exclude_change=lambda fn, ch: calls.append((fn, ch)))
        tab.populate([_make_result("a.png"), _make_result("b.png")])
        calls.clear()
        tab.sync_exclude({"a.png"})
        # No callbacks should have fired
        assert calls == []
        assert tab._rows[0][1].isChecked() is True
        assert tab._rows[1][1].isChecked() is False

    def test_checkbox_fires_callback(self):
        calls = []
        tab = results_tab.ResultsTab(on_exclude_change=lambda fn, ch: calls.append((fn, ch)))
        tab.populate([_make_result("x.png")])
        # Simulate user checking the box
        tab._rows[0][1].setChecked(True)
        assert ("x.png", True) in calls

    def test_populate_empty_clears_rows(self):
        tab = results_tab.ResultsTab()
        tab.populate([_make_result("a.png")])
        tab.populate([])
        assert tab._rows == []
        assert tab._table.rowCount == 0
