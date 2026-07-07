"""
Results tab — QTableWidget showing all measurements.
"""

import qt


COLUMNS = [
    ("Filename",              "filename",     str),
    ("Length (µm)",           "length",       lambda v: f"{v:.1f}" if v is not None else ""),
    ("Curvature class",       "curvature",    lambda v: str(v) if v is not None else ""),
    ("Length/straight ratio", "ratio",        lambda v: f"{v:.3f}" if v is not None else ""),
    ("Eye area (µm²)",        "eye_area",     lambda v: f"{v:.1f}" if v is not None else ""),
    ("Eye diameter (µm)",     "eye_diameter", lambda v: f"{v:.1f}" if v is not None else ""),
    ("Error",                 "error",        lambda v: v or ""),
]

_EXCL_COL = len(COLUMNS)  # index of the Exclude checkbox column


class ResultsTab(qt.QWidget):
    def __init__(self, on_exclude_change=None):
        super().__init__()
        self._on_exclude_change = on_exclude_change  # callable(filename, checked)
        self._rows = []  # list of (filename, QCheckBox)

        self._table = qt.QTableWidget(0, len(COLUMNS) + 1)
        headers = [c[0] for c in COLUMNS] + ["Exclude"]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setSectionResizeMode(
            0, qt.QHeaderView.Stretch
        )
        self._table.editTriggers = qt.QAbstractItemView.NoEditTriggers
        self._table.selectionBehavior = qt.QAbstractItemView.SelectRows

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._table)

    def populate(self, results, excluded=None) -> None:
        if excluded is None:
            excluded = set()
        n = len(results)
        self._table.rowCount = n
        self._rows = []
        for row in range(n):
            r = results[row]
            for col in range(len(COLUMNS)):
                _, key, fmt = COLUMNS[col]
                val = r.get(key)
                self._table.setItem(row, col, qt.QTableWidgetItem(fmt(val)))
            # Exclude checkbox — auto-check error rows or previously excluded
            chk = qt.QCheckBox()
            is_excluded = r["filename"] in excluded or bool(r.get("error"))
            chk.setChecked(is_excluded)
            filename = r["filename"]
            chk.toggled.connect(
                lambda checked, fn=filename: self._on_exclude_change and self._on_exclude_change(fn, checked)
            )
            self._table.setCellWidget(row, _EXCL_COL, chk)
            self._rows.append((filename, chk))

    def sync_exclude(self, excluded: set) -> None:
        """Update checkbox states from outside without firing callbacks."""
        for fn, chk in self._rows:
            chk.blockSignals(True)
            chk.setChecked(fn in excluded)
            chk.blockSignals(False)

    def get_excluded(self) -> set:
        return {fn for fn, chk in self._rows if chk.isChecked()}
