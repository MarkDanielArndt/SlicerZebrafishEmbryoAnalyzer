"""
Detail tab — full-resolution overlay + metrics for the selected image.

show_result(index, results) — display result at index.
"""

import qt
import numpy as np
from ZebrafishEmbryoAnalyzerLib.zoom_view import ZoomableImageView


def _numpy_to_qpixmap(rgb_array: np.ndarray) -> "qt.QPixmap":
    from PIL import Image as PILImage
    import io
    arr = np.ascontiguousarray(rgb_array.clip(0, 255).astype("uint8"))
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="BMP")  # BMP: no compression, fast encode
    data = qt.QByteArray(buf.getvalue())
    pixmap = qt.QPixmap()
    pixmap.loadFromData(data)
    return pixmap


def _build_rgb_array(result: dict) -> np.ndarray:
    """Build the RGB overlay array synchronously on the main thread."""
    from ZebrafishEmbryoAnalyzerLib.overlay import make_full_overlay
    import cv2
    bgr = make_full_overlay(result)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DetailTab(qt.QWidget):
    def __init__(self, on_navigate=None, on_back=None, logic=None, on_exclude_change=None):
        super().__init__()
        self._on_navigate = on_navigate
        self._on_back = on_back
        self._logic = logic
        self._on_exclude_change = on_exclude_change  # callable(filename, checked)
        self._current_filename = None
        self._full_pixmap = None
        self._results = []
        self._current_idx = 0
        self._cache = {}          # index → QPixmap  (main thread only)
        self._pending_reset_zoom = True  # True=reset zoom on next pixmap update
        self.setFocusPolicy(qt.Qt.StrongFocus)

        self._manual_mode = False
        self._manual_points = []   # list of (row, col) in original image space
        self._params_getter = None  # set by widget after construction

        # Main image viewer
        self._view = ZoomableImageView()
        self._view._on_navigate = self._on_navigate
        self._view._tap_handler = self._on_image_tap

        self._btn_manual_adjust = qt.QPushButton("✏ Manual Adjust")
        self._btn_revert_auto = qt.QPushButton("↩ Revert to Auto")
        self._btn_revert_auto.setVisible(False)
        self._manual_status = qt.QLabel("")
        self._manual_status.setAlignment(qt.Qt.AlignCenter)
        self._manual_status.setStyleSheet("font-size: 11px; color: #aaa; padding: 2px;")
        self._manual_status.setVisible(False)

        self._btn_manual_adjust.clicked.connect(self._on_manual_adjust_clicked)
        self._btn_revert_auto.clicked.connect(self._on_revert_auto_clicked)

        # Navigation buttons
        self._btn_prev = qt.QPushButton("◄")
        self._btn_next = qt.QPushButton("►")
        self._nav_label = qt.QLabel("")
        self._nav_label.setAlignment(qt.Qt.AlignCenter)

        for btn in (self._btn_prev, self._btn_next):
            btn.setFixedWidth(48)
            btn.setFixedHeight(32)
            btn.setStyleSheet("font-size: 16px;")

        self._btn_prev.clicked.connect(lambda: self._on_navigate and self._on_navigate(-1))
        self._btn_next.clicked.connect(lambda: self._on_navigate and self._on_navigate(1))

        # Exclude checkbox
        self._chk_exclude = qt.QCheckBox("Exclude from export")
        self._chk_exclude.setEnabled(False)
        self._chk_exclude.toggled.connect(self._on_exclude_toggled)

        # Metrics label
        self._metrics_label = qt.QLabel("")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("font-size: 12px; padding: 4px;")

        _nav_row = qt.QHBoxLayout()
        _nav_row.addStretch(1)
        _nav_row.addWidget(self._btn_prev)
        _nav_row.addWidget(self._nav_label)
        _nav_row.addWidget(self._btn_next)
        _nav_row.addStretch(1)

        _manual_row = qt.QHBoxLayout()
        _manual_row.addStretch(1)
        _manual_row.addWidget(self._btn_manual_adjust)
        _manual_row.addWidget(self._btn_revert_auto)
        _manual_row.addStretch(1)

        # Container widget so we can hide the entire row reliably in PythonQt
        self._manual_row_widget = qt.QWidget()
        self._manual_row_widget.setLayout(_manual_row)
        self._manual_row_widget.setVisible(False)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._manual_row_widget, 0)
        layout.addWidget(self._manual_status, 0)
        layout.addLayout(_nav_row, 0)
        layout.addWidget(self._chk_exclude, 0)
        layout.addWidget(self._metrics_label, 0)

        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_result(self, index: int, results: list) -> None:
        # Exit manual mode when navigating to a new image
        if self._manual_mode:
            self._manual_mode = False
            self._manual_points = []
            self._manual_status.setVisible(False)
            self._view.set_manual_mode(False)
            self._view.clear_dots()

        self._results = results
        self._current_idx = index
        result = results[index]
        self._current_filename = result["filename"]
        self._chk_exclude.setEnabled(True)

        self._metrics_label.setText(_format_metrics(result))

        # Sync button state — only show after analysis (stubs have length=None)
        analyzed = result.get("length") is not None or result.get("mask") is not None
        self._manual_row_widget.setVisible(analyzed)
        is_corrected = bool(result.get("manual_corrected"))
        self._btn_revert_auto.setVisible(is_corrected)
        self._btn_manual_adjust.setText(
            "✏ Redo Manual" if is_corrected else "✏ Manual Adjust"
        )
        self._manual_status.setText("")
        self._manual_status.setVisible(False)

        self._pending_reset_zoom = True  # navigation → always reset zoom
        if index not in self._cache:
            self._cache[index] = _numpy_to_qpixmap(_build_rgb_array(results[index]))
        self._full_pixmap = self._cache[index]
        qt.QTimer.singleShot(0, self._update_display)

        self._update_nav_state()

    def show_raw_image(self, rgb: np.ndarray, caption: str = "") -> None:
        """Display an arbitrary RGB numpy array — used for scalebar preview."""
        self._results = []
        self._current_idx = 0
        self._cache.clear()
        self._pending_reset_zoom = True  # preview always resets zoom to fit
        self._manual_mode = False
        self._manual_points = []
        self._view.set_manual_mode(False)
        self._view.clear_dots()
        self._full_pixmap = _numpy_to_qpixmap(rgb)
        self._metrics_label.setText(caption)
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._nav_label.setText("")
        qt.QTimer.singleShot(0, self._update_display)

    def invalidate_cache(self):
        """Call after a new batch run so stale pixmaps are discarded."""
        self._cache.clear()

    def sync_exclude(self, is_excluded: bool) -> None:
        """Update exclude checkbox state without firing callbacks."""
        self._chk_exclude.blockSignals(True)
        self._chk_exclude.setChecked(is_excluded)
        self._chk_exclude.blockSignals(False)

    def reset(self):
        """Clear all visible and internal state after scene close."""
        self.invalidate_cache()
        self._results = []
        self._current_idx = 0
        self._current_filename = None
        self._full_pixmap = None
        self._manual_mode = False
        self._manual_points = []
        self._view.show_placeholder("Select an image from the Gallery.")
        self._view.set_manual_mode(False)
        self._metrics_label.setText("")
        self._nav_label.setText("")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._manual_row_widget.setVisible(False)
        self._manual_status.setVisible(False)
        self._chk_exclude.blockSignals(True)
        self._chk_exclude.setChecked(False)
        self._chk_exclude.setEnabled(False)
        self._chk_exclude.blockSignals(False)

    def cleanup(self):
        """Invalidate cache."""
        self.invalidate_cache()

    def _on_exclude_toggled(self, checked: bool) -> None:
        if self._current_filename and self._on_exclude_change:
            self._on_exclude_change(self._current_filename, checked)

    def _update_nav_state(self) -> None:
        n = len(self._results)
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < n - 1)
        if n > 0:
            self._nav_label.setText(f"{self._current_idx + 1} / {n}")
        else:
            self._nav_label.setText("")

    def _ensure_cached(self, index: int) -> None:
        """Build and cache the pixmap for index synchronously if not yet cached."""
        if index in self._cache:
            return
        if index < 0 or index >= len(self._results):
            return
        self._cache[index] = _numpy_to_qpixmap(_build_rgb_array(self._results[index]))

    def _start_job(self, index: int) -> None:
        """Synchronously rebuild the overlay for index and update the display."""
        self._cache.pop(index, None)
        self._ensure_cached(index)
        if index in self._cache:
            self._full_pixmap = self._cache[index]
            qt.QTimer.singleShot(0, self._update_display)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        self._view.set_pixmap(self._full_pixmap, reset_zoom=self._pending_reset_zoom)

    def keyPressEvent(self, event):
        if event.key() == qt.Qt.Key_Escape:
            if self._on_back:
                self._on_back()
            return
        if self._on_navigate:
            if event.key() == qt.Qt.Key_Right:
                self._on_navigate(1)
                return
            if event.key() == qt.Qt.Key_Left:
                self._on_navigate(-1)
                return
        qt.QWidget.keyPressEvent(self, event)

    # ------------------------------------------------------------------
    # Manual correction — tap mode
    # ------------------------------------------------------------------

    def _on_manual_adjust_clicked(self):
        """Enter tap mode to place head/tail points."""
        if not self._results:
            return
        self._manual_mode = True
        self._manual_points = []
        self._manual_status.setText("Click HEAD point (1/2)")
        self._manual_status.setVisible(True)
        self._view.set_manual_mode(True)
        self._view.clear_dots()

    def _on_revert_auto_clicked(self):
        """Restore auto-computed values for current fish."""
        if not self._results:
            return
        result = self._results[self._current_idx]
        self._logic.revert_manual_correction(result)

        self._cache.pop(self._current_idx, None)

        self._manual_mode = False
        self._manual_points = []
        self._manual_status.setText("Reverted to auto.")
        self._manual_status.setVisible(True)
        self._btn_revert_auto.setVisible(False)
        self._btn_manual_adjust.setText("✏ Manual Adjust")
        self._metrics_label.setText(_format_metrics(result))
        self._view.set_manual_mode(False)
        self._view.clear_dots()

        self._pending_reset_zoom = False  # preserve zoom after correction rebuild
        self._start_job(self._current_idx)

    def _on_image_tap(self, row: int, col: int) -> None:
        """Called by ZoomableImageView tap handler with pre-mapped (row, col) coords."""
        if not self._manual_mode:
            return

        self._manual_points.append((row, col))
        self._view.add_dot(
            row, col,
            qt.QColor(0, 220, 0) if len(self._manual_points) == 1 else qt.QColor(220, 0, 0)
        )

        if len(self._manual_points) == 1:
            self._manual_status.setText("Click TAIL point (2/2)")
        elif len(self._manual_points) >= 2:
            self._manual_mode = False
            self._view.set_manual_mode(False)
            self._manual_status.setText("Computing…")
            qt.QTimer.singleShot(0, self._apply_correction)

    def _apply_correction(self):
        """Apply 2-point manual correction to current result and refresh."""
        if len(self._manual_points) < 2:
            return
        result = self._results[self._current_idx]
        params = self._params_getter() if callable(self._params_getter) else {}

        self._logic.apply_manual_correction(
            result, self._manual_points[0], self._manual_points[1], params
        )

        self._cache.pop(self._current_idx, None)
        self._manual_points = []
        self._full_pixmap = None
        self._pending_reset_zoom = False  # preserve zoom — user was zoomed in for precision
        self._view.clear_dots()           # remove placement dots before overlay rebuild

        self._manual_status.setText("Manual correction applied.")
        self._btn_revert_auto.setVisible(True)
        self._btn_manual_adjust.setText("✏ Redo Manual")
        self._metrics_label.setText(_format_metrics(result))

        self._start_job(self._current_idx)


def _format_metrics(r: dict) -> str:
    if r.get("error"):
        return f"<b>{r['filename']}</b>  ERROR: {r['error']}"
    parts = [f"<b>{r['filename']}</b>"]
    if r.get("length")    is not None: parts.append(f"Length: {r['length']:.1f} µm")
    if r.get("curvature") is not None: parts.append(f"Class: {r['curvature']}")
    if r.get("ratio")     is not None: parts.append(f"Ratio: {r['ratio']:.3f}")
    if r.get("eye_area")  is not None: parts.append(f"Eye area: {r['eye_area']:.1f} µm²")
    return "  |  ".join(parts)
