"""
Main widget for the ZebrafishEmbryoAnalyzer Slicer extension.

Left panel: input, analysis toggles, model selection, scalebar, run, export.
Right panel: QTabWidget with Gallery / Detail / Results tabs.
"""

import logging
import math

import qt
import ctk
import slicer

from ZebrafishEmbryoAnalyzerLib.errors import AnalysisInputError, MRMLAdapterError


# ---------------------------------------------------------------------------
# Model registry — (display_label, stable_id, (body_file, encoder, eye_file))
# ---------------------------------------------------------------------------

_MODEL_ENTRIES = [
    ("General Model",   "general", ("best_model_body_3400_vgg19.pth", "vgg19", None)),
    ("Fine-tuned DESY", "desy",    ("best_model_body_finetuned.pth",  "vgg19", "best_model_eye_finetuned.pth")),
]
_MODEL_BY_ID  = {mid: data for _, mid, data in _MODEL_ENTRIES}
_DEFAULT_MODEL_ID = "general"


# ---------------------------------------------------------------------------
# Parameter node schema — names and string-encoded defaults
# ---------------------------------------------------------------------------

PARAM_LENGTH_ENABLED               = "lengthEnabled"
PARAM_CURVATURE_ENABLED            = "curvatureEnabled"
PARAM_RATIO_ENABLED                = "ratioEnabled"
PARAM_EYES_ENABLED                 = "eyesEnabled"
PARAM_CONFIDENCE_THRESHOLD_ENABLED = "confidenceThresholdEnabled"
PARAM_CONFIDENCE_THRESHOLD         = "confidenceThreshold"
PARAM_UM_PER_PX                    = "micrometersPerPixel"
PARAM_MODEL_ID                     = "selectedModelId"

PARAM_DEFAULTS = {
    PARAM_LENGTH_ENABLED:               "true",
    PARAM_CURVATURE_ENABLED:            "true",
    PARAM_RATIO_ENABLED:                "true",
    PARAM_EYES_ENABLED:                 "false",
    PARAM_CONFIDENCE_THRESHOLD_ENABLED: "false",
    PARAM_CONFIDENCE_THRESHOLD:         "0.85",
    PARAM_UM_PER_PX:                    "22.99",
    PARAM_MODEL_ID:                     _DEFAULT_MODEL_ID,
}





class ZebrafishEmbryoAnalyzerMainWidget:
    def __init__(self, parent_layout, logic):
        self._logic = logic

        self._results = []
        self._excluded = set()
        self._image_paths = []
        self._current_detail_idx = 0
        self._updatingGUIFromParameterNode = False
        self._on_settings_changed = None  # callable set by ZebrafishEmbryoAnalyzerWidget
        self._active_downloader = None
        self._active_runner = None
        self._disposed = False
        self._run_token = 0
        self._deps_ok = True
        self._loading = False          # a load is in progress; the load buttons act as Cancel
        self._load_cancelled = False

        self._saved_layout_id = None
        self._saved_central_visible = None
        self._saved_pydock_floating = None
        self._saved_pydock_dock_area = None
        self._saved_dataprobe_collapsed = None

        self._build_ui(parent_layout)
        self._connect_signals()
        self._refresh_run_button()

    def _expand_panel(self):
        mw = slicer.util.mainWindow()
        central = mw.centralWidget()
        if central:
            central.setMinimumWidth(0)
            central.hide()
        panelDock = mw.findChild(qt.QDockWidget, "PanelDockWidget")
        if panelDock:
            panelDock.setMinimumHeight(300)
            mw.resizeDocks([panelDock], [mw.width], qt.Qt.Horizontal)
        _pyDock = mw.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
        if _pyDock:
            mw.resizeDocks([_pyDock], [150], qt.Qt.Vertical)
        dataProbe = mw.findChild(ctk.ctkCollapsibleButton, "DataProbeCollapsibleWidget")
        if dataProbe:
            dataProbe.collapsed = True

    def apply_shell_layout(self):
        """Save current Slicer shell state and apply module-specific layout.

        Called from ZebrafishEmbryoAnalyzerWidget.enter(). Must be paired with
        restore_shell_layout() in exit() so the host application is not
        permanently altered.
        """
        # Guard: prevent double-application and wrong state capture.
        # If _saved_layout_id is already set, layout was already applied — return.
        if self._saved_layout_id is not None:
            return
        try:
            mw = slicer.util.mainWindow()

            # -- Save current state ------------------------------------------------
            self._saved_layout_id = slicer.app.layoutManager().layout

            central = mw.centralWidget()
            self._saved_central_visible = central.isVisible() if central else True

            _pyDock = mw.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
            if _pyDock:
                self._saved_pydock_floating = _pyDock.isFloating()
                self._saved_pydock_dock_area = mw.dockWidgetArea(_pyDock)
            else:
                self._saved_pydock_floating = None
                self._saved_pydock_dock_area = None

            dataProbe = mw.findChild(ctk.ctkCollapsibleButton, "DataProbeCollapsibleWidget")
            self._saved_dataprobe_collapsed = dataProbe.collapsed if dataProbe else None

            # -- Apply module-specific shell layout --------------------------------
            slicer.app.layoutManager().setLayout(
                slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView
            )

            mw.setCorner(qt.Qt.BottomLeftCorner,  qt.Qt.BottomDockWidgetArea)
            mw.setCorner(qt.Qt.BottomRightCorner, qt.Qt.BottomDockWidgetArea)

            if _pyDock:
                _pyDock.setFloating(False)
                mw.addDockWidget(qt.Qt.BottomDockWidgetArea, _pyDock)
                _pyDock.setMinimumHeight(1)
                inner = _pyDock.widget()
                if inner:
                    for w in [inner] + list(inner.findChildren(qt.QWidget)):
                        w.setMinimumHeight(0)
                        sp = w.sizePolicy
                        sp.setVerticalPolicy(qt.QSizePolicy.Ignored)
                        w.setSizePolicy(sp)

            qt.QTimer.singleShot(0, self._expand_panel)
        except Exception:
            pass

    def restore_shell_layout(self):
        """Restore Slicer shell state saved in apply_shell_layout().

        Called from ZebrafishEmbryoAnalyzerWidget.exit(). Safe to call even if
        apply_shell_layout() was never called.
        """
        try:
            mw = slicer.util.mainWindow()

            # Restore layout
            if self._saved_layout_id is not None:
                slicer.app.layoutManager().setLayout(self._saved_layout_id)

            # Restore central widget visibility
            central = mw.centralWidget()
            if central:
                central.setMinimumWidth(0)
                if self._saved_central_visible:
                    central.show()

            # Restore corner settings to Slicer defaults
            mw.setCorner(qt.Qt.BottomLeftCorner,  qt.Qt.LeftDockWidgetArea)
            mw.setCorner(qt.Qt.BottomRightCorner, qt.Qt.RightDockWidgetArea)

            # Restore Python console dock position
            _pyDock = mw.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
            if _pyDock and self._saved_pydock_floating is not None:
                if self._saved_pydock_floating:
                    _pyDock.setFloating(True)
                elif self._saved_pydock_dock_area is not None:
                    mw.addDockWidget(self._saved_pydock_dock_area, _pyDock)

            # Restore DataProbe collapsed state
            if self._saved_dataprobe_collapsed is not None:
                dataProbe = mw.findChild(ctk.ctkCollapsibleButton, "DataProbeCollapsibleWidget")
                if dataProbe:
                    dataProbe.collapsed = self._saved_dataprobe_collapsed

            # Clear saved state so a double-call is a no-op
            self._saved_layout_id = None
            self._saved_central_visible = None
            self._saved_pydock_floating = None
            self._saved_pydock_dock_area = None
            self._saved_dataprobe_collapsed = None
        except Exception:
            pass

    def _build_ui(self, layout):
        layout.setAlignment(qt.Qt.Alignment())  # clear AlignTop set by Slicer base class

        splitter = qt.QSplitter(qt.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        splitter.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        layout.addWidget(splitter, 1)  # stretch=1 → fills all available vertical space

        self._build_left_panel(splitter)
        self._build_right_panel(splitter)
        splitter.setStretchFactor(1, 1)

        # progress bar removed — run button serves as progress indicator

    def _build_left_panel(self, splitter):
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(200)
        scroll.setMaximumWidth(500)
        splitter.addWidget(scroll)

        left = qt.QWidget()
        vbox = qt.QVBoxLayout(left)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(6)
        scroll.setWidget(left)

        input_box = ctk.ctkCollapsibleButton()
        input_box.text = "Input"
        vbox.addWidget(input_box)
        input_box.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        in_layout = qt.QVBoxLayout(input_box)

        self._btn_folder = qt.QPushButton("Load Folder…")
        self._btn_files  = qt.QPushButton("Load Images…")
        _load_row = qt.QHBoxLayout()
        _load_row.addWidget(self._btn_folder)
        _load_row.addWidget(self._btn_files)
        in_layout.addLayout(_load_row)
        in_layout.addWidget(qt.QLabel("Queue:"))
        self._queue_list = qt.QListWidget()
        self._queue_list.setMaximumHeight(120)
        in_layout.addWidget(self._queue_list)
        in_layout.addStretch()

        analysis_box = ctk.ctkCollapsibleButton()
        analysis_box.text = "Analysis"
        vbox.addWidget(analysis_box)
        analysis_box.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
        an_layout = qt.QVBoxLayout(analysis_box)

        self._chk_length    = qt.QCheckBox("Body length");        self._chk_length.setChecked(True)
        self._chk_curvature = qt.QCheckBox("Curvature class");    self._chk_curvature.setChecked(True)
        self._chk_ratio     = qt.QCheckBox("Length/straight ratio"); self._chk_ratio.setChecked(True)
        self._chk_eyes      = qt.QCheckBox("Eye segmentation");   self._chk_eyes.setChecked(False)
        self._chk_hitl      = qt.QCheckBox("Confidence threshold"); self._chk_hitl.setChecked(False)

        for chk in (self._chk_length, self._chk_curvature, self._chk_ratio,
                    self._chk_eyes, self._chk_hitl):
            an_layout.addWidget(chk)

        self._threshold_slider = ctk.ctkSliderWidget()
        self._threshold_slider.minimum    = 0.0
        self._threshold_slider.maximum    = 1.0
        self._threshold_slider.singleStep = 0.01
        self._threshold_slider.value      = 0.85
        self._threshold_slider.decimals   = 2
        an_layout.addWidget(self._threshold_slider)
        an_layout.addStretch()

        model_box = ctk.ctkCollapsibleButton()
        model_box.text      = "Model"
        model_box.collapsed = True
        vbox.addWidget(model_box)
        m_layout = qt.QFormLayout(model_box)

        self._model_combo = qt.QComboBox()
        for _label, _mid, _ in _MODEL_ENTRIES:
            self._model_combo.addItem(_label, _mid)
        m_layout.addRow("Segmentation model:", self._model_combo)

        scale_box = ctk.ctkCollapsibleButton()
        scale_box.text      = "Scale bar"
        scale_box.collapsed = False
        vbox.addWidget(scale_box)
        sc_layout = qt.QVBoxLayout(scale_box)

        self._btn_detect_scale = qt.QPushButton("Auto-detect from first image")
        sc_layout.addWidget(self._btn_detect_scale)

        self._scale_status = qt.QLabel("Load images first.")
        self._scale_status.setWordWrap(True)
        self._scale_status.setStyleSheet("color: #888; font-size: 11px;")
        sc_layout.addWidget(self._scale_status)

        form = qt.QFormLayout()
        self._bar_um_edit = qt.QLineEdit()
        self._bar_um_edit.setPlaceholderText("e.g. 500")
        form.addRow("Physical bar length (µm):", self._bar_um_edit)
        sc_layout.addLayout(form)

        self._btn_apply_scale = qt.QPushButton("Apply")
        sc_layout.addWidget(self._btn_apply_scale)

        sep = qt.QLabel("— or enter µm/px directly —")
        sep.setStyleSheet("color: #888; font-size: 11px;")
        sep.setAlignment(qt.Qt.AlignCenter)
        sc_layout.addWidget(sep)

        direct = qt.QFormLayout()
        self._um_per_px = ctk.ctkDoubleSpinBox()
        self._um_per_px.minimum    = 0.001
        self._um_per_px.maximum    = 9999.0
        self._um_per_px.singleStep = 0.01
        self._um_per_px.value      = 22.99
        self._um_per_px.decimals   = 4
        self._um_per_px.suffix     = " µm/px"
        direct.addRow("µm per pixel:", self._um_per_px)
        sc_layout.addLayout(direct)

        vbox.addStretch(1)  # push run + export to bottom

        # Non-modal notice about missing packages. Deliberately not a dialog: opening the
        # module must not interrupt, but the user has to learn about a pending install
        # before spending time loading images and setting parameters — otherwise the first
        # Run would end in a restart and throw that work away.
        self._deps_notice = qt.QWidget()
        _dn = qt.QVBoxLayout(self._deps_notice)
        _dn.setContentsMargins(0, 0, 0, 6)
        _dn.setSpacing(4)
        self._deps_notice_label = qt.QLabel()
        self._deps_notice_label.setWordWrap(True)
        _dn.addWidget(self._deps_notice_label)
        self._btn_install_deps = qt.QPushButton("Install Python packages…")
        _dn.addWidget(self._btn_install_deps)
        self._deps_notice.setVisible(False)
        vbox.addWidget(self._deps_notice)

        self._btn_run = qt.QPushButton("▶  Run Analysis")
        self._btn_run.setStyleSheet("font-weight: bold; padding: 6px;")

        self._run_progress = qt.QProgressBar()
        self._run_progress.setTextVisible(True)
        self._run_progress.setStyleSheet(
            "QProgressBar { text-align: center; border-radius: 3px; }"
            " QProgressBar::chunk { background: #2e7d32; border-radius: 2px; }"
        )

        self._run_stack = qt.QStackedWidget()
        self._run_stack.addWidget(self._btn_run)       # index 0 — idle

        # Build the running-state widget: status label on top, [progress bar | stop] below
        _run_widget = qt.QWidget()
        _run_vbox = qt.QVBoxLayout(_run_widget)
        _run_vbox.setContentsMargins(0, 0, 0, 0)
        _run_vbox.setSpacing(2)

        self._run_status_label = qt.QLabel("Loading models…")
        self._run_status_label.setStyleSheet("font-size: 11px;")
        self._run_status_label.setAlignment(qt.Qt.AlignCenter)
        _run_vbox.addWidget(self._run_status_label)

        _run_hbox = qt.QHBoxLayout()
        _run_hbox.setContentsMargins(0, 0, 0, 0)
        _run_hbox.setSpacing(4)
        _run_hbox.addWidget(self._run_progress)
        self._btn_stop = qt.QPushButton("✕ Stop")
        self._btn_stop.setFixedWidth(70)
        self._btn_stop.setToolTip("Stop download or analysis")
        _run_hbox.addWidget(self._btn_stop)
        _run_vbox.addLayout(_run_hbox)

        self._run_stack.addWidget(_run_widget)   # index 1 — running state

        vbox.addWidget(self._run_stack)

        export_box = ctk.ctkCollapsibleButton()
        export_box.text = "Export"
        vbox.addWidget(export_box)
        ex_layout = qt.QHBoxLayout(export_box)
        self._btn_excel = qt.QPushButton("Excel")
        self._btn_csv   = qt.QPushButton("CSV")
        ex_layout.addWidget(self._btn_excel)
        ex_layout.addWidget(self._btn_csv)


    def _build_right_panel(self, splitter):
        self._tabs = qt.QTabWidget()
        splitter.addWidget(self._tabs)

        from ZebrafishEmbryoAnalyzerLib.gallery_tab import GalleryTab
        self._gallery = GalleryTab(on_select=self._on_gallery_select)
        self._tabs.addTab(self._gallery, "Gallery")

        from ZebrafishEmbryoAnalyzerLib.detail_tab import DetailTab
        self._detail = DetailTab(
            on_navigate=self._navigate_detail,
            on_back=lambda: self._tabs.setCurrentIndex(0),
            logic=self._logic,
            on_exclude_change=self._on_exclude_change,
        )
        self._detail._params_getter = self._get_correction_params
        self._tabs.addTab(self._detail, "Detail")

        from ZebrafishEmbryoAnalyzerLib.results_tab import ResultsTab
        self._results_tab = ResultsTab(on_exclude_change=self._on_exclude_change)
        self._tabs.addTab(self._results_tab, "Results")

    def _connect_signals(self):
        self._btn_folder.clicked.connect(self._on_load_folder)
        self._btn_files.clicked.connect(self._on_load_files)
        self._btn_detect_scale.clicked.connect(self._on_detect_scale)
        self._btn_apply_scale.clicked.connect(self._on_apply_scale)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_install_deps.clicked.connect(self._on_install_deps_clicked)
        self._btn_stop.clicked.connect(self._cancel_workers)
        self._btn_excel.clicked.connect(self._on_export_excel)
        self._btn_csv.clicked.connect(self._on_export_csv)

        # Notify parameter node owner whenever any analysis setting changes.
        for _signal in (
            self._chk_length.toggled,
            self._chk_curvature.toggled,
            self._chk_ratio.toggled,
            self._chk_eyes.toggled,
            self._chk_hitl.toggled,
        ):
            _signal.connect(self._notify_settings_changed)
        self._threshold_slider.valueChanged.connect(self._notify_settings_changed)
        self._um_per_px.valueChanged.connect(self._notify_settings_changed)
        self._model_combo.currentIndexChanged.connect(self._notify_settings_changed)

    def _on_load_folder(self):
        if self._cancel_load_if_running():
            return
        if not self.ensure_dependencies("images"):
            return
        settings = qt.QSettings()
        last = str(settings.value("ZebrafishEmbryoAnalyzer/lastFolder", "")) or ""
        folder = qt.QFileDialog.getExistingDirectory(None, "Select image folder", last)
        if not folder:
            return
        folder = str(folder)
        settings.setValue("ZebrafishEmbryoAnalyzer/lastFolder", folder)
        import os
        exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts and not f.startswith(".")
        ])
        self._set_queue(paths, self._btn_folder)

    def _on_load_files(self):
        if self._cancel_load_if_running():
            return
        if not self.ensure_dependencies("images"):
            return
        paths = qt.QFileDialog.getOpenFileNames(
            None, "Select images", "",
            "Images (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], list):
            paths = paths[0]  # Slicer Qt binding wraps in extra tuple
        if paths:
            self._set_queue(sorted(paths), self._btn_files)

    def _set_queue(self, paths, button=None):
        self._run_token = getattr(self, "_run_token", 0) + 1
        # Cancel any in-flight inference so the subprocess doesn't produce stale results.
        if getattr(self, "_active_runner", None) is not None:
            self._active_runner.cancel()
            self._active_runner = None
            try:
                self._run_stack.setCurrentIndex(0)
            except Exception:
                pass
        import os
        self._image_paths = paths
        self._queue_list.clear()

        stubs = []
        for p in paths:
            self._queue_list.addItem(os.path.basename(p))
            stubs.append({"filename": os.path.basename(p), "original": None,
                          "mask": None, "error": None, "length": None})

        self._results = stubs
        self._excluded = set()
        self._detail.reset()
        self._results_tab.populate([], set())
        self._gallery.populate(stubs)
        self._tabs.setCurrentIndex(0)

        # Read image header only (no pixel data) to get dimensions for µm/px default.
        # PIL.Image.open is lazy — reads TIFF header in milliseconds.
        if paths:
            try:
                from PIL import Image as _PIL
                with _PIL.open(paths[0]) as _img:
                    w, h = _img.size
                self._um_per_px.value = round(5885.0 / h, 4)
            except Exception:
                pass

        self._load_originals(paths, stubs, button)
        self._refresh_run_button()

    def _load_originals(self, paths, stubs, button=None):
        """Load original images after an explicit user action.

        Runs on the main thread, so the event loop has to be given a turn between images —
        without it the whole application freezes until the last file is read, which on the
        very first load after installing the packages takes long enough to show the system's
        busy cursor. Processing events also lets the thumbnails appear one by one instead of
        all at once at the end.
        """
        import cv2
        from ZebrafishEmbryoAnalyzerLib.gallery_tab import THUMB_SIZE as _THUMB_SIZE

        if not paths:
            return

        # Progress is reported on the button that started the load, so the feedback appears
        # where the user just clicked, and the restored label is the signal that it is done —
        # thumbnails filling in one by one shows activity but not completion.
        #
        # That button stays enabled and doubles as Cancel: its handler sees _loading and
        # treats the click as an abort instead of starting a second load. The other button
        # is disabled so there is only one way to interrupt.
        buttons = [b for b in (self._btn_folder, self._btn_files) if b is not None]
        labels = [b.text for b in buttons]
        for b in buttons:
            b.setEnabled(b is button)

        self._loading = True
        self._load_cancelled = False
        try:
            for i, p in enumerate(paths):
                if stubs is not self._results or self._load_cancelled:
                    return
                img = cv2.imread(p)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    stubs[i]["original"] = rgb
                    h, w = rgb.shape[:2]
                    scale = _THUMB_SIZE / max(h, w)
                    thumb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
                    self._gallery.update_thumb_prebuilt(i, thumb)
                if button is not None:
                    button.setText(f"Cancel ({i + 1}/{len(paths)})")
                slicer.app.processEvents()
        finally:
            self._loading = False
            for b, text in zip(buttons, labels):
                b.setText(text)
                b.setEnabled(True)

    def _cancel_load_if_running(self) -> bool:
        """Turn a click on the loading button into an abort. True when it was handled.

        Cancelling stops the thumbnails from loading; the images stay queued, so an analysis
        can still be started — it reads the files itself and does not need the previews.
        """
        if not getattr(self, "_loading", False):
            return False
        self._load_cancelled = True
        return True

    def _required_model_entries(self, model_id):
        """Return the model entries required by the current settings."""
        from ZebrafishEmbryoAnalyzerLib.model_manifest import MODEL_SETS
        model_set = MODEL_SETS.get(model_id, MODEL_SETS[_DEFAULT_MODEL_ID])
        required = {"body": model_set["body"]}
        if self._chk_curvature.isChecked() and "curvature" in model_set:
            required["curvature"] = model_set["curvature"]
        if self._chk_eyes.isChecked() and "eye" in model_set:
            required["eye"] = model_set["eye"]
        return required

    def _missing_required_models(self, model_id):
        from ZebrafishEmbryoAnalyzerLib.model_manifest import get_missing_models
        return get_missing_models(self._required_model_entries(model_id))

    def _prompt_download_models(self, missing):
        """Ask the user whether to download the missing model entries."""
        try:
            import slicer as _slicer
            if _slicer.app.testingEnabled():
                return False
        except ImportError:
            return False

        def _mb_label(byte_count):
            return round(byte_count / 1_048_576)

        def _size_label(e):
            sb = e.get("size_bytes", 0)
            if sb > 0:
                return f"  • {e['label']} (~{_mb_label(sb)} MB)"
            return f"  • {e['label']}"

        total_bytes = sum(e.get("size_bytes", 0) for e in missing)
        lines = [_size_label(e) for e in missing]
        body = (
            "The following models need to be downloaded:\n\n"
            + "\n".join(lines)
        )
        if total_bytes > 0:
            body += f"\n\nTotal: ~{_mb_label(total_bytes)} MB"
        summary = "The required models are not downloaded yet."
        if total_bytes > 0:
            summary += f" About {_mb_label(total_bytes)} MB will be downloaded."
        summary += "\n\nDownload now? This is only needed once and requires internet access."

        if not slicer.util.confirmOkCancelDisplay(
            summary, "Download required models", detailedText=body
        ):
            return False

        return True

    def _on_detect_scale(self):
        if not self._image_paths:
            self._scale_status.setText("Load images first.")
            return
        if not self.ensure_dependencies("scalebar"):
            return
        result = self._logic.detect_scalebar(self._image_paths[0])
        if result.get("bar_found"):
            um_per_px = result.get("scale_um_per_px")
            bar_px = result.get("bar_length_px")
            if um_per_px is not None:
                self._um_per_px.value = um_per_px
                label_detected = result.get("label_um_detected")
                if label_detected is not None:
                    self._bar_um_edit.text = f"{label_detected:.0f}"
                self._scale_status.setText(f"Detected: {um_per_px:.4f} µm/px")
                self._scale_status.setStyleSheet("color: #4CAF50;")
            else:
                bar_info = f"  ({bar_px:.0f} px)" if bar_px is not None else ""
                self._scale_status.setText(
                    f"Bar found{bar_info}. Enter physical length (µm) + click Apply."
                )
                self._scale_status.setStyleSheet("color: #FFC107;")
        else:
            self._scale_status.setText(
                "No scalebar detected. Enter µm/px directly."
            )
            self._scale_status.setStyleSheet("color: #F44336;")

        if result.get("debug_img") is not None:
            self._detail.show_raw_image(
                result["debug_img"],
                result.get("message", ""),
            )
            self._tabs.setCurrentIndex(1)

    def _on_apply_scale(self):
        text = self._bar_um_edit.text.strip()
        if not text or not self._image_paths:
            return
        try:
            label_um = float(text)
        except ValueError:
            self._scale_status.setText("Invalid value — enter a number.")
            return
        result = self._logic.detect_scalebar(self._image_paths[0], label_um=label_um)
        if result.get("success"):
            self._um_per_px.value = result["scale_um_per_px"]
            self._scale_status.setText(
                f"Applied: {result['scale_um_per_px']:.4f} µm/px"
            )

    def _on_run(self):
        if not self.ensure_dependencies("analysis"):
            return

        with slicer.util.tryWithErrorDisplay("Failed to start the analysis.", waitCursor=True):
            self._run_token += 1
            token = self._run_token

            model_id = self._model_combo.currentData or _DEFAULT_MODEL_ID

            params = {
                "length":    self._chk_length.isChecked(),
                "curvature": self._chk_curvature.isChecked(),
                "ratio":     self._chk_ratio.isChecked(),
                "eyes":      self._chk_eyes.isChecked(),
                "hitl":      self._chk_hitl.isChecked(),
                "threshold": self._threshold_slider.value,
                "um_per_px": self._um_per_px.value,
                "model_id":  model_id,
            }

            missing = self._missing_required_models(model_id)
            if missing:
                if not self._prompt_download_models(missing):
                    return
                self._start_model_download(missing, model_id, params, token)
                return

            self._start_inference_process(model_id, params, token)

    def _start_model_download(self, missing, model_id, params, token):
        """Start an asynchronous Qt download and continue analysis only on success."""
        if self._active_downloader is not None:
            slicer.util.warningDisplay("A model download is already running.")
            return

        try:
            self._run_status_label.setText("Downloading models…")
            self._run_progress.setRange(0, 100)
            self._run_progress.setValue(0)
            self._run_progress.setFormat("")
            self._run_stack.setCurrentIndex(1)

            from ZebrafishEmbryoAnalyzerLib.model_downloader import start_model_download

            def _finished(success, state, message, controller):
                if self._disposed or controller is not self._active_downloader:
                    return
                self._active_downloader = None
                if not success:
                    self._run_stack.setCurrentIndex(0)
                    return
                if self._missing_required_models(model_id):
                    slicer.util.errorDisplay(
                        "Model download finished, but required models were not verified."
                    )
                    self._run_stack.setCurrentIndex(0)
                    return
                # Download succeeded. Set up "Loading models…" state now so it can
                # render during the one event-loop cycle we defer by below.
                self._run_status_label.setText("Loading models…")
                self._run_progress.setRange(0, 0)
                self._run_progress.setFormat("")
                # Keep _run_stack at index 1 (running view); do NOT reset to 0 here.
                # Schedule analysis for the next event-loop iteration. This gives
                # the download dialog one cycle to close and the "Loading models…"
                # state one cycle to render before the main thread blocks on
                # preload_models(). No polling, no processEvents(), exactly once.
                def _deferred_analysis():
                    if self._disposed:
                        self._run_stack.setCurrentIndex(0)
                        return
                    if self._run_token != token:
                        # A newer run superseded this one; clean up silently.
                        self._run_stack.setCurrentIndex(0)
                        return
                    self._start_inference_process(model_id, params, token)

                qt.QTimer.singleShot(0, _deferred_analysis)

            self._active_downloader = start_model_download(
                missing,
                _finished,
                parent=slicer.util.mainWindow(),
            )
        except Exception as exc:
            # Restore UI to idle state if download setup raised before connecting.
            self._run_stack.setCurrentIndex(0)
            self._active_downloader = None
            logging.exception("ZebrafishEmbryoAnalyzer: failed to start model download")
            slicer.util.errorDisplay(f"Could not start model download:\n{exc}")

    def _start_inference_process(self, model_id, params, token):
        """Launch analysis as a QProcess worker subprocess."""
        if self._active_runner is not None:
            slicer.util.warningDisplay("Analysis is already running.")
            return
        try:
            originals = [r.get("original") for r in self._results]
            image_paths = list(self._image_paths)

            self._run_status_label.setText("Loading models…")
            self._run_progress.setRange(0, 0)   # native Qt marquee: fixed chunk moves left→right
            self._run_progress.setFormat("")    # text shown in label above, not in bar
            self._run_stack.setCurrentIndex(1)

            def _on_progress(i, n):
                self._run_status_label.setText(f"Running analysis…  {i} / {n}")
                self._run_progress.setRange(0, n)
                self._run_progress.setValue(i)
                self._run_progress.setFormat("")

            def _on_runner_finished(success, state, message, controller):
                logging.debug("ZebrafishEmbryoAnalyzer: _on_runner_finished state=%s success=%s token=%s run_token=%s", state, success, token, self._run_token)
                if self._disposed or controller is not self._active_runner:
                    return
                self._active_runner = None
                if self._run_token != token:
                    self._run_stack.setCurrentIndex(0)
                    return
                if not success:
                    self._run_stack.setCurrentIndex(0)
                    if state not in ("cancelled", "disposed"):
                        ui_message = self._categorize_inference_error(message, controller)
                        slicer.util.errorDisplay(ui_message)
                    return
                self._results = controller.results
                self._run_stack.setCurrentIndex(0)
                logging.debug("ZebrafishEmbryoAnalyzer: results ready, count=%d", len(self._results))
                try:
                    self._on_results_ready()
                    self._try_update_mrml_table(self._results)
                except Exception:
                    logging.exception("ZebrafishEmbryoAnalyzer: exception in _on_results_ready")
                    raise

            from ZebrafishEmbryoAnalyzerLib.inference_runner import start_inference
            self._active_runner = start_inference(
                image_paths, model_id, params, originals,
                on_finished=_on_runner_finished,
                on_progress=_on_progress,
                parent=slicer.util.mainWindow(),
            )
        except Exception as exc:
            self._active_runner = None
            self._run_stack.setCurrentIndex(0)
            logging.exception("ZebrafishEmbryoAnalyzer: failed to start inference process")
            slicer.util.errorDisplay(f"Could not start analysis:\n{exc}")

    def _try_update_mrml_table(self, results):
        """Update the MRML results table; log and show a status warning on failure."""
        try:
            self._logic.update_results_table(results)
        except MRMLAdapterError as exc:
            logging.warning("ZebrafishEmbryoAnalyzer: MRML table update failed: %s", exc)
            slicer.util.showStatusMessage(
                "Analysis complete — results table update failed. Check the application log.",
                5000,
            )

    def _try_update_mrml_image(self, result):
        # NOTE: _on_detect_scale / show_raw_image does NOT call this method (E2b scope).
        # The MRML node intentionally reflects the last gallery selection, not the
        # scalebar debug overlay. Re-sync by clicking any gallery item.
        try:
            self._logic.update_current_image_node(result, self._um_per_px.value)
        except MRMLAdapterError as exc:
            logging.warning("ZebrafishEmbryoAnalyzer: MRML image node update failed: %s", exc)
            slicer.util.showStatusMessage(
                "Image node update failed. Check the application log.",
                5000,
            )

    def _try_update_mrml_segmentation(self, result):
        try:
            self._logic.update_current_segmentation_node(result, self._um_per_px.value)
        except MRMLAdapterError as exc:
            logging.warning("ZebrafishEmbryoAnalyzer: MRML segmentation node update failed: %s", exc)
            slicer.util.showStatusMessage(
                "Segmentation node update failed. Check the application log.",
                5000,
            )

    def _get_correction_params(self):
        """Return current hitl/threshold settings for manual correction curvature recompute."""
        return {
            "hitl": self._chk_hitl.isChecked(),
            "threshold": float(self._threshold_slider.value),
        }

    def _on_exclude_change(self, filename: str, checked: bool) -> None:
        if checked:
            self._excluded.add(filename)
        else:
            self._excluded.discard(filename)
        self._results_tab.sync_exclude(self._excluded)
        self._detail.sync_exclude(filename in self._excluded)

    def _on_gallery_select(self, index: int):
        self._current_detail_idx = index
        self._tabs.setCurrentIndex(1)
        self._detail.show_result(index, self._results)
        if index < len(self._results):
            self._detail.sync_exclude(self._results[index]["filename"] in self._excluded)
        self._detail.setFocus()
        if index < len(self._results):
            self._try_update_mrml_image(self._results[index])
            self._try_update_mrml_segmentation(self._results[index])

    def _navigate_detail(self, delta: int):
        if not self._results:
            return
        idx = max(0, min(len(self._results) - 1, self._current_detail_idx + delta))
        if idx != self._current_detail_idx:
            self._on_gallery_select(idx)

    def _notify_settings_changed(self, *args):
        """Forward any setting control change to the parameter node owner."""
        if not self._updatingGUIFromParameterNode and callable(self._on_settings_changed):
            self._on_settings_changed()

    def updateGUIFromParameterNode(self, node):
        """Read parameter values from node and apply to all setting controls."""
        if node is None:
            return
        self._updatingGUIFromParameterNode = True
        try:
            def _b(key, default_bool):
                v = node.GetParameter(key)
                if v == "true":
                    return True
                if v == "false":
                    return False
                return default_bool

            def _f_clamp(key, lo, hi, fallback):
                v = node.GetParameter(key)
                try:
                    f = float(v)
                    if math.isfinite(f) and lo <= f <= hi:
                        return f
                except (ValueError, TypeError):
                    pass
                return fallback

            def _s_model(key):
                v = node.GetParameter(key)
                return v if v in _MODEL_BY_ID else _DEFAULT_MODEL_ID

            self._chk_length.setChecked(_b(PARAM_LENGTH_ENABLED, True))
            self._chk_curvature.setChecked(_b(PARAM_CURVATURE_ENABLED, True))
            self._chk_ratio.setChecked(_b(PARAM_RATIO_ENABLED, True))
            self._chk_eyes.setChecked(_b(PARAM_EYES_ENABLED, False))
            self._chk_hitl.setChecked(_b(PARAM_CONFIDENCE_THRESHOLD_ENABLED, False))
            self._threshold_slider.value = _f_clamp(PARAM_CONFIDENCE_THRESHOLD, 0.0, 1.0, 0.85)
            self._um_per_px.value = _f_clamp(PARAM_UM_PER_PX, 0.001, 9999.0, 22.99)

            model_id = _s_model(PARAM_MODEL_ID)
            for i in range(self._model_combo.count):
                if self._model_combo.itemData(i) == model_id:
                    self._model_combo.setCurrentIndex(i)
                    break
        finally:
            self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, node):
        """Write all setting control values to the parameter node."""
        if node is None or self._updatingGUIFromParameterNode:
            return
        wasModified = node.StartModify()
        try:
            node.SetParameter(PARAM_LENGTH_ENABLED,
                              "true" if self._chk_length.isChecked() else "false")
            node.SetParameter(PARAM_CURVATURE_ENABLED,
                              "true" if self._chk_curvature.isChecked() else "false")
            node.SetParameter(PARAM_RATIO_ENABLED,
                              "true" if self._chk_ratio.isChecked() else "false")
            node.SetParameter(PARAM_EYES_ENABLED,
                              "true" if self._chk_eyes.isChecked() else "false")
            node.SetParameter(PARAM_CONFIDENCE_THRESHOLD_ENABLED,
                              "true" if self._chk_hitl.isChecked() else "false")
            node.SetParameter(PARAM_CONFIDENCE_THRESHOLD,
                              str(round(float(self._threshold_slider.value), 4)))
            node.SetParameter(PARAM_UM_PER_PX,
                              str(round(float(self._um_per_px.value), 6)))
            model_id = self._model_combo.currentData or _DEFAULT_MODEL_ID
            node.SetParameter(PARAM_MODEL_ID,
                              model_id if model_id in _MODEL_BY_ID else _DEFAULT_MODEL_ID)
        finally:
            node.EndModify(wasModified)

    def _refresh_run_button(self):
        """Enable Run as soon as images are queued.

        Missing packages deliberately do not disable the button: they are installed when
        the analysis starts, so blocking it here would leave no way to trigger that.
        """
        enabled = len(self._image_paths) > 0
        self._btn_run.setEnabled(enabled)
        if not enabled:
            self._btn_run.setToolTip("Load images before running analysis.")
        elif not self._deps_ok:
            self._btn_run.setToolTip(
                "Missing Python packages will be installed when the analysis starts."
            )
        else:
            self._btn_run.setToolTip("")

    def refresh_dependency_status(self):
        ml = self._logic.dependency_status()  # {"torch": bool, "cv2": bool, ...}
        missing_ml = [k for k, v in ml.items() if not v]

        self._deps_ok = not bool(missing_ml)
        self._refresh_run_button()
        self._refresh_dependency_notice()

    def _refresh_dependency_notice(self):
        """Show or hide the in-panel notice about packages that still need installing."""
        notice = getattr(self, "_deps_notice", None)
        if notice is None:
            return

        from ZebrafishEmbryoAnalyzerLib import dependency_installer
        missing = dependency_installer.get_missing_packages("analysis")
        count = len(missing["torch"]) + len(missing["general"])
        if not count:
            notice.setVisible(False)
            return

        self._deps_notice_label.setText(
            f"{count} Python package(s) still need to be installed before an analysis "
            "can run. This needs a network connection and takes a few minutes."
        )
        notice.setVisible(True)

    def _on_install_deps_clicked(self):
        self.ensure_dependencies("analysis")

    def _categorize_inference_error(self, message, controller):
        """Return a user-facing error string based on exit_code; suppress raw tracebacks."""
        exit_code = getattr(controller, "exit_code", None)
        msg = message or ""

        # Raw Python traceback: log full text, show generic UI message.
        if "Traceback" in msg:
            logging.warning("ZebrafishEmbryoAnalyzer: inference traceback:\n%s", msg)
            return "Analysis failed. Check the application log for details."

        if exit_code == 1:
            first_line = msg.split("\n")[0].strip() if msg else ""
            return f"Analysis failed: {first_line}" if first_line else "Analysis failed."
        if exit_code == 2:
            return "Required models are not loaded. Run the analysis again to trigger a download."
        if exit_code == 3:
            return "Internal error: bad analysis request. Check the application log."
        if exit_code == 4:
            return "Internal error: could not write temporary results. Check disk space."
        return "Analysis failed. Check the application log."

    def ensure_dependencies(self, purpose="analysis"):
        """Make sure the Python packages needed for ``purpose`` are installed.

        Called at the start of every action that needs them — never when the module is
        merely opened, so browsing existing results never raises an installation question.

        Returns True when the caller may proceed — including right after a successful
        install, since a freshly installed package that was never imported in this session
        is usable immediately. Returns False when the user declines, when the install
        fails, and when a restart is genuinely required because something already held in
        memory was replaced.
        """
        try:
            import slicer
            if slicer.app.testingEnabled():
                return True
        except ImportError:
            return True

        from ZebrafishEmbryoAnalyzerLib import dependency_installer
        missing = dependency_installer.get_missing_packages(purpose)
        if not any(missing.values()):
            return True

        items = []
        if missing["torch"]:
            items.append("torch + torchvision (via the PyTorch extension)")
        items.extend(missing["general"])

        detail = "Packages that will be installed:\n" + "\n".join(f"  • {i}" for i in items)
        if missing["torch"]:
            # PyTorchUtils only becomes importable after a restart, so torch itself
            # can only be installed on the following run.
            detail += (
                "\n\nIf the PyTorch extension is not installed yet, it is installed first "
                "and a second restart is required before PyTorch itself follows."
            )

        if not slicer.util.confirmOkCancelDisplay(
            "This module requires additional Python packages. Installation needs a network "
            "connection, takes several minutes, and Slicer has to be restarted afterwards.",
            "Confirm Python package installation",
            detailedText=detail,
        ):
            return False

        import qt
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            # install_packages reports the failures it knows about and returns "failed";
            # anything reaching here is an unexpected defect worth surfacing.
            outcome = dependency_installer.install_packages(missing)
        except Exception as exc:
            import logging
            logging.exception("Dependency install failed: %s", exc)
            slicer.util.errorDisplay(f"Installation failed: {exc}")
            return False
        finally:
            slicer.app.restoreOverrideCursor()

        self.refresh_dependency_status()

        if outcome == "ready":
            return True
        if outcome == "restart":
            self._show_restart_dialog()
        return False

    def _show_restart_dialog(self):
        """Ask to restart after a package installation, the way Slicer does it elsewhere."""
        if slicer.util.confirmOkCancelDisplay(
            "Application restart is required to complete the installation of the required "
            "Python packages.\nPress OK to restart.",
            "Confirm application restart",
        ):
            slicer.util.restart()
            return

        self.refresh_dependency_status()

    def _cancel_workers(self):
        """Cancel active asynchronous operations and invalidate transient state."""
        self._run_token = getattr(self, "_run_token", 0) + 1  # invalidate any pending deferred continuation
        if getattr(self, "_active_downloader", None) is not None:
            self._active_downloader.cancel(silent=True)
            self._active_downloader = None
        if getattr(self, "_active_runner", None) is not None:
            self._active_runner.cancel()
            self._active_runner = None
        if hasattr(self, "_run_stack"):
            self._run_stack.setCurrentIndex(0)
        self._results = []
        self._detail.invalidate_cache()

    def reset_for_scene_close(self):
        """Clear all session state after the MRML scene is closed."""
        self._cancel_workers()
        self._results = []
        self._image_paths = []
        self._excluded = set()
        self._detail.reset()
        self._queue_list.clear()
        self._gallery.populate([])
        self._results_tab.populate([], set())
        self._run_stack.setCurrentIndex(0)
        # Scale-bar temporary state. _um_per_px is intentionally not reset here;
        # the caller syncs it from the new parameter node immediately after.
        self._scale_status.setText("Load images first.")
        self._scale_status.setStyleSheet("color: #888; font-size: 11px;")
        self._bar_um_edit.setText("")

    def cleanup(self):
        """Stop persistent resources before the widget is torn down."""
        self._disposed = True
        if getattr(self, "_active_downloader", None) is not None:
            self._active_downloader.dispose()
            self._active_downloader = None
        if getattr(self, "_active_runner", None) is not None:
            self._active_runner.dispose()
            self._active_runner = None
        self._results = []
        self._detail.cleanup()  # invalidates cache

    def _on_results_ready(self):
        self._detail.invalidate_cache()
        self._gallery.populate(self._results)
        # Auto-exclude error rows so the visual state and export filter are consistent.
        self._excluded = {r["filename"] for r in self._results if r.get("error")}
        self._results_tab.populate(self._results, self._excluded)
        self._tabs.setCurrentIndex(0)
        errors = [r for r in self._results if r.get("error")]
        if errors:
            # Short summary in the message, full text including tracebacks behind the
            # dialog's "Details" — readable without opening the Python console, which the
            # previous version required.
            names = "\n".join(f"• {r['filename']}" for r in errors[:10])
            if len(errors) > 10:
                names += f"\n… and {len(errors) - 10} more"
            detail = "\n\n".join(f"{r['filename']}:\n{r['error']}" for r in errors)
            slicer.util.warningDisplay(
                f"{len(errors)} of {len(self._results)} image(s) could not be analysed:\n\n{names}",
                detailedText=detail,
            )

    def _on_export_excel(self):
        from ZebrafishEmbryoAnalyzerLib.export import export_excel
        if not self._results:
            slicer.util.warningDisplay("No results to export. Run analysis first.")
            return
        if not self.ensure_dependencies("excel"):
            return
        path = qt.QFileDialog.getSaveFileName(None, "Save Excel", "", "Excel (*.xlsx)")
        if path:
            if not path.endswith(".xlsx"):
                path += ".xlsx"
            active = [r for r in self._results if r["filename"] not in self._excluded]
            try:
                export_excel(active, path)
                slicer.util.infoDisplay(f"Saved {len(active)} rows to:\n{path}")
            except Exception as e:
                slicer.util.errorDisplay(f"Export failed:\n{e}")

    def _on_export_csv(self):
        from ZebrafishEmbryoAnalyzerLib.export import export_csv
        if not self._results:
            slicer.util.warningDisplay("No results to export. Run analysis first.")
            return
        path = qt.QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV (*.csv)")
        if path:
            if not path.endswith(".csv"):
                path += ".csv"
            active = [r for r in self._results if r["filename"] not in self._excluded]
            try:
                export_csv(active, path)
                slicer.util.infoDisplay(f"Saved {len(active)} rows to:\n{path}")
            except Exception as e:
                slicer.util.errorDisplay(f"Export failed:\n{e}")
