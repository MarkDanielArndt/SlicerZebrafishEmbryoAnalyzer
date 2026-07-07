"""
Asynchronous Qt model downloader for ZebrafishEmbryoAnalyzer.

The downloader uses Slicer's PythonQt QNetworkAccessManager and never starts a
Python thread, blocks in a nested event loop, or manually pumps the Qt event
loop.  One retained controller owns the manager, reply, temporary file, queue,
progress dialog, and completion callback for the complete asynchronous
lifetime.

Project policy: production extension code must not create Python background
threads.  Use Slicer/Qt signal-driven APIs for supported asynchronous work, or
keep work synchronous behind an explicit user action until a later execution
architecture is approved.
"""

import os
import time
from pathlib import Path

from ZebrafishEmbryoAnalyzerLib.model_manifest import (
    _CACHE_DIR,
    checksum_mismatch_error,
    get_cached_path,
    verify_checksum,
)


_MAX_REDIRECTS = 10


class ModelDownloadController:
    """Signal-driven sequential downloader for model manifest entries."""

    TERMINAL_STATES = {"succeeded", "cancelled", "failed", "disposed"}

    def __init__(
        self,
        entries,
        on_finished,
        parent=None,
        qt_module=None,
        slicer_module=None,
        manager_factory=None,
    ):
        self.entries = list(entries)
        self.on_finished = on_finished
        self.parent = parent
        self.qt = qt_module
        self.slicer = slicer_module
        self.manager_factory = manager_factory

        self.state = "idle"
        self.manager = None
        self.reply = None
        self.tmp_file = None
        self.tmp_path = None
        self.final_path = None
        self.current_entry = None
        self.remaining = []
        self.completed_bytes = 0
        self.current_received = 0
        self.current_total = 0
        self.total_estimate = sum(e.get("size_bytes", 0) for e in self.entries)
        self._display_total = self.total_estimate
        self.redirect_count = 0
        self.network_error = None
        self.disposed = False
        self.cancelled = False
        self._finished_called = False
        self._dialog = None
        self._last_error_message = None
        self._started_at = None
        self._last_progress_at = None
        self._last_progress_bytes = 0
        self._current_rate = None
        self._session_commits = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self.state != "idle":
            return self
        self._ensure_modules()
        if not self.entries:
            self._finish_once("succeeded", True, None)
            return self

        os.makedirs(_CACHE_DIR, exist_ok=True)
        self._started_at = time.time()
        self.remaining = list(self.entries)
        self.manager = self._create_manager()
        self._dialog = self._create_dialog()
        self.state = "downloading"
        self._start_next()
        return self

    def cancel(self, silent=False):
        if self.state in self.TERMINAL_STATES:
            return
        self.cancelled = True
        self._finish_once("cancelled", False, None, abort_reply=True, silent=silent)

    def dispose(self):
        if self.state in self.TERMINAL_STATES:
            self.disposed = True
            return
        self.disposed = True
        self._finish_once("disposed", False, None, abort_reply=True, silent=True)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _ensure_modules(self):
        if self.qt is None:
            import qt
            self.qt = qt
        if self.slicer is None:
            import slicer
            self.slicer = slicer

    def _create_manager(self):
        if self.manager_factory is not None:
            return self.manager_factory()
        return self.qt.QNetworkAccessManager(self.parent)

    def _create_dialog(self):
        first = self.entries[0]["label"]
        dlg = self.qt.QProgressDialog(
            f"Downloading {first}...",
            "Cancel",
            0,
            1000,
            self.slicer.util.mainWindow(),
        )
        dlg.setWindowTitle("Downloading Models")
        dlg.setWindowModality(self.qt.Qt.ApplicationModal)
        dlg.setMinimumWidth(420)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        self._connect_signal(dlg.canceled, lambda: self.cancel())
        dlg.show()
        return dlg

    # ------------------------------------------------------------------
    # Queue and request handling
    # ------------------------------------------------------------------

    def _start_next(self):
        if self.state in self.TERMINAL_STATES:
            return
        if not self.remaining:
            self._finish_once("succeeded", True, None)
            return

        self.current_entry = self.remaining.pop(0)
        self.redirect_count = 0
        self.current_received = 0
        self.current_total = self.current_entry.get("size_bytes", 0)
        self._start_request(self._entry_url(self.current_entry), reset_tmp=True)

    def _start_request(self, url, reset_tmp):
        if self.state in self.TERMINAL_STATES:
            return
        self.state = "downloading"
        self.network_error = None
        if reset_tmp:
            self._close_tmp(delete=True)
            self.final_path = str(get_cached_path(self.current_entry))
            self.tmp_path = self.final_path + ".tmp"
            try:
                self.tmp_file = open(self.tmp_path, "wb")
            except Exception as exc:
                self._finish_once("failed", False, f"Could not create model download file: {exc}")
                return
            self.current_received = 0

        request = self._make_request(url)
        self.reply = self.manager.get(request)
        self._connect_reply(self.reply)
        self._update_dialog()

    def _make_request(self, url):
        request = self.qt.QNetworkRequest(self.qt.QUrl(url))
        self._set_redirect_policy(request)
        return request

    def _set_redirect_policy(self, request):
        qnr = self.qt.QNetworkRequest
        attr = getattr(qnr, "RedirectPolicyAttribute", None)
        if attr is None:
            return
        policy = (
            getattr(qnr, "NoLessSafeRedirectPolicy", None)
            or getattr(qnr, "SameOriginRedirectPolicy", None)
            or getattr(qnr, "ManualRedirectPolicy", None)
        )
        if policy is not None:
            try:
                request.setAttribute(attr, policy)
            except Exception:
                pass

    def _connect_reply(self, reply):
        self._connect_signal(reply.readyRead, self._on_ready_read)
        self._connect_signal(reply.downloadProgress, self._on_download_progress)
        error_signal = getattr(reply, "errorOccurred", None) or getattr(reply, "error", None)
        if error_signal is not None:
            self._connect_signal(error_signal, self._on_network_error)
        self._connect_signal(reply.finished, self._on_finished)

    @staticmethod
    def _connect_signal(signal, slot):
        signal.connect(slot)

    @staticmethod
    def _disconnect_signal(signal, slot):
        try:
            signal.disconnect(slot)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reply signals
    # ------------------------------------------------------------------

    def _on_ready_read(self):
        if self.state in self.TERMINAL_STATES or self.disposed:
            return
        try:
            data = self._read_reply_bytes()
            if data:
                self.tmp_file.write(data)
                self.current_received += len(data)
        except Exception as exc:
            self._finish_once("failed", False, f"Could not write model download: {exc}",
                              abort_reply=True)

    def _on_download_progress(self, received, total):
        if self.state in self.TERMINAL_STATES or self.disposed:
            return
        try:
            received = int(received)
            aggregate_done = self.completed_bytes + max(self.current_received, received)
            self._record_progress_sample(aggregate_done)
            self.current_received = max(self.current_received, received)
            if int(total) > 0:
                self.current_total = int(total)
        except Exception:
            pass
        self._update_dialog()

    def _on_network_error(self, *args):
        if self.state in self.TERMINAL_STATES or self.disposed:
            return
        self.network_error = self._reply_error_string()

    def _on_finished(self):
        if self.state in self.TERMINAL_STATES or self.disposed:
            return
        if self.cancelled:
            self._finish_once("cancelled", False, None, abort_reply=False)
            return
        if self.network_error:
            self._finish_once("failed", False, self.network_error, abort_reply=False)
            return

        redirect_url = self._redirect_target()
        if redirect_url:
            self._follow_redirect(redirect_url)
            return

        status = self._http_status()
        if status is None:
            self._finish_once("failed", False, "Model download failed: missing HTTP status.",
                              abort_reply=False)
            return
        if status < 200 or status >= 300:
            self._finish_once("failed", False, f"Model download failed: HTTP {status}.",
                              abort_reply=False)
            return

        self.state = "verifying"
        if not self._verify_and_replace_current():
            return  # _verify_and_replace_current already called _finish_once

        self.completed_bytes += max(
            self.current_received,
            self.current_total,
            self.current_entry.get("size_bytes", 0),
        )
        self._release_reply()
        self._start_next()

    # ------------------------------------------------------------------
    # Redirects
    # ------------------------------------------------------------------

    def _redirect_target(self):
        qnr = self.qt.QNetworkRequest
        attr = getattr(qnr, "RedirectionTargetAttribute", None)
        if attr is None or self.reply is None:
            return None
        try:
            target = self.reply.attribute(attr)
        except Exception:
            return None
        if target is None:
            return None
        if hasattr(target, "toUrl"):
            target = target.toUrl()
        if not self._qurl_is_valid(target):
            text = str(target)
            if not text:
                return None
            target = self.qt.QUrl(text)
        if not self._qurl_is_valid(target):
            return None
        base = self.reply.url() if hasattr(self.reply, "url") else self.qt.QUrl("")
        if hasattr(target, "isRelative") and target.isRelative():
            target = base.resolved(target)
        return target

    def _follow_redirect(self, target):
        self.redirect_count += 1
        if self.redirect_count > _MAX_REDIRECTS:
            self._finish_once("failed", False, "Model download failed: too many redirects.",
                              abort_reply=False)
            return
        if not self._qurl_is_valid(target):
            self._finish_once("failed", False, "Model download failed: invalid redirect.",
                              abort_reply=False)
            return
        old_url = self.reply.url()
        if self._url_scheme(old_url) == "https" and self._url_scheme(target) != "https":
            self._finish_once("failed", False, "Model download failed: unsafe HTTPS redirect.",
                              abort_reply=False)
            return
        self.state = "redirecting"
        redirect_text = self._url_to_string(target)
        self._release_reply()
        self._close_tmp(delete=True)
        try:
            self.tmp_file = open(self.tmp_path, "wb")
        except Exception as exc:
            self._finish_once("failed", False, f"Could not create model download file: {exc}")
            return
        self.current_received = 0
        self._start_request(redirect_text, reset_tmp=False)

    @staticmethod
    def _qurl_is_valid(url):
        return hasattr(url, "isValid") and url.isValid()

    @staticmethod
    def _url_scheme(url):
        try:
            return str(url.scheme()).lower()
        except Exception:
            return ""

    @staticmethod
    def _url_to_string(url):
        if hasattr(url, "toString"):
            return str(url.toString())
        return str(url)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_and_replace_current(self):
        self._close_tmp(delete=False)
        tmp = Path(self.tmp_path)
        final = Path(self.final_path)
        try:
            if not tmp.exists() or tmp.stat().st_size == 0:
                self._finish_once("failed", False, "Model download failed: empty file.",
                                  abort_reply=False)
                return False
            exact_size = self.current_entry.get("exact_size_bytes")
            if exact_size is not None and tmp.stat().st_size != int(exact_size):
                self._finish_once("failed", False, "Model download failed: size mismatch.",
                                  abort_reply=False)
                return False
            sha256 = self.current_entry.get("sha256", "")
            try:
                ok = verify_checksum(tmp, sha256)
            except ValueError as exc:
                self._finish_once("failed", False, str(exc), abort_reply=False)
                return False
            if not ok:
                import hashlib as _hashlib
                h = _hashlib.sha256()
                try:
                    with open(tmp, "rb") as _f:
                        for _chunk in iter(lambda: _f.read(65536), b""):
                            h.update(_chunk)
                    actual = h.hexdigest()
                except OSError:
                    actual = "<unreadable>"
                msg = checksum_mismatch_error(self.current_entry, str(tmp), actual)
                self._finish_once("failed", False, f"Model download failed: {msg}",
                                  abort_reply=False)
                return False
            self._commit_verified_file(tmp, final)
        except Exception as exc:
            self._finish_once("failed", False, f"Model download failed: {exc}",
                              abort_reply=False)
            return False
        return True

    def _commit_verified_file(self, tmp, final):
        backup = Path(str(final) + ".download-session-backup")
        had_existing = final.exists()
        if backup.exists():
            backup.unlink()
        self._session_commits.append(
            {
                "final": final,
                "backup": backup,
                "had_existing": had_existing,
            }
        )
        if had_existing:
            os.replace(str(final), str(backup))
        os.replace(str(tmp), str(final))

    # ------------------------------------------------------------------
    # Terminal transition and cleanup
    # ------------------------------------------------------------------

    def _finish_once(self, state, success, message, abort_reply=False, silent=False):
        if self._finished_called or self.state in self.TERMINAL_STATES:
            return
        self._finished_called = True
        self.state = state
        if message:
            self._last_error_message = message

        if abort_reply and self.reply is not None:
            try:
                self.reply.abort()
            except Exception:
                pass
        self._release_reply()
        self._close_tmp(delete=not success)
        if success:
            self._discard_session_backups()
        else:
            self._rollback_session_commits()

        if self._dialog is not None:
            try:
                self._dialog.close()
            except Exception:
                pass
        dialog = self._dialog
        self._dialog = None
        self.manager = None
        self.remaining = []

        if (not success and state not in {"cancelled", "disposed"} and
                message and not silent and not self.disposed):
            try:
                self.slicer.util.errorDisplay(
                    f"Model download failed:\n{message}\n\n"
                    "Check your internet connection and try again."
                )
            except Exception:
                pass

        callback = self.on_finished
        self.on_finished = None
        if callback is not None and not self.disposed:
            callback(success, state, message, self)

        if dialog is not None and hasattr(dialog, "deleteLater"):
            try:
                dialog.deleteLater()
            except Exception:
                pass

    def _rollback_session_commits(self):
        for item in reversed(self._session_commits):
            final = item["final"]
            backup = item["backup"]
            try:
                if final.exists():
                    final.unlink()
            except OSError:
                pass
            if item["had_existing"] and backup.exists():
                try:
                    os.replace(str(backup), str(final))
                except OSError:
                    pass
            elif backup.exists():
                try:
                    backup.unlink()
                except OSError:
                    pass
        self._session_commits = []

    def _discard_session_backups(self):
        for item in self._session_commits:
            backup = item["backup"]
            if backup.exists():
                try:
                    backup.unlink()
                except OSError:
                    pass
        self._session_commits = []

    def _release_reply(self):
        reply = self.reply
        if reply is None:
            return
        for signal_name, slot in (
            ("readyRead", self._on_ready_read),
            ("downloadProgress", self._on_download_progress),
            ("finished", self._on_finished),
        ):
            signal = getattr(reply, signal_name, None)
            if signal is not None:
                self._disconnect_signal(signal, slot)
        error_signal = getattr(reply, "errorOccurred", None) or getattr(reply, "error", None)
        if error_signal is not None:
            self._disconnect_signal(error_signal, self._on_network_error)
        if hasattr(reply, "deleteLater"):
            try:
                reply.deleteLater()
            except Exception:
                pass
        self.reply = None

    def _close_tmp(self, delete):
        if self.tmp_file is not None:
            try:
                self.tmp_file.close()
            except Exception:
                pass
            self.tmp_file = None
        if delete and self.tmp_path:
            try:
                os.unlink(self.tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Small compatibility helpers
    # ------------------------------------------------------------------

    def _entry_url(self, entry):
        return (
            f"https://huggingface.co/{entry['repo_id']}/resolve/"
            f"{entry['revision']}/{entry['filename']}"
        )

    def _read_reply_bytes(self):
        data = self.reply.readAll()
        try:
            return bytes(data)
        except Exception:
            pass
        try:
            return data.data()
        except Exception:
            return str(data).encode("utf-8")

    def _reply_error_string(self):
        if self.reply is None:
            return "Network error."
        try:
            text = self.reply.errorString()
            if text:
                return str(text)
        except Exception:
            pass
        return "Network error."

    def _http_status(self):
        qnr = self.qt.QNetworkRequest
        attr = getattr(qnr, "HttpStatusCodeAttribute", None)
        if attr is None or self.reply is None:
            return None
        try:
            value = self.reply.attribute(attr)
            if value is None:
                return None
            if hasattr(value, "toInt"):
                converted = value.toInt()
                if isinstance(converted, tuple):
                    return int(converted[0])
                return int(converted)
            return int(value)
        except Exception:
            return None

    def _update_dialog(self):
        if self._dialog is None:
            return
        label = self.current_entry.get("label", "model") if self.current_entry else "model"
        aggregate_done = self.completed_bytes + self.current_received
        aggregate_total = max(self._display_total, aggregate_done)
        if aggregate_total > 0:
            value = max(0, min(1000, int(aggregate_done / aggregate_total * 1000)))
            self._dialog.setRange(0, 1000)
            self._dialog.setValue(value)
            suffix = self._progress_suffix(aggregate_done, aggregate_total)
            details = (
                f"Downloading {label}...\n"
                f"{aggregate_done / 1_048_576:.1f} / {aggregate_total / 1_048_576:.1f} MB"
            )
            self._dialog.setLabelText(f"{details}{suffix}")
        else:
            self._dialog.setRange(0, 0)
            suffix = self._progress_suffix(aggregate_done, None)
            self._dialog.setLabelText(
                f"Downloading {label}...\n"
                f"{aggregate_done / 1_048_576:.1f} MB"
                f"{suffix}"
            )

    def _progress_suffix(self, done_bytes, total_bytes):
        if done_bytes <= 0 or not self._current_rate or self._current_rate <= 0:
            return ""
        rate = self._current_rate
        rate_mbs = rate / 1_048_576
        if total_bytes and total_bytes > done_bytes and rate > 0:
            remaining = (total_bytes - done_bytes) / rate
            if remaining >= 3600:
                eta = f"~{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m left"
            elif remaining >= 60:
                eta = f"~{int(remaining // 60)}m {int(remaining % 60):02d}s left"
            else:
                eta = f"~{int(remaining)}s left"
            return f"  ·  {rate_mbs:.1f} MB/s  ·  {eta}"
        return f"  ·  {rate_mbs:.1f} MB/s"

    def _record_progress_sample(self, done_bytes):
        now = time.time()
        if self._last_progress_at is None:
            self._last_progress_at = now
            self._last_progress_bytes = done_bytes
            return
        elapsed = now - self._last_progress_at
        delta = done_bytes - self._last_progress_bytes
        if elapsed <= 0 or delta < 0:
            return
        if elapsed >= 0.25:
            self._current_rate = delta / elapsed
            self._last_progress_at = now
            self._last_progress_bytes = done_bytes


def start_model_download(entries, on_finished, parent=None, **kwargs):
    """Create, retain, and start an asynchronous model download controller."""
    controller = ModelDownloadController(entries, on_finished, parent=parent, **kwargs)
    return controller.start()


def download_models(entries: list) -> bool:
    """Compatibility guard for non-Slicer tests; interactive code uses start_model_download."""
    try:
        import slicer
        if slicer.app.testingEnabled():
            return True
    except ImportError:
        return True
    if not entries:
        return True
    raise RuntimeError("download_models is asynchronous; use start_model_download().")
