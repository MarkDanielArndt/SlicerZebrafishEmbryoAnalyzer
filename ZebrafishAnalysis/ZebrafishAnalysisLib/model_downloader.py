"""
Model downloader for ZebrafishAnalysis.

Slicer-aware: imports qt and slicer for dialog management.
Download I/O runs in a background thread using requests (no torch).
Atomic replacement (tmp file + os.replace) and cancellation are preserved
from the original widget.py implementation.

A single QProgressDialog covers all models in one download session.
Label format: "Downloading {model}…\\n{done_mb:.1f} / {total_mb:.1f} MB"
"""

import os
import threading
import time as _time

from ZebrafishAnalysisLib.model_manifest import _CACHE_DIR


def _run_downloads(models_with_urls, hf_headers, progress_state, cancel_event):
    """
    Stream-download each (url, filename, label, est_bytes) to the model cache.

    Runs in a background thread.  Uses requests only — no torch.
    Atomic replacement: writes to <filename>.tmp then os.replace to final path.

    progress_state keys written by this function:
        downloaded_bytes  -- cumulative bytes across all finished + current model
        total_bytes       -- running estimate of grand total (updated from Content-Length)
        current_label     -- human label of the model currently being downloaded
        done_flag         -- set True when all downloads complete successfully
        cancelled         -- set True on cancellation
        error             -- error string on failure
    """
    # Pre-initialize huggingface_hub if available.  segmentation_models_pytorch
    # imports it at module level; initialising here (background thread, no
    # concurrency risk) ensures sys.modules["huggingface_hub"] is populated
    # before analyse_images runs on the main thread.
    try:
        import huggingface_hub as _hf  # noqa: F401
    except Exception:
        pass

    import requests

    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp_path = None

    bytes_done_before = 0  # cumulative bytes from models already finished

    try:
        for i, (url, filename, label, est_bytes) in enumerate(models_with_urls):
            if cancel_event.is_set():
                progress_state["cancelled"] = True
                return

            progress_state["current_label"] = label

            # Bytes still to download from models not yet started (after this one).
            remaining_est = sum(m[3] for m in models_with_urls[i + 1 :])

            local_path = str(_CACHE_DIR / filename)
            tmp_path = local_path + ".tmp"

            # HEAD request: try to learn actual size before streaming.
            actual_total = est_bytes
            try:
                head = requests.head(
                    url, headers=hf_headers, allow_redirects=True, timeout=15
                )
                cl = int(head.headers.get("content-length", 0))
                if cl > 0:
                    actual_total = cl
            except Exception:
                pass

            # Update grand total using actual size for this model slot.
            new_total = bytes_done_before + actual_total + remaining_est
            if new_total > 0:
                progress_state["total_bytes"] = max(
                    progress_state.get("total_bytes", 0), new_total
                )

            resp = requests.get(
                url, headers=hf_headers, stream=True, allow_redirects=True, timeout=60
            )
            resp.raise_for_status()

            # Fall back to Content-Length from GET response if HEAD gave nothing.
            if actual_total == est_bytes:
                cl = int(resp.headers.get("content-length", 0))
                if cl > 0:
                    actual_total = cl
                    new_total = bytes_done_before + actual_total + remaining_est
                    if new_total > 0:
                        progress_state["total_bytes"] = max(
                            progress_state.get("total_bytes", 0), new_total
                        )

            bytes_this_model = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if cancel_event.is_set():
                        progress_state["cancelled"] = True
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        return
                    if chunk:
                        f.write(chunk)
                        bytes_this_model += len(chunk)
                        progress_state["downloaded_bytes"] = (
                            bytes_done_before + bytes_this_model
                        )

            os.replace(tmp_path, local_path)
            tmp_path = None
            bytes_done_before += bytes_this_model
            progress_state["downloaded_bytes"] = bytes_done_before

        progress_state["done_flag"] = True
    except Exception as exc:
        progress_state["error"] = str(exc)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def download_models(entries: list) -> bool:
    """
    Show a single QProgressDialog and download the given model entries in a
    background thread.

    Parameters
    ----------
    entries : list[dict]
        Model entries from ``MODELS`` or ``MODEL_SETS`` values.

    Returns
    -------
    bool
        True when all models were downloaded successfully.
        False when the user cancelled or a network error occurred.

    Guard
    -----
    Returns True immediately when ``slicer.app.testingEnabled()`` is True,
    so automated tests never show a dialog or touch the network.
    """
    try:
        import slicer
        if slicer.app.testingEnabled():
            return True
    except ImportError:
        return True

    if not entries:
        return True

    import qt

    total_bytes_estimate = sum(e.get("size_bytes", 0) for e in entries)

    models_with_urls = [
        (
            f"https://huggingface.co/{e['repo_id']}/resolve/{e['revision']}/{e['filename']}",
            e["filename"],
            e["label"],
            e.get("size_bytes", 0),
        )
        for e in entries
    ]
    hf_headers: dict = {}

    progress_state: dict = {
        "downloaded_bytes": 0,
        "total_bytes": total_bytes_estimate,
        "current_label": entries[0]["label"],
        "done_flag": False,
        "cancelled": False,
        "error": None,
    }
    cancel_event = threading.Event()

    first_label = entries[0]["label"]
    dlg = qt.QProgressDialog(
        f"Downloading {first_label}…",
        "Cancel",
        0,
        1,
        slicer.util.mainWindow(),
    )
    dlg.setValue(0)
    dlg.setWindowTitle("Downloading Models")
    dlg.setWindowModality(qt.Qt.ApplicationModal)
    dlg.setMinimumWidth(420)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    slicer.app.processEvents()

    thread = threading.Thread(
        target=_run_downloads,
        args=(models_with_urls, hf_headers, progress_state, cancel_event),
        daemon=True,
    )
    thread.start()
    _determinate = False
    t0 = _time.time()

    while (
        thread.is_alive()
        and not progress_state["done_flag"]
        and not progress_state["cancelled"]
    ):
        if dlg.wasCanceled:
            cancel_event.set()
            progress_state["cancelled"] = True
            break

        downloaded = progress_state.get("downloaded_bytes", 0)
        total = progress_state.get("total_bytes", 0)
        current_label = progress_state.get("current_label", "")
        elapsed = _time.time() - t0

        downloaded_mb = downloaded / 1_048_576
        total_mb = total / 1_048_576

        if total > 0:
            if not _determinate:
                dlg.setRange(0, 1000)
                _determinate = True
            permille = int(downloaded / total * 1000)
            dlg.setValue(permille)
            if total_mb > 0 and downloaded_mb > 0.1 and elapsed > 2:
                rate = downloaded / elapsed  # bytes/sec
                remaining_bytes = total - downloaded
                eta_s = remaining_bytes / rate if rate > 0 else 0
                if eta_s >= 3600:
                    eta_str = f"~{int(eta_s // 3600)}h {int((eta_s % 3600) // 60)}m left"
                elif eta_s >= 60:
                    eta_str = f"~{int(eta_s // 60)}m {int(eta_s % 60):02d}s left"
                else:
                    eta_str = f"~{int(eta_s)}s left"
                label_text = (
                    f"Downloading {current_label}…\n"
                    f"{downloaded_mb:.1f} / {total_mb:.1f} MB  ·  {eta_str}"
                )
            else:
                label_text = (
                    f"Downloading {current_label}…\n"
                    f"{downloaded_mb:.1f} / {total_mb:.1f} MB"
                )
            dlg.setLabelText(label_text)
        elif downloaded > 0:
            dlg.setLabelText(
                f"Downloading {current_label}…\n"
                f"{downloaded_mb:.1f} MB  ·  {int(elapsed)}s elapsed"
            )
        else:
            dlg.setLabelText(
                f"Downloading {current_label}…\n"
                f"{int(elapsed)}s elapsed"
            )

        slicer.app.processEvents()
        _time.sleep(0.2)

    thread.join(timeout=2.0)
    dlg.close()

    if progress_state.get("cancelled"):
        return False

    if progress_state.get("error"):
        slicer.util.errorDisplay(
            f"Model download failed:\n{progress_state['error']}\n\n"
            "Check your internet connection and try again."
        )
        return False

    return True
