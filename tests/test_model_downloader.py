import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urljoin, urlparse

import pytest


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        if slot in self.slots:
            self.slots.remove(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class FakeQUrl:
    def __init__(self, url):
        self._url = str(url)

    def isValid(self):
        return bool(self._url)

    def isRelative(self):
        return not bool(urlparse(self._url).scheme)

    def resolved(self, other):
        return FakeQUrl(urljoin(self._url, other.toString()))

    def scheme(self):
        return urlparse(self._url).scheme

    def toString(self):
        return self._url

    def __str__(self):
        return self._url


class FakeRequest:
    RedirectPolicyAttribute = "redirect-policy"
    RedirectionTargetAttribute = "redirect-target"
    HttpStatusCodeAttribute = "http-status"
    NoLessSafeRedirectPolicy = "no-less-safe"

    def __init__(self, url):
        self.url = url
        self.attributes = {}

    def setAttribute(self, attr, value):
        self.attributes[attr] = value


class FakeDialog:
    def __init__(self, *args):
        self.canceled = Signal()
        self.closed = False
        self.deleted = False
        self.labels = []
        self.values = []

    def setWindowTitle(self, title): pass
    def setWindowModality(self, modality): pass
    def setMinimumWidth(self, width): pass
    def setAutoClose(self, value): pass
    def setAutoReset(self, value): pass
    def show(self): pass
    def setRange(self, low, high): self.range = (low, high)
    def setValue(self, value): self.values.append(value)
    def setLabelText(self, text): self.labels.append(text)
    def close(self): self.closed = True
    def deleteLater(self): self.deleted = True


class FakeQt:
    QNetworkRequest = FakeRequest
    QUrl = FakeQUrl
    QProgressDialog = FakeDialog

    class Qt:
        ApplicationModal = 1


class FakeReply:
    def __init__(self, url, *, status=200, chunks=None, redirect=None, error=None):
        self._url = FakeQUrl(url)
        self.status = status
        self.redirect = FakeQUrl(redirect) if redirect is not None else None
        self.error_text = error
        self.readyRead = Signal()
        self.downloadProgress = Signal()
        self.finished = Signal()
        self.errorOccurred = Signal()
        self._chunks = list(chunks or [])
        self.aborted = False
        self.deleted = False

    def url(self):
        return self._url

    def readAll(self):
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def attribute(self, attr):
        if attr == FakeRequest.HttpStatusCodeAttribute:
            return self.status
        if attr == FakeRequest.RedirectionTargetAttribute:
            return self.redirect
        return None

    def errorString(self):
        return self.error_text or "Network error"

    def abort(self):
        self.aborted = True

    def deleteLater(self):
        self.deleted = True


class FakeManager:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def get(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        reply._url = request.url
        return reply


def _entry(name="model.pth", data=b"data", **extra):
    item = {
        "repo_id": "org/repo",
        "revision": "main",
        "filename": name,
        "label": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    item.update(extra)
    return item


@pytest.fixture
def md(tmp_path, monkeypatch):
    import importlib
    import ZebrafishEmbryoAnalyzerLib.model_downloader as module

    module = importlib.reload(module)
    monkeypatch.setattr(module, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        "ZebrafishEmbryoAnalyzerLib.model_manifest._CACHE_DIR",
        tmp_path,
        raising=False,
    )
    return module


def _controller(md, entries, manager):
    calls = []
    slicer = MagicMock()
    slicer.util.mainWindow.return_value = None
    controller = md.start_model_download(
        entries,
        lambda *args: calls.append(args),
        qt_module=FakeQt,
        slicer_module=slicer,
        manager_factory=lambda: manager,
    )
    return controller, calls, slicer


def _finish_success(reply, data=b"data"):
    reply.readyRead.emit()
    reply.downloadProgress.emit(len(data), len(data))
    reply.finished.emit()


def test_download_models_guard_returns_true_in_testing_mode(monkeypatch):
    slicer = MagicMock()
    slicer.app.testingEnabled.return_value = True
    monkeypatch.setitem(sys.modules, "slicer", slicer)

    from ZebrafishEmbryoAnalyzerLib.model_downloader import download_models

    assert download_models([_entry()]) is True
    slicer.util.mainWindow.assert_not_called()


def test_successful_2xx_response_replaces_after_integrity(md, tmp_path):
    reply = FakeReply("https://example.test/model.pth", chunks=[b"data"])
    manager = FakeManager([reply])
    controller, calls, _ = _controller(md, [_entry()], manager)

    _finish_success(reply)

    assert calls == [(True, "succeeded", None, controller)]
    assert (tmp_path / "model.pth").read_bytes() == b"data"
    assert not (tmp_path / "model.pth.tmp").exists()


def test_progress_label_includes_rate_and_eta(md, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(md.time, "time", lambda: now[0])
    reply = FakeReply("https://example.test/model.pth", chunks=[b"data"])
    manager = FakeManager([reply])
    controller, _, _ = _controller(md, [_entry(data=b"data", size_bytes=8)], manager)

    reply.downloadProgress.emit(1, 8)
    now[0] = 104.0
    reply.downloadProgress.emit(4, 8)

    labels = controller._dialog.labels
    assert any("MB/s" in label for label in labels)
    assert any("left" in label for label in labels)


def test_progress_rate_uses_current_delta_not_session_average(md, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(md.time, "time", lambda: now[0])
    mib = 1_048_576
    reply = FakeReply("https://example.test/model.pth")
    manager = FakeManager([reply])
    controller, _, _ = _controller(md, [_entry(data=b"", size_bytes=100 * mib)], manager)

    reply.downloadProgress.emit(1 * mib, 100 * mib)
    now[0] = 110.0
    reply.downloadProgress.emit(2 * mib, 100 * mib)
    assert "0.1 MB/s" in controller._dialog.labels[-1]

    now[0] = 111.0
    reply.downloadProgress.emit(12 * mib, 100 * mib)
    assert "10.0 MB/s" in controller._dialog.labels[-1]


def test_progress_total_does_not_drop_below_prompt_estimate(md, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(md.time, "time", lambda: now[0])
    mib = 1_048_576
    reply = FakeReply("https://example.test/model.pth")
    manager = FakeManager([reply])
    controller, _, _ = _controller(
        md,
        [_entry(data=b"", size_bytes=1400 * mib)],
        manager,
    )

    reply.downloadProgress.emit(1 * mib, 980 * mib)
    now[0] = 104.0
    reply.downloadProgress.emit(10 * mib, 980 * mib)

    label = controller._dialog.labels[-1]
    assert "10.0 / 1400.0 MB" in label
    assert "MB/s" in label
    assert "Current file" not in label
    assert "estimated total" not in label


def test_absolute_https_redirect(md, tmp_path):
    redirect = FakeReply(
        "https://huggingface.co/a",
        status=302,
        redirect="https://cdn.example.test/model.pth",
    )
    final = FakeReply("https://cdn.example.test/model.pth", chunks=[b"data"])
    manager = FakeManager([redirect, final])
    controller, calls, _ = _controller(md, [_entry()], manager)

    redirect.finished.emit()
    _finish_success(final)

    assert calls == [(True, "succeeded", None, controller)]
    assert manager.requests[1].url.toString() == "https://cdn.example.test/model.pth"
    assert (tmp_path / "model.pth").read_bytes() == b"data"


def test_relative_redirect(md):
    redirect = FakeReply(
        "https://host.test/path/start",
        status=302,
        redirect="../files/model.pth",
    )
    final = FakeReply("https://host.test/files/model.pth", chunks=[b"data"])
    manager = FakeManager([redirect, final])
    _, calls, _ = _controller(md, [_entry()], manager)

    redirect.finished.emit()
    _finish_success(final)

    assert calls[0][0] is True
    assert manager.requests[1].url.toString() == (
        "https://huggingface.co/org/repo/resolve/files/model.pth"
    )


def test_redirect_limit_exceeded_cleans_tmp(md, tmp_path, monkeypatch):
    monkeypatch.setattr(md, "_MAX_REDIRECTS", 0)
    reply = FakeReply("https://host.test/a", status=302, redirect="https://host.test/b")
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry()], manager)

    reply.readyRead.emit()
    reply.finished.emit()

    assert calls[0][1] == "failed"
    assert not (tmp_path / "model.pth.tmp").exists()


def test_https_to_http_redirect_rejected(md, tmp_path):
    reply = FakeReply("https://host.test/a", status=302, redirect="http://host.test/b")
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry()], manager)

    reply.finished.emit()

    assert calls[0][1] == "failed"
    assert "unsafe" in calls[0][2]
    assert not (tmp_path / "model.pth.tmp").exists()


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_cleans_tmp_and_preserves_cache(md, tmp_path, status):
    final = tmp_path / "model.pth"
    final.write_bytes(b"valid")
    reply = FakeReply("https://example.test/model.pth", status=status, chunks=[b"bad"])
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry()], manager)

    reply.readyRead.emit()
    reply.finished.emit()

    assert calls[0][1] == "failed"
    assert final.read_bytes() == b"valid"
    assert not (tmp_path / "model.pth.tmp").exists()


def test_network_error_and_finished_callback_once(md, tmp_path):
    reply = FakeReply("https://example.test/model.pth", chunks=[b"bad"], error="offline")
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry()], manager)

    reply.readyRead.emit()
    reply.errorOccurred.emit(1)
    reply.finished.emit()
    reply.finished.emit()

    assert len(calls) == 1
    assert calls[0][1] == "failed"
    assert "offline" in calls[0][2]
    assert not (tmp_path / "model.pth.tmp").exists()


def test_cancel_racing_finished_callback_once(md, tmp_path):
    reply = FakeReply("https://example.test/model.pth", chunks=[b"partial"])
    manager = FakeManager([reply])
    controller, calls, _ = _controller(md, [_entry()], manager)

    reply.readyRead.emit()
    controller.cancel()
    reply.finished.emit()

    assert len(calls) == 1
    assert calls[0][1] == "cancelled"
    assert reply.aborted is True
    assert not (tmp_path / "model.pth.tmp").exists()


def test_cancel_rolls_back_completed_session_file(md, tmp_path):
    first = FakeReply("https://example.test/a.pth", chunks=[b"a"])
    second = FakeReply("https://example.test/b.pth", chunks=[b"partial"])
    manager = FakeManager([first, second])
    entries = [_entry("a.pth", b"a"), _entry("b.pth", b"b")]
    controller, calls, _ = _controller(md, entries, manager)

    first.readyRead.emit()
    first.finished.emit()
    assert (tmp_path / "a.pth").exists()

    second.readyRead.emit()
    controller.cancel()

    assert calls[0][1] == "cancelled"
    assert not (tmp_path / "a.pth").exists()
    assert not (tmp_path / "b.pth").exists()
    assert not (tmp_path / "b.pth.tmp").exists()


def test_failure_rolls_back_replaced_existing_file(md, tmp_path):
    original = tmp_path / "a.pth"
    original.write_bytes(b"original")
    first = FakeReply("https://example.test/a.pth", chunks=[b"new"])
    second = FakeReply("https://example.test/b.pth", status=500, chunks=[b"bad"])
    manager = FakeManager([first, second])
    entries = [_entry("a.pth", b"new"), _entry("b.pth", b"b")]
    _, calls, _ = _controller(md, entries, manager)

    first.readyRead.emit()
    first.finished.emit()
    assert original.read_bytes() == b"new"

    second.readyRead.emit()
    second.finished.emit()

    assert calls[0][1] == "failed"
    assert original.read_bytes() == b"original"
    assert not (tmp_path / "a.pth.download-session-backup").exists()


def test_file_write_error_cleans_tmp(md, monkeypatch):
    reply = FakeReply("https://example.test/model.pth", chunks=[b"data"])
    manager = FakeManager([reply])
    controller, calls, _ = _controller(md, [_entry()], manager)

    def fail_write(data):
        raise OSError("disk full")

    controller.tmp_file.write = fail_write
    reply.readyRead.emit()
    reply.finished.emit()

    assert len(calls) == 1
    assert calls[0][1] == "failed"
    assert "disk full" in calls[0][2]


def test_exact_size_mismatch_rejected_before_replace(md, tmp_path):
    final = tmp_path / "model.pth"
    final.write_bytes(b"valid")
    reply = FakeReply("https://example.test/model.pth", chunks=[b"short"])
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry(exact_size_bytes=10)], manager)

    _finish_success(reply, b"short")

    assert calls[0][1] == "failed"
    assert final.read_bytes() == b"valid"


def test_checksum_mismatch_rejected(md, tmp_path):
    final = tmp_path / "model.pth"
    final.write_bytes(b"valid")
    reply = FakeReply("https://example.test/model.pth", chunks=[b"bad"])
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry(sha256="0" * 64)], manager)

    _finish_success(reply, b"bad")

    assert calls[0][1] == "failed"
    assert final.read_bytes() == b"valid"


def test_checksum_match_replaces(md, tmp_path):
    data = b"verified"
    sha = hashlib.sha256(data).hexdigest()
    reply = FakeReply("https://example.test/model.pth", chunks=[data])
    manager = FakeManager([reply])
    _, calls, _ = _controller(md, [_entry(data=data, sha256=sha, exact_size_bytes=len(data))], manager)

    _finish_success(reply, data)

    assert calls[0][0] is True
    assert (tmp_path / "model.pth").read_bytes() == data


def test_sequential_multi_model_download(md, tmp_path):
    first = FakeReply("https://example.test/a.pth", chunks=[b"a"])
    second = FakeReply("https://example.test/b.pth", chunks=[b"bb"])
    manager = FakeManager([first, second])
    entries = [_entry("a.pth", b"a"), _entry("b.pth", b"bb")]
    _, calls, _ = _controller(md, entries, manager)

    first.readyRead.emit()
    first.finished.emit()
    assert len(manager.requests) == 2
    assert calls == []

    second.readyRead.emit()
    second.finished.emit()
    assert len(calls) == 1
    assert calls[0][0] is True
    assert (tmp_path / "a.pth").read_bytes() == b"a"
    assert (tmp_path / "b.pth").read_bytes() == b"bb"


def test_failure_on_second_prevents_third(md):
    first = FakeReply("https://example.test/a.pth", chunks=[b"a"])
    second = FakeReply("https://example.test/b.pth", status=500, chunks=[b"bad"])
    third = FakeReply("https://example.test/c.pth", chunks=[b"c"])
    manager = FakeManager([first, second, third])
    entries = [_entry("a.pth", b"a"), _entry("b.pth", b"b"), _entry("c.pth", b"c")]
    _, calls, _ = _controller(md, entries, manager)

    first.readyRead.emit()
    first.finished.emit()
    second.readyRead.emit()
    second.finished.emit()

    assert len(calls) == 1
    assert calls[0][1] == "failed"
    assert len(manager.requests) == 2


def test_dispose_aborts_and_suppresses_callback(md, tmp_path):
    reply = FakeReply("https://example.test/model.pth", chunks=[b"partial"])
    manager = FakeManager([reply])
    controller, calls, _ = _controller(md, [_entry()], manager)

    reply.readyRead.emit()
    controller.dispose()
    reply.finished.emit()

    assert calls == []
    assert reply.aborted is True
    assert not (tmp_path / "model.pth.tmp").exists()
