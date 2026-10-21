"""
Bug-report assembly for the Developer Tools panel.

build_report() is pure Python plus optional ``slicer`` access and never
raises — a broken environment still produces a partial report instead of
none. save_report() is Slicer-only, called only from an explicit user action.
"""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from ZebrafishEmbryoAnalyzerLib.dependency_installer import REQUIRED_PACKAGES, TORCH_PACKAGES

_TRACKED_PACKAGES = ["numpy"] + REQUIRED_PACKAGES + TORCH_PACKAGES


def is_developer_mode() -> bool:
    """Whether Slicer's own Developer Mode setting is enabled.

    Mirrors the check ``ScriptedLoadableModuleWidget.setup()`` does
    internally (``Developer/DeveloperMode`` setting) — our widget hierarchy
    does not expose that flag to this module.
    """
    import slicer
    return slicer.util.settingsValue(
        "Developer/DeveloperMode", False, converter=slicer.util.toBool
    )


def _extension_commit() -> str:
    """Short git commit of this extension checkout, or a documented fallback."""
    repo_dir = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git checkout)"


def _package_versions() -> list:
    lines = []
    for name in _TRACKED_PACKAGES:
        try:
            lines.append(f"- {name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"- {name}: not installed")
    return lines


def _slicer_info() -> list:
    try:
        import slicer
    except ImportError:
        return ["- Slicer: not running inside Slicer"]
    try:
        return [
            f"- Slicer version: {slicer.app.majorVersion}.{slicer.app.minorVersion} "
            f"({slicer.app.releaseType}, revision {slicer.app.revision})",
            f"- OS: {slicer.app.os} ({slicer.app.arch})",
            f"- Developer mode: {is_developer_mode()}",
        ]
    except Exception as exc:
        # Never let a malformed/partial `slicer` (e.g. a test stub left over
        # in sys.modules) break bug-report generation.
        return [f"- Slicer: environment info unavailable ({exc})"]


def build_report() -> str:
    """Assemble a Markdown bug report from the current environment."""
    sections = [
        "## Environment",
        *_slicer_info(),
        f"- Extension commit: {_extension_commit()}",
        f"- Python: {sys.version.split()[0]} ({platform.platform()})",
        "",
        "## Package versions",
        *_package_versions(),
        "",
        "## Steps to reproduce",
        "1. ",
        "",
        "## Expected vs. actual",
        "- Expected: ",
        "- Actual: ",
    ]
    return "\n".join(sections)


def save_report(report: str) -> str:
    """Write ``report`` to a timestamped file in Slicer's temporary directory.

    Returns the path written to. Raises on failure — the caller decides how
    to surface that to the user.
    """
    import datetime
    import slicer
    out_dir = Path(slicer.app.temporaryPath) / "ZebrafishEmbryoAnalyzerBugReports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"bug-report-{stamp}.md"
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)
