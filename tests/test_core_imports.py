import os
import subprocess
import sys
import textwrap


MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ZebrafishEmbryoAnalyzer"
)


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": MODULE_DIR},
    )


def test_core_length_imports_without_matplotlib():
    # Block matplotlib so `import matplotlib[.pyplot]` raises ImportError, regardless
    # of whether matplotlib is installed in the running environment. The normal
    # analysis entry points used by the Slicer extension must still import.
    result = _run_in_subprocess(
        """
        import sys
        sys.modules["matplotlib"] = None
        sys.modules["matplotlib.pyplot"] = None
        from ZebrafishEmbryoAnalyzerCore.length import (
            load_model,
            tube_length_border2border,
            classification_curvature,
        )
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "core.length must import without matplotlib.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_core_modules_import_without_segmentation_models_pytorch():
    # segmentation_models_pytorch is only available inside Slicer's Python
    # environment. Both core modules must be importable without it; only the
    # functions that actually load models should require it at call time.
    result = _run_in_subprocess(
        """
        import sys
        sys.modules["segmentation_models_pytorch"] = None
        from ZebrafishEmbryoAnalyzerCore import seg, length
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "core modules must import without segmentation_models_pytorch.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
