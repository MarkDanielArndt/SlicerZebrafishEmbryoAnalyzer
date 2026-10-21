"""
Tests for BLK-04 (safe checkpoint loading) and BLK-06 (no timm pretrained download).

These tests are static and behavioural — no model files are required.
"""

import os
import subprocess
import sys

import pytest

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(WORKTREE, "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzerCore")
LIB_DIR = os.path.join(WORKTREE, "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzerLib")

PRODUCTION_DIRS = [CORE_DIR, LIB_DIR]


# ---------------------------------------------------------------------------
# BLK-04: every torch.load in production source uses weights_only=True
# ---------------------------------------------------------------------------

def _grep_production(pattern):
    """Return grep stdout for pattern across all production source dirs."""
    results = []
    for d in PRODUCTION_DIRS:
        if not os.path.isdir(d):
            continue
        result = subprocess.run(
            ["grep", "-r", "--include=*.py", "-n", pattern, d],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            results.append(result.stdout.strip())
    return "\n".join(results)


def test_no_torch_load_without_weights_only():
    """Every torch.load() in production source must include weights_only=True."""
    # Find all torch.load calls
    all_calls = _grep_production("torch\\.load(")
    if not all_calls:
        return  # no torch.load calls at all — trivially satisfied

    # Every line with torch.load must also contain weights_only=True
    bad_lines = []
    for line in all_calls.splitlines():
        if "torch.load(" in line and "weights_only=True" not in line:
            bad_lines.append(line)

    assert not bad_lines, (
        "Found torch.load() calls without weights_only=True in production source:\n"
        + "\n".join(bad_lines)
    )


def test_no_weights_only_false_in_production():
    """weights_only=False must not appear in runtime production source."""
    hits = _grep_production("weights_only=False")
    assert not hits.strip(), (
        f"weights_only=False found in production source:\n{hits}"
    )


# ---------------------------------------------------------------------------
# BLK-04: seg.py raises RuntimeError (not silent None) on bad checkpoint
# ---------------------------------------------------------------------------

def test_seg_loader_raises_runtime_error_on_bad_checkpoint(tmp_path):
    """_load_unet_model(local_only=True) raises RuntimeError on a corrupt checkpoint."""
    # find_spec instead of importorskip: importing torch/smp in the pytest process
    # would leak into other tests that stub those modules; the subprocess imports them.
    import importlib.util
    for mod in ("torch", "segmentation_models_pytorch", "zebrafish_measure"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed")
    # Write a file that is not a valid PyTorch state dict (simulates pickled model)
    bad_file = tmp_path / "bad.pth"
    bad_file.write_bytes(b"not a valid pytorch file")

    module_dir = os.path.join(WORKTREE, "ZebrafishEmbryoAnalyzer")
    script = f"""
import sys
sys.path.insert(0, {module_dir!r})
from ZebrafishEmbryoAnalyzerCore.seg import _load_unet_model
try:
    _load_unet_model(model_path={str(bad_file)!r}, label="test model",
                     encoder_name="vgg16", model_type="Unet",
                     local_only=True, device="cpu")
    print("NO_ERROR")
except RuntimeError as e:
    print("RUNTIME_ERROR:", str(e)[:80])
except Exception as e:
    print("OTHER_ERROR:", type(e).__name__, str(e)[:80])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout + result.stderr
    assert "RUNTIME_ERROR:" in result.stdout, out
    assert "weights_only=False" not in out, "seg.py fell back to weights_only=False"


# ---------------------------------------------------------------------------
# BLK-06: no pretrained=True in production timm calls
# ---------------------------------------------------------------------------

def test_timm_pretrained_false_in_production_code():
    """Static check: no timm.create_model call uses pretrained=True."""
    hits = _grep_production("pretrained=True")
    assert not hits.strip(), (
        f"pretrained=True found in production source:\n{hits}"
    )


# ---------------------------------------------------------------------------
# Static: the library code the shims delegate to uses weights_only=True
# (the Core shims themselves contain no loading code; the library's own test
# suite covers its loaders in depth)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("submodule", ["length", "seg"])
def test_library_loaders_use_weights_only_true(submodule):
    """zebrafish_measure.<length|seg> must contain weights_only=True and never False."""
    import importlib.util
    spec = importlib.util.find_spec(f"zebrafish_measure.{submodule}")
    assert spec is not None and spec.origin, "zebrafish-measure is not installed"
    content = open(spec.origin, encoding="utf-8").read()
    assert "weights_only=True" in content
    assert "weights_only=False" not in content


def test_logic_always_passes_local_only_true():
    """logic.py must never let the library contact Hugging Face: every
    segmentation_pipeline kwargs dict and every _cached_load_unet call passes
    local_only=True."""
    import re
    logic_path = os.path.join(LIB_DIR, "logic.py")
    content = open(logic_path, encoding="utf-8").read()
    # _seg_kwargs for segmentation_pipeline
    assert re.search(r"_seg_kwargs\s*=\s*dict\((?:.|\n)*?local_only=True", content)
    # each _cached_load_unet(...) call in preload_models (skip the def itself)
    calls = re.findall(r"\n\s+_cached_load_unet\((.*?)\n\s{4,8}\)", content, re.S)
    assert len(calls) == 4
    for c in calls:
        assert "local_only=True" in c
