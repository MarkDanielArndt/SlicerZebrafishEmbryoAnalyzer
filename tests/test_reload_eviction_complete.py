"""Every shipped submodule must be listed for eviction on module reload.

A missing entry is invisible until someone edits that file and reloads: Slicer
serves the cached copy, so the change appears to have no effect. Both halves of
this list have been wrong in the past — Core was omitted entirely, and
bug_report.py was forgotten when the Developer Tools panel arrived.

Parsed rather than imported: ZebrafishEmbryoAnalyzer.py imports slicer and vtk
at module level, which plain pytest has no way to provide.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MODULE_FILE = ROOT / "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py"
PACKAGES = {
    "_LIB_MODULES": ROOT / "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzerLib",
    "_CORE_MODULES": ROOT / "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzerCore",
}


def _listed_modules(name):
    """Return the string entries of a module-level tuple assignment."""
    tree = ast.parse(MODULE_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
            }
    raise AssertionError(f"{name} not found in {MODULE_FILE.name}")


def _shipped_modules(package_dir):
    """Qualified names of every .py in the package, __init__ excluded."""
    return {
        f"{package_dir.name}.{path.stem}"
        for path in package_dir.glob("*.py")
        if path.stem != "__init__"
    }


@pytest.mark.parametrize("list_name", sorted(PACKAGES))
def test_every_submodule_is_evicted_on_reload(list_name):
    listed = _listed_modules(list_name)
    shipped = _shipped_modules(PACKAGES[list_name])

    missing = shipped - listed
    assert not missing, (
        f"{list_name} is missing {sorted(missing)} — editing those files and "
        "reloading the module in Slicer would keep serving the cached copy"
    )

    stale = listed - shipped
    assert not stale, f"{list_name} lists {sorted(stale)}, which no longer exist"


def test_both_lists_are_evicted():
    """A list that exists but is never evicted protects nothing."""
    source = MODULE_FILE.read_text(encoding="utf-8")
    assert "_RELOAD_MODULES = _LIB_MODULES + _CORE_MODULES" in source
    assert "for _m in _RELOAD_MODULES:" in source
