"""Verify that Slicer can find the self-test through both of its entry points.

"Reload and Test" resolves the class as an attribute of the module namespace
(ScriptedLoadableModule.runTest), while CTest imports the Testing/Python script
by name. Both must reach the same class, and it must exist exactly once.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODULE_FILE = ROOT / "ZebrafishEmbryoAnalyzer" / "ZebrafishEmbryoAnalyzer.py"
CTEST_SCRIPT = (
    ROOT / "ZebrafishEmbryoAnalyzer" / "Testing" / "Python" / "ZebrafishEmbryoAnalyzerTest.py"
)
TEST_CLASS = "ZebrafishEmbryoAnalyzerTest"


def _top_level_classes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def test_test_class_lives_in_the_module_namespace():
    """Without this, Reload and Test raises 'Test case class not found'."""
    assert TEST_CLASS in _top_level_classes(MODULE_FILE)


def test_ctest_script_reexports_rather_than_redefines():
    """A second definition would let the two entry points drift apart."""
    assert TEST_CLASS not in _top_level_classes(CTEST_SCRIPT)

    tree = ast.parse(CTEST_SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "ZebrafishEmbryoAnalyzer"
        for alias in node.names
    }
    assert TEST_CLASS in imported
