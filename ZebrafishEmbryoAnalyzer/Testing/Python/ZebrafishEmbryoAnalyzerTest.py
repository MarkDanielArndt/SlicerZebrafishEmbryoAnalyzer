"""CTest entry point for the ZebrafishEmbryoAnalyzer self-test.

The test case itself lives in ZebrafishEmbryoAnalyzer.py, because Slicer's
"Reload and Test" button resolves it as an attribute of the module namespace
(ScriptedLoadableModule.runTest). This script re-exports it, so
slicer.testing.runUnitTest — which loads tests from the module named after this
file — keeps finding it.

Run inside Slicer only — requires slicer, vtk, and MRML APIs.
Not executable with plain pytest.
"""

from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerTest  # noqa: F401
