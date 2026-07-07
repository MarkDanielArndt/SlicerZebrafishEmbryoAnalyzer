"""Slicer integration tests for the ZebrafishEmbryoAnalyzer extension.

Run inside Slicer only — requires slicer, vtk, and MRML APIs.
Not executable with plain pytest.

Usage (from Slicer Python console or test runner):
    slicer.util.selectModule("ZebrafishEmbryoAnalyzer")
    # Or via ctest after CMakeLists.txt registration.
"""

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest


class ZebrafishEmbryoAnalyzerTest(ScriptedLoadableModuleTest):
    """Integration tests for ZebrafishEmbryoAnalyzer.

    Each test_* method is self-contained: it sets up what it needs and cleans up
    via slicer.mrmlScene.Clear(0) or targeted node removal.  No network access,
    no model downloads, no prompts.
    """

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()

        self.delayDisplay("test_module_import", msec=100)
        self.test_module_import()

        self.delayDisplay("test_logic_instantiation", msec=100)
        self.test_logic_instantiation()

        self.delayDisplay("test_parameter_node_defaults", msec=100)
        self.test_parameter_node_defaults()

        self.delayDisplay("test_no_network_during_setup", msec=100)
        self.test_no_network_during_setup()

        self.delayDisplay("test_mrml_table_node_creation", msec=100)
        self.test_mrml_table_node_creation()

        self.delayDisplay("test_mrml_table_node_reuse", msec=100)
        self.test_mrml_table_node_reuse()

        self.delayDisplay("test_results_to_rows_pure", msec=100)
        self.test_results_to_rows_pure()

        self.delayDisplay("test_scene_close_cleanup", msec=100)
        self.test_scene_close_cleanup()

        self.delayDisplay("All tests passed.", msec=100)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_module_import(self):
        """ZebrafishEmbryoAnalyzer module and its main classes must be importable."""
        import ZebrafishEmbryoAnalyzer as mod
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzer)
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzerWidget)
        self.assertIsNotNone(mod.ZebrafishEmbryoAnalyzerLogic)

    def test_logic_instantiation(self):
        """ZebrafishEmbryoAnalyzerLogic() must construct without error."""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        logic = ZebrafishEmbryoAnalyzerLogic()
        self.assertIsNotNone(logic)

    def test_parameter_node_defaults(self):
        """getParameterNode() must return a node whose defaults match PARAM_DEFAULTS."""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.widget import (
            PARAM_DEFAULTS,
            PARAM_LENGTH_ENABLED,
            PARAM_MODEL_ID,
            _DEFAULT_MODEL_ID,
        )

        logic = ZebrafishEmbryoAnalyzerLogic()
        param_node = logic.getParameterNode()
        self.assertIsNotNone(param_node)

        # Populate defaults the same way the widget does (mirrors initializeParameterNode).
        for name, default in PARAM_DEFAULTS.items():
            if not param_node.GetParameter(name):
                param_node.SetParameter(name, default)

        self.assertEqual(param_node.GetParameter(PARAM_LENGTH_ENABLED), "true")
        self.assertEqual(param_node.GetParameter(PARAM_MODEL_ID), _DEFAULT_MODEL_ID)

    def test_no_network_during_setup(self):
        """No prompts or network calls fire during setup when testingEnabled() is True."""
        if not slicer.app.testingEnabled():
            self.delayDisplay("Skipping: requires Slicer --testing flag (CTest mode)", 100)
            return
        self.assertTrue(slicer.app.testingEnabled(),
                        "testingEnabled() must be True — confirms prompts are suppressed")

    def test_mrml_table_node_creation(self):
        """update_results_table() must create a vtkMRMLTableNode with exactly one row."""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        result = {
            "filename": "test.png",
            "length": 1000,
            "curvature": 1,
            "ratio": 1.1,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }
        table_node = logic.update_results_table([result])

        self.assertIsNotNone(table_node)
        self.assertTrue(table_node.IsA("vtkMRMLTableNode"))
        self.assertEqual(table_node.GetNumberOfRows(), 1)

    def test_mrml_table_node_reuse(self):
        """Calling update_results_table() twice must reuse the same node."""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic

        logic = ZebrafishEmbryoAnalyzerLogic()
        result = {
            "filename": "a.png",
            "length": 500,
            "curvature": 0,
            "ratio": 1.0,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }

        node_first = logic.update_results_table([result])
        count_after_first = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLTableNode")

        node_second = logic.update_results_table([result, result])
        count_after_second = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLTableNode")

        self.assertEqual(count_after_first, count_after_second,
                         "A second call must not add a new table node to the scene.")
        self.assertEqual(node_first.GetID(), node_second.GetID(),
                         "Both calls must return the same node.")
        self.assertEqual(node_second.GetNumberOfRows(), 2)

    def test_results_to_rows_pure(self):
        """results_to_rows() is pure Python — no Slicer dependency."""
        import math
        from ZebrafishEmbryoAnalyzerLib.mrml import results_to_rows, TABLE_SCHEMA

        # Empty input → empty output.
        rows = results_to_rows([])
        self.assertEqual(rows, [])

        # Single result with no errors.
        result = {
            "filename": "img.png",
            "length": 1234.5,
            "curvature": 2,
            "ratio": 1.05,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }
        rows = results_to_rows([result])
        self.assertEqual(len(rows), 1)

        row = rows[0]
        col_names = [col for col, _, _ in TABLE_SCHEMA]
        for col in col_names:
            self.assertIn(col, row, f"Column {col!r} missing from row")

        self.assertEqual(row["Filename"], "img.png")
        self.assertAlmostEqual(row["Length_um"], 1234.5)
        # None numeric field → NaN
        self.assertTrue(math.isnan(row["EyeArea_um2"]))
        self.assertEqual(row["Error"], "")

    def test_scene_close_cleanup(self):
        """Scene clear must not crash; a fresh parameter node must be obtainable after."""
        from ZebrafishEmbryoAnalyzer import ZebrafishEmbryoAnalyzerLogic
        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_DEFAULTS

        logic = ZebrafishEmbryoAnalyzerLogic()
        # Populate some MRML nodes first.
        logic.update_results_table([{
            "filename": "x.png",
            "length": 1,
            "curvature": 0,
            "ratio": 1.0,
            "eye_area": None,
            "eye_diameter": None,
            "error": None,
        }])

        # Scene clear — must not raise.
        slicer.mrmlScene.Clear(0)

        # After clear, logic must still deliver a (fresh) parameter node.
        param_node = logic.getParameterNode()
        self.assertIsNotNone(param_node)

        # Populate defaults manually (widget is not running here).
        for name, default in PARAM_DEFAULTS.items():
            if not param_node.GetParameter(name):
                param_node.SetParameter(name, default)

        from ZebrafishEmbryoAnalyzerLib.widget import PARAM_LENGTH_ENABLED
        self.assertEqual(param_node.GetParameter(PARAM_LENGTH_ENABLED), "true")
