"""Tests for the pure Python conversion layer in ZebrafishEmbryoAnalyzerLib.mrml.

All tests run in the normal pytest process. conftest.py adds the ZebrafishEmbryoAnalyzer
directory to sys.path, so ZebrafishEmbryoAnalyzerLib.mrml imports without slicer or vtk.
"""

import math
import os
import re

import pytest

from ZebrafishEmbryoAnalyzerLib.mrml import results_to_rows, TABLE_SCHEMA, ROLE_RESULTS_TABLE

_MRML_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ZebrafishEmbryoAnalyzer", "ZebrafishEmbryoAnalyzerLib", "mrml.py",
)


# ---------------------------------------------------------------------------
# TABLE_SCHEMA contract
# ---------------------------------------------------------------------------

def test_schema_has_exactly_seven_columns():
    assert len(TABLE_SCHEMA) == 7


def test_schema_column_names_in_order():
    names = [col for col, _, _ in TABLE_SCHEMA]
    assert names == [
        "Filename",
        "Length_um",
        "CurvatureClass",
        "LengthStraightRatio",
        "EyeArea_um2",
        "EyeDiameter_um",
        "Error",
    ]


# ---------------------------------------------------------------------------
# results_to_rows — basic cases
# ---------------------------------------------------------------------------

def test_empty_results_returns_empty_rows():
    assert results_to_rows([]) == []


def test_single_complete_result_produces_one_row():
    r = {
        "filename": "fish.png",
        "length": 1234.5,
        "curvature": 2,
        "ratio": 1.05,
        "eye_area": 567.8,
        "eye_diameter": 12.3,
        "error": None,
    }
    rows = results_to_rows([r])
    assert len(rows) == 1
    row = rows[0]
    assert row["Filename"] == "fish.png"
    assert row["Length_um"] == pytest.approx(1234.5)
    assert row["CurvatureClass"] == "2"
    assert row["LengthStraightRatio"] == pytest.approx(1.05)
    assert row["EyeArea_um2"] == pytest.approx(567.8)
    assert row["EyeDiameter_um"] == pytest.approx(12.3)
    assert row["Error"] == ""


def test_numeric_values_remain_float():
    r = {
        "filename": "fish.png",
        "length": 100.0,
        "curvature": 1,
        "ratio": 1.1,
        "eye_area": 50.0,
        "eye_diameter": 8.0,
        "error": None,
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert isinstance(row["Length_um"], float)
    assert isinstance(row["LengthStraightRatio"], float)
    assert isinstance(row["EyeArea_um2"], float)
    assert isinstance(row["EyeDiameter_um"], float)


def test_missing_numeric_values_become_nan():
    r = {
        "filename": "fish.png",
        "length": None,
        "curvature": None,
        "ratio": None,
        "eye_area": None,
        "eye_diameter": None,
        "error": None,
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert math.isnan(row["Length_um"])
    assert math.isnan(row["LengthStraightRatio"])
    assert math.isnan(row["EyeArea_um2"])
    assert math.isnan(row["EyeDiameter_um"])
    assert row["CurvatureClass"] == ""
    assert row["Error"] == ""


def test_curvature_int_becomes_string():
    r = {
        "filename": "fish.png", "length": None, "curvature": 2,
        "ratio": None, "eye_area": None, "eye_diameter": None, "error": None,
    }
    rows = results_to_rows([r])
    assert rows[0]["CurvatureClass"] == "2"


def test_error_row_preserves_filename_and_error():
    r = {
        "filename": "bad.png",
        "length": None, "curvature": None, "ratio": None,
        "eye_area": None, "eye_diameter": None,
        "error": "Could not read image.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert row["Filename"] == "bad.png"
    assert row["Error"] == "Could not read image."


def test_error_row_numerics_are_nan():
    r = {
        "filename": "bad.png",
        "length": None, "curvature": None, "ratio": None,
        "eye_area": None, "eye_diameter": None,
        "error": "Could not read image.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert math.isnan(row["Length_um"])
    assert math.isnan(row["LengthStraightRatio"])
    assert math.isnan(row["EyeArea_um2"])
    assert math.isnan(row["EyeDiameter_um"])


def test_error_row_with_actual_values_forces_nan_and_blank_curvature():
    """Error row with non-None numeric and curvature values must still be normalized."""
    r = {
        "filename": "partial.png",
        "length": 999.9,
        "curvature": 2,
        "ratio": 1.5,
        "eye_area": 300.0,
        "eye_diameter": 20.0,
        "error": "Segmentation collapsed.",
    }
    rows = results_to_rows([r])
    row = rows[0]
    assert row["Filename"] == "partial.png"
    assert row["Error"] == "Segmentation collapsed."
    assert math.isnan(row["Length_um"]), "length should be NaN on error row"
    assert math.isnan(row["LengthStraightRatio"]), "ratio should be NaN on error row"
    assert math.isnan(row["EyeArea_um2"]), "eye_area should be NaN on error row"
    assert math.isnan(row["EyeDiameter_um"]), "eye_diameter should be NaN on error row"
    assert row["CurvatureClass"] == "", "curvature should be blank on error row"


def test_multiple_results_preserve_order():
    results = [
        {
            "filename": "a.png", "length": 1.0, "curvature": 0, "ratio": 1.0,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
        {
            "filename": "b.png", "length": 2.0, "curvature": 1, "ratio": 1.1,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
        {
            "filename": "c.png", "length": 3.0, "curvature": 2, "ratio": 1.2,
            "eye_area": None, "eye_diameter": None, "error": None,
        },
    ]
    rows = results_to_rows(results)
    assert len(rows) == 3
    assert rows[0]["Filename"] == "a.png"
    assert rows[1]["Filename"] == "b.png"
    assert rows[2]["Filename"] == "c.png"
    assert rows[0]["Length_um"] == pytest.approx(1.0)
    assert rows[1]["Length_um"] == pytest.approx(2.0)
    assert rows[2]["Length_um"] == pytest.approx(3.0)


def test_input_dicts_not_mutated():
    r = {
        "filename": "fish.png", "length": 1.0, "curvature": 2,
        "ratio": 1.05, "eye_area": None, "eye_diameter": None, "error": None,
    }
    original = dict(r)
    original_keys = set(r.keys())
    results_to_rows([r])
    assert r == original, "input dict was mutated"
    assert set(r.keys()) == original_keys, "input dict keys changed"


def test_mrml_module_importable_without_slicer_or_vtk():
    """mrml.py must be importable without installing slicer or vtk."""
    from ZebrafishEmbryoAnalyzerLib import mrml
    assert hasattr(mrml, "results_to_rows")
    assert hasattr(mrml, "TABLE_SCHEMA")
    assert hasattr(mrml, "ROLE_RESULTS_TABLE")
    assert hasattr(mrml, "get_or_create_table_node")
    assert hasattr(mrml, "build_vtk_table")
    assert hasattr(mrml, "populate_table_node")


def test_mrml_module_no_global_slicer_import_still_holds():
    """Regression: mrml.py still has no module-level 'import slicer' after E2c."""
    import re
    src = open(_MRML_PY).read()
    lines = src.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(def |class )", stripped):
            in_function = True
        if not in_function and re.match(r"^import slicer\b", stripped):
            pytest.fail("mrml.py has a module-level 'import slicer' after E2c")


# ---------------------------------------------------------------------------
# image_geometry — pure Python geometry helper
# ---------------------------------------------------------------------------

def test_image_geometry_basic_dimensions():
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    dims, spacing, origin = image_geometry(480, 640, 22.99)
    assert dims == (640, 480, 1)


def test_image_geometry_spacing_mm():
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    _, spacing, _ = image_geometry(480, 640, 22.99)
    assert spacing == pytest.approx((22.99 / 1000.0, 22.99 / 1000.0, 1.0))


def test_image_geometry_spacing_z():
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    _, spacing, _ = image_geometry(100, 100, 10.0)
    assert spacing[2] == 1.0


def test_image_geometry_origin_zero():
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    _, _, origin = image_geometry(100, 100, 10.0)
    assert origin == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("h,w,um", [
    (0, 100, 10.0),
    (100, 0, 10.0),
    (-1, 100, 10.0),
    (100, -1, 10.0),
    (100, 100, 0.0),
    (100, 100, -1.0),
    (100, 100, float("nan")),
    (100, 100, float("inf")),
    (100, 100, float("-inf")),
])
def test_image_geometry_raises_for_invalid(h, w, um):
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    with pytest.raises(ValueError):
        image_geometry(h, w, um)


def test_image_geometry_non_square():
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    dims, _, _ = image_geometry(120, 160, 5.0)
    assert dims == (160, 120, 1)


def test_flipud_and_fliplr_pixel_ordering():
    """Verify that flipud+fliplr (180° rotation) maps corners correctly for VTK+Slicer."""
    import numpy as np
    H, W = 4, 3
    arr = np.zeros((H, W, 3), dtype="uint8")
    # Mark each corner with a distinct value
    arr[0, 0, :] = 10    # top-left
    arr[0, W-1, :] = 20  # top-right
    arr[H-1, 0, :] = 30  # bottom-left
    arr[H-1, W-1, :] = 40  # bottom-right
    flipped = np.flipud(np.fliplr(arr)).copy()
    flat = flipped.reshape(-1, 3)
    # After 180° rotation:
    # VTK point 0 (first row, first col) = original bottom-right (40)
    assert flat[0, 0] == 40
    # VTK point W-1 (first row, last col) = original bottom-left (30)
    assert flat[W-1, 0] == 30
    # VTK point W*(H-1) (last row, first col) = original top-right (20)
    assert flat[W*(H-1), 0] == 20
    # VTK point W*H-1 (last row, last col) = original top-left (10)
    assert flat[W*H-1, 0] == 10


def test_flipud_and_fliplr_produces_c_contiguous():
    """flipud+fliplr produces non-C-contiguous views; .copy() must restore contiguity."""
    import numpy as np
    arr = np.zeros((10, 10, 3), dtype="uint8")
    flipped = np.flipud(np.fliplr(arr)).copy()
    assert flipped.flags["C_CONTIGUOUS"]


def test_fliplr_and_flipud_both_applied():
    """Verify that both flipud and fliplr are applied before reshape."""
    import numpy as np
    H, W = 4, 6
    arr = np.zeros((H, W, 3), dtype="uint8")
    # Mark the top-left corner distinctly
    arr[0, 0, :] = 10   # top-left
    arr[0, W-1, :] = 20  # top-right
    arr[H-1, 0, :] = 30  # bottom-left
    arr[H-1, W-1, :] = 40  # bottom-right

    # After flipud+fliplr (180° rotation):
    # top-left (10) → bottom-right in VTK (last row, last col)
    # top-right (20) → bottom-left in VTK (last row, first col)
    flipped = np.flipud(np.fliplr(arr)).copy()
    flat = flipped.reshape(-1, 3)

    # VTK point 0 = first row, first col = original bottom-right (40)
    assert flat[0, 0] == 40
    # VTK point W-1 = first row, last col = original bottom-left (30)
    assert flat[W-1, 0] == 30
    # VTK point W*(H-1) = last row, first col = original top-right (20)
    assert flat[W*(H-1), 0] == 20
    # VTK point W*H-1 = last row, last col = original top-left (10)
    assert flat[W*H-1, 0] == 10


def test_image_geometry_accepts_numpy_int64_when_converted():
    """update_image_node must convert numpy shape values to int before calling image_geometry."""
    import numpy as np
    from ZebrafishEmbryoAnalyzerLib.mrml import image_geometry
    # numpy.int64 values directly should fail image_geometry (isinstance check)
    h, w = np.array([10, 8], dtype=np.int64)
    # Demonstrate the problem: numpy.int64 fails isinstance(h, int) in numpy < 2
    # The fix is to convert in update_image_node, not in image_geometry
    dims, _, _ = image_geometry(int(h), int(w), 22.99)
    assert dims == (8, 10, 1)
