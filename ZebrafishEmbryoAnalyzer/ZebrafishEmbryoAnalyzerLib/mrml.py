"""
MRML adapter for ZebrafishEmbryoAnalyzer.

``results_to_rows`` is pure Python with no Slicer or VTK dependency and is
testable with standard pytest.

``get_or_create_table_node`` and ``populate_table_node`` require the Slicer
runtime.  ``populate_table_node`` imports vtk lazily inside its body so this
module is importable in plain Python test environments.
"""

import math

# Central table schema: (column_name, result_dict_key, vtk_array_type)
# vtk_array_type is "double" or "string".
# All tests, VTK column creation, and conversion use this single definition.
TABLE_SCHEMA = [
    ("Filename",            "filename",     "string"),
    ("Length_um",           "length",       "double"),
    ("CurvatureClass",      "curvature",    "string"),
    ("LengthStraightRatio", "ratio",        "double"),
    ("EyeArea_um2",         "eye_area",     "double"),
    ("EyeDiameter_um",      "eye_diameter", "double"),
    ("Error",               "error",        "string"),
]

ROLE_RESULTS_TABLE = "ResultsTable"
ROLE_CURRENT_IMAGE = "CurrentImage"
ROLE_CURRENT_SEGMENTATION = "CurrentSegmentation"

# String columns whose values are always preserved verbatim, even on error rows.
_PRESERVE_ON_ERROR = frozenset({"filename", "error"})


def results_to_rows(results):
    """Convert analysis result dicts to row dicts for the MRML table.

    Pure Python — no vtk or slicer imports. Testable with standard pytest.
    Input dicts are not mutated.

    Conversion rules (applied per column):
    - error row (non-empty "error" key): numeric → NaN, CurvatureClass → "",
      Filename and Error are always preserved
    - numeric field, value is None → math.nan
    - numeric field, value present → float(value)
    - string field, value is None  → ""
    - string field, value present  → str(value)

    Parameters
    ----------
    results : list[dict]
        List of result dicts from analyse_images().

    Returns
    -------
    list[dict]
        One dict per result, keyed by TABLE_SCHEMA column names, in schema order.
    """
    rows = []
    for r in results:
        has_error = bool(r.get("error"))
        row = {}
        for col_name, key, vtk_type in TABLE_SCHEMA:
            val = r.get(key)
            if vtk_type == "double":
                row[col_name] = math.nan if (has_error or val is None) else float(val)
            else:
                if key in _PRESERVE_ON_ERROR or not has_error:
                    row[col_name] = str(val) if val is not None else ""
                else:
                    row[col_name] = ""
        rows.append(row)
    return rows


def get_or_create_table_node(param_node, scene):
    """Return the existing ResultsTable node or create exactly one new node.

    Looks up the node by the stored node-reference role, not by display name.
    If no valid reference exists (missing or wrong node type), creates a new
    vtkMRMLTableNode, sets its initial display name, and registers its ID on
    the parameter node.  A wrong-type foreign node is left in the scene
    unchanged.

    Parameters
    ----------
    param_node : vtkMRMLScriptedModuleNode
        The module parameter node that owns the ResultsTable reference.
    scene : vtkMRMLScene
        The active MRML scene.

    Returns
    -------
    vtkMRMLTableNode
    """
    existing = param_node.GetNodeReference(ROLE_RESULTS_TABLE)
    if existing is not None and existing.IsA("vtkMRMLTableNode"):
        return existing

    node = scene.AddNewNodeByClass("vtkMRMLTableNode")
    node.SetName("ZebrafishEmbryoAnalyzer Results")
    param_node.SetNodeReferenceID(ROLE_RESULTS_TABLE, node.GetID())
    return node


def build_vtk_table(rows):
    """Build a complete vtkTable from conversion rows. No MRML side effects.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().

    Returns
    -------
    vtk.vtkTable
    """
    import vtk  # lazy — not available in plain Python test environments

    n = len(rows)
    table = vtk.vtkTable()

    for col_name, _, vtk_type in TABLE_SCHEMA:
        if vtk_type == "double":
            arr = vtk.vtkDoubleArray()
        else:
            arr = vtk.vtkStringArray()
        arr.SetName(col_name)
        arr.SetNumberOfTuples(n)
        for i, row in enumerate(rows):
            arr.SetValue(i, row[col_name])
        table.AddColumn(arr)

    return table


def populate_table_node(rows, node):
    """Replace node content atomically with data from results_to_rows().

    Builds a complete vtk.vtkTable in memory first; applies it to the node
    only after all columns and values have been set successfully.  If
    construction fails the existing node content is preserved unchanged.

    Parameters
    ----------
    rows : list[dict]
        Output of results_to_rows().
    node : vtkMRMLTableNode
        Target node to update.
    """
    table = build_vtk_table(rows)
    node.SetAndObserveTable(table)


def image_geometry(h_orig: int, w_orig: int, um_per_px: float):
    """Return (dims, spacing, origin) for a vtkMRMLVectorVolumeNode.

    dims    = (w_orig, h_orig, 1)              VTK IJK order
    spacing = (um_per_px/1000, um_per_px/1000, 1.0)  mm, isotropic
    origin  = (0.0, 0.0, 0.0)

    Pure Python — no vtk or slicer imports. Testable with standard pytest.

    Raises ValueError for non-positive h_orig, w_orig, or um_per_px
    (including zero, negative, NaN, and inf).
    """
    if not isinstance(h_orig, int) or h_orig <= 0:
        raise ValueError(f"h_orig must be a positive integer, got {h_orig!r}")
    if not isinstance(w_orig, int) or w_orig <= 0:
        raise ValueError(f"w_orig must be a positive integer, got {w_orig!r}")
    if not math.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError(f"um_per_px must be finite and positive, got {um_per_px!r}")
    spacing_mm = um_per_px / 1000.0
    dims = (w_orig, h_orig, 1)
    spacing = (spacing_mm, spacing_mm, 1.0)
    origin = (0.0, 0.0, 0.0)
    return dims, spacing, origin


def get_or_create_image_node(param_node, scene):
    """Return the existing CurrentImage node or create exactly one new node.

    Looks up by reference role ROLE_CURRENT_IMAGE (not display name).
    Creates a new vtkMRMLVectorVolumeNode named "ZebrafishEmbryoAnalyzer Current Image"
    if no valid reference exists. Stores new node ID in param_node.
    A wrong-type foreign node is left in scene unchanged; a new node is created.
    """
    existing = param_node.GetNodeReference(ROLE_CURRENT_IMAGE)
    if existing is not None and existing.IsA("vtkMRMLVectorVolumeNode"):
        return existing
    node = scene.AddNewNodeByClass(
        "vtkMRMLVectorVolumeNode", "ZebrafishEmbryoAnalyzer Current Image"
    )
    param_node.SetNodeReferenceID(ROLE_CURRENT_IMAGE, node.GetID())
    return node


def update_image_node(image_rgb, um_per_px, node):
    """Write a uint8 RGB array into an existing vtkMRMLVectorVolumeNode.

    image_rgb must be uint8, shape (H, W, 3).
    um_per_px is the original-image physical scale in micrometers per pixel.
    result["spacing"] must NOT be used here (it is calibrated to 256x256 mask space).

    NOTE: _on_detect_scale / show_raw_image is out of scope for E2b.
    The MRML node intentionally reflects the last gallery selection, not the
    scalebar debug overlay.

    VTK step order:
      1. derive h_orig, w_orig from image_rgb.shape
      2. compute geometry via image_geometry()
      3. flipud + fliplr + copy (corrects VTK bottom-left origin and Slicer radiological convention)
      4. reshape and convert to VTK array (no AllocateScalars)
      5. build vtkImageData: SetDimensions, GetPointData().SetScalars()
      6. reset direction cosines to identity
      7. set spacing and origin on node (before SetAndObserveImageData)
      8. SetAndObserveImageData as final step
    """
    import vtk
    from vtk.util import numpy_support
    import numpy as np

    h_orig, w_orig = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"image_rgb must have shape (H, W, 3), got {image_rgb.shape}"
        )
    dims, spacing, origin = image_geometry(h_orig, w_orig, um_per_px)

    # flipud: numpy row 0 (image top) → VTK last row (visual top)
    # fliplr: compensates for Slicer's radiological convention (R axis → left of screen)
    # .copy() restores C-contiguity after the non-contiguous views
    flipped = np.flipud(np.fliplr(image_rgb)).copy()
    flat = flipped.reshape(-1, 3)

    vtk_array = numpy_support.numpy_to_vtk(
        flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_array.SetNumberOfComponents(3)
    vtk_array.SetName("ImageScalars")

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.GetPointData().SetScalars(vtk_array)  # no AllocateScalars

    identity = vtk.vtkMatrix4x4()
    node.SetIJKToRASDirectionMatrix(identity)
    node.SetSpacing(*spacing)
    node.SetOrigin(*origin)
    node.SetAndObserveImageData(image_data)  # final step — fires observers with complete geometry


def resample_mask_to_original(mask_2d, h_orig, w_orig):
    """Resample a 2-D mask to (h_orig, w_orig) with nearest-neighbour interpolation.

    Pure Python (cv2 + numpy only) — no vtk or slicer. Testable with standard pytest.

    Parameters
    ----------
    mask_2d : ndarray
        2-D array of any dtype. Values > 0 are treated as body / eye pixels.
    h_orig : int
        Target height in pixels.
    w_orig : int
        Target width in pixels.

    Returns
    -------
    ndarray
        uint8 array of shape (h_orig, w_orig) with values 0 or 1.
    """
    import cv2
    import numpy as np

    binary = (mask_2d > 0).astype(np.uint8)
    # cv2.resize expects (width, height)
    return cv2.resize(binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)


def get_or_create_segmentation_node(param_node, scene):
    """Return the existing CurrentSegmentation node or create exactly one new node.

    Looks up by reference role ROLE_CURRENT_SEGMENTATION (not display name).
    Creates a new vtkMRMLSegmentationNode named
    "ZebrafishEmbryoAnalyzer Current Segmentation" if no valid reference exists.
    Stores new node ID in param_node.
    A wrong-type foreign node is left in scene unchanged; a new node is created.
    """
    existing = param_node.GetNodeReference(ROLE_CURRENT_SEGMENTATION)
    if existing is not None and existing.IsA("vtkMRMLSegmentationNode"):
        return existing
    node = scene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", "ZebrafishEmbryoAnalyzer Current Segmentation"
    )
    node.CreateDefaultDisplayNodes()
    param_node.SetNodeReferenceID(ROLE_CURRENT_SEGMENTATION, node.GetID())
    return node


def update_segmentation_node(result, um_per_px, node, image_node=None):
    """Write body and eye masks from a result dict into an existing vtkMRMLSegmentationNode.

    result["original"]: uint8 ndarray shape (H_orig, W_orig, 3).
    result["mask"]: 2-D ndarray shape (256, 256) — body mask (>0 means body).
    result["eye_mask"]: 2-D bool ndarray shape (256, 256) or None — eye mask.
    um_per_px: physical scale of the original image in micrometres per pixel.
    image_node: optional vtkMRMLVectorVolumeNode — used to set reference geometry
        so Slicer can position the segmentation in slice views.

    VTK step order:
      1. Lazy-import vtk, numpy, slicer, vtkSegmentationCore inside function.
      2. Derive h_orig, w_orig from result["original"].shape; skip gracefully if absent.
      3. Resample body mask and eye mask (if applicable) to (h_orig, w_orig).
      4. Compute geometry via image_geometry(h_orig, w_orig, um_per_px).
      5. Build vtkOrientedImageData for each segment (uint8, values 0/1).
         - flipud + fliplr + ascontiguousarray to match VTK coordinate convention.
      6. Set master representation to binary labelmap.
      7. Wrap full modification in StartModify/EndModify to suppress intermediate events.
      8. node.GetSegmentation().RemoveAllSegments()
      9. Add "Body" segment (green) — always.
      10. Add "Eye" segment (red) — only when eye_mask is not None and eye_mask.any().
      11. Populate each segment via SetBinaryLabelmapToSegment.
      12. Set reference image geometry from image_node if provided.
    """
    import vtk
    from vtk.util import numpy_support
    import numpy as np
    import slicer
    import vtkSegmentationCore

    original = result.get("original") if result else None
    if original is None:
        return

    h_orig, w_orig = int(original.shape[0]), int(original.shape[1])
    dims, spacing, origin = image_geometry(h_orig, w_orig, um_per_px)
    spacing_mm = spacing[0]

    mask_2d = result.get("mask")
    eye_mask_2d = result.get("eye_mask")

    body_2d = resample_mask_to_original(mask_2d, h_orig, w_orig) if mask_2d is not None else None
    has_eye = (
        eye_mask_2d is not None
        and hasattr(eye_mask_2d, "any")
        and eye_mask_2d.any()
    )
    eye_2d = resample_mask_to_original(eye_mask_2d, h_orig, w_orig) if has_eye else None

    def _make_oriented_image(arr_2d):
        """Build a vtkOrientedImageData from a 2-D uint8 (0/1) array."""
        flipped = np.ascontiguousarray(np.flipud(np.fliplr(arr_2d)))
        flat = flipped.reshape(-1)
        vtk_array = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_array.SetNumberOfComponents(1)
        oid = vtkSegmentationCore.vtkOrientedImageData()
        oid.SetDimensions(w_orig, h_orig, 1)
        oid.GetPointData().SetScalars(vtk_array)
        oid.SetSpacing(spacing_mm, spacing_mm, 1.0)
        oid.SetOrigin(0.0, 0.0, 0.0)
        return oid

    node.GetSegmentation().SetSourceRepresentationName(
        slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    )

    was_modifying = node.StartModify()
    try:
        seg = node.GetSegmentation()
        seg.RemoveAllSegments()

        if body_2d is not None:
            body_id = seg.AddEmptySegment("Body", "Body", [0.0, 1.0, 0.0])
            slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                _make_oriented_image(body_2d), node, body_id
            )

        if eye_2d is not None:
            eye_id = seg.AddEmptySegment("Eye", "Eye", [1.0, 0.0, 0.0])
            slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
                _make_oriented_image(eye_2d), node, eye_id
            )

        if image_node is not None:
            node.SetReferenceImageGeometryParameterFromVolumeNode(image_node)
    finally:
        node.EndModify(was_modifying)
