"""
Subprocess entry point for zebrafish analysis inference.

Launched by InferenceController as a standalone Python process.
Protocol: reads request JSON, runs analyse_images, writes result JSON + npz files.

Exit codes:
  0 -- success
  1 -- analysis exception
  2 -- model not cached
  3 -- bad/unreadable request
  4 -- result write failure
"""

import json
import os
import sys
from pathlib import Path


def run_worker(request_path: str) -> int:
    """Execute inference from request_path. Returns exit code 0-4."""
    # --- 1. Read and validate request ---
    try:
        with open(request_path, "r", encoding="utf-8") as fh:
            req = json.load(fh)
        if req.get("protocol_version") != 1:
            return 3
        model_id = req["model_id"]
        image_paths = req["image_paths"]
        params = req["params"]
        result_json_path = req["result_json"]
        arrays_dir = req["arrays_dir"]
    except Exception:
        return 3

    # --- 2. Import logic modules ---
    try:
        from ZebrafishEmbryoAnalyzerLib.logic import preload_models, analyse_images
        from ZebrafishEmbryoAnalyzerLib.errors import ModelNotCachedError
    except Exception:
        return 3

    # --- 3. Preload models ---
    try:
        preload_params = dict(params)
        preload_params["model_id"] = model_id
        preload_models(preload_params)
    except ModelNotCachedError:
        return 2
    except Exception as exc:
        print(f"preload_models failed: {exc}", file=sys.stderr)
        return 2

    # --- 4. Run analysis ---
    n = len(image_paths)

    def _progress_cb(i, total):
        sys.stdout.write(f"PROGRESS {i}/{total}\n")
        sys.stdout.flush()

    try:
        results = analyse_images(image_paths, params, _progress_cb)
    except Exception as exc:
        _write_error(result_json_path, 1, str(exc))
        return 1

    # --- 5. Write arrays and strip from results ---
    try:
        os.makedirs(arrays_dir, exist_ok=True)
    except Exception as exc:
        _write_error(result_json_path, 4, f"Cannot create arrays dir: {exc}")
        return 4

    _ARRAY_KEYS = ("mask", "grown", "eye_mask", "path_points", "straight_line_points")

    serializable_results = []
    for i, r in enumerate(results):
        stem = _safe_stem(r.get("filename", str(i)))
        npz_path = os.path.join(arrays_dir, f"{stem}_{i}.npz")
        arrays = {}
        try:
            import numpy as np
            for k in _ARRAY_KEYS:
                v = r.get(k)
                if v is None:
                    continue
                if not isinstance(v, np.ndarray):
                    try:
                        v = np.asarray(v, dtype=float)
                    except Exception:
                        continue
                arrays[k] = v
        except Exception:
            pass

        if arrays:
            try:
                import numpy as np
                np.savez(npz_path, **arrays)
            except Exception as exc:
                _write_error(result_json_path, 4, f"Cannot write npz: {exc}")
                return 4
        else:
            npz_path = None

        sr = {
            "filename": r.get("filename"),
            "image_path": r.get("image_path"),
            "length_um": r.get("length"),
            "curvature_class": r.get("curvature"),
            "length_straight_ratio": r.get("ratio"),
            "eye_area_um2": r.get("eye_area"),
            "eye_diameter_um": r.get("eye_diameter"),
            "spacing": r.get("spacing"),
            "error": r.get("error"),
            "arrays_npz": npz_path,
        }
        serializable_results.append(sr)

    # --- 6. Write result JSON ---
    result_data = {
        "protocol_version": 1,
        "status": "ok",
        "error_code": 0,
        "error_message": "",
        "results": serializable_results,
    }
    try:
        with open(result_json_path, "w", encoding="utf-8") as fh:
            json.dump(result_data, fh)
    except Exception:
        return 4

    return 0


def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe stem from a filename."""
    return Path(filename).stem.replace(" ", "_")[:64]


def _write_error(result_json_path: str, error_code: int, message: str) -> None:
    """Best-effort write of error result JSON."""
    try:
        error_data = {
            "protocol_version": 1,
            "status": "error",
            "error_code": error_code,
            "error_message": message,
            "results": [],
        }
        with open(result_json_path, "w", encoding="utf-8") as fh:
            json.dump(error_data, fh)
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(3)
    sys.exit(run_worker(sys.argv[1]))
