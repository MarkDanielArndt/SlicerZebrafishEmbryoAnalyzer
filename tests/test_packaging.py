"""Verify that every .py file under ZebrafishEmbryoAnalyzer/ is listed in the
CMakeLists.txt SCRIPTS section and that every SCRIPTS entry exists on disk.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CMAKE_FILE = ROOT / "ZebrafishEmbryoAnalyzer" / "CMakeLists.txt"
ZEBRA_DIR = ROOT / "ZebrafishEmbryoAnalyzer"

_EXCLUDE_DIRS = {"__pycache__", "Testing"}


def _expand_cmake_vars(text: str, cmake_text: str) -> str:
    """Expand ${VAR} references using set(VAR value) definitions in cmake_text."""
    def replacer(m):
        var_name = m.group(1)
        definition = re.search(
            rf"set\(\s*{re.escape(var_name)}\s+([^\)]+)\)", cmake_text
        )
        if definition:
            return definition.group(1).strip()
        return m.group(0)  # leave unexpanded if not found

    # Expand up to 5 passes to handle nested vars
    for _ in range(5):
        expanded = re.sub(r"\$\{([^}]+)\}", replacer, text)
        if expanded == text:
            break
        text = expanded
    return text


def _parse_scripts_from_cmake(cmake_text: str) -> set:
    """Return the set of .py filenames listed under SCRIPTS in cmake_text.

    Paths are relative to ZEBRA_DIR (i.e. relative to CMakeLists.txt location).
    """
    # Rely only on RESOURCES as the block terminator (avoids matching a bare
    # closing parenthesis mid-block on some CMake layouts).
    m = re.search(r"SCRIPTS\s+(.*?)RESOURCES", cmake_text, re.DOTALL)
    if m is None:
        return set()
    block = _expand_cmake_vars(m.group(1), cmake_text)
    entries = set()
    for token in block.split():
        if token.endswith(".py"):
            entries.add(token)
    return entries


def _filesystem_py_files() -> set:
    """Return paths of all .py files under ZEBRA_DIR, relative to ZEBRA_DIR,
    excluding __pycache__ and Testing directories.
    """
    result = set()
    for path in ZEBRA_DIR.rglob("*.py"):
        # Skip excluded directories anywhere in the path
        if any(part in _EXCLUDE_DIRS for part in path.parts):
            continue
        result.add(path.relative_to(ZEBRA_DIR).as_posix())
    return result


def test_cmake_scripts_match_filesystem():
    cmake_text = CMAKE_FILE.read_text(encoding="utf-8")
    listed = _parse_scripts_from_cmake(cmake_text)
    on_disk = _filesystem_py_files()

    missing_from_cmake = on_disk - listed
    missing_from_disk = listed - on_disk

    assert not missing_from_cmake, (
        f"Files on disk but NOT listed in CMakeLists.txt SCRIPTS:\n"
        + "\n".join(f"  {f}" for f in sorted(missing_from_cmake))
    )
    assert not missing_from_disk, (
        f"Files listed in CMakeLists.txt SCRIPTS but NOT found on disk:\n"
        + "\n".join(f"  {f}" for f in sorted(missing_from_disk))
    )
