set(KIT_PYTHON_SCRIPTS
  Python/ZebrafishEmbryoAnalyzerTest.py
)

set(KIT_PYTHON_RESOURCES)

# The script path must be absolute. slicer_add_python_unittest derives the directory it
# puts on sys.path from SCRIPT, and only falls back to CMAKE_CURRENT_SOURCE_DIR when that
# directory comes out empty. A relative "Python/..." yields the bare relative "Python",
# which is not empty, so the fallback is skipped and the test module cannot be imported.
foreach(script ${KIT_PYTHON_SCRIPTS})
  slicer_add_python_unittest(
    SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/${script}
    SLICER_ARGS --disable-cli-modules
  )
endforeach()
