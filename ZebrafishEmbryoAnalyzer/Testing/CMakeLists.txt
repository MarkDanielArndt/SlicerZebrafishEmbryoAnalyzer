set(KIT_PYTHON_SCRIPTS
  Python/ZebrafishEmbryoAnalyzerTest.py
)

set(KIT_PYTHON_RESOURCES)

foreach(script ${KIT_PYTHON_SCRIPTS})
  slicer_add_python_unittest(
    SCRIPT ${script}
    SLICER_ARGS --disable-cli-modules
  )
endforeach()
