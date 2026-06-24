set(MODULE_NAME ZebrafishAnalysis)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS
    ${MODULE_NAME}.py
    ZebrafishAnalysisLib/__init__.py
    ZebrafishAnalysisLib/dependency_installer.py
    ZebrafishAnalysisLib/detail_tab.py
    ZebrafishAnalysisLib/exclude_tab.py
    ZebrafishAnalysisLib/export.py
    ZebrafishAnalysisLib/gallery_tab.py
    ZebrafishAnalysisLib/logic.py
    ZebrafishAnalysisLib/overlay.py
    ZebrafishAnalysisLib/results_tab.py
    ZebrafishAnalysisLib/widget.py
    ZebrafishAnalysisLib/zoom_view.py
    ZebrafishAnalysisCore/__init__.py
    ZebrafishAnalysisCore/length.py
    ZebrafishAnalysisCore/manual.py
    ZebrafishAnalysisCore/scalebar.py
    ZebrafishAnalysisCore/seg.py
    ZebrafishAnalysisCore/seg_helper.py
  RESOURCES
    Resources/Icons/${MODULE_NAME}.png
)

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
