set(MODULE_NAME ZebrafishEmbryoAnalyzer)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS
    ${MODULE_NAME}.py
    ZebrafishEmbryoAnalyzerLib/__init__.py
    ZebrafishEmbryoAnalyzerLib/dependency_installer.py
    ZebrafishEmbryoAnalyzerLib/detail_tab.py
    ZebrafishEmbryoAnalyzerLib/errors.py
    ZebrafishEmbryoAnalyzerLib/inference_runner.py
    ZebrafishEmbryoAnalyzerLib/inference_worker.py
    ZebrafishEmbryoAnalyzerLib/model_downloader.py
    ZebrafishEmbryoAnalyzerLib/model_manifest.py
    ZebrafishEmbryoAnalyzerLib/mrml.py
    ZebrafishEmbryoAnalyzerLib/export.py
    ZebrafishEmbryoAnalyzerLib/gallery_tab.py
    ZebrafishEmbryoAnalyzerLib/logic.py
    ZebrafishEmbryoAnalyzerLib/overlay.py
    ZebrafishEmbryoAnalyzerLib/results_tab.py
    ZebrafishEmbryoAnalyzerLib/widget.py
    ZebrafishEmbryoAnalyzerLib/zoom_view.py
    ZebrafishEmbryoAnalyzerCore/__init__.py
    ZebrafishEmbryoAnalyzerCore/length.py
    ZebrafishEmbryoAnalyzerCore/manual.py
    ZebrafishEmbryoAnalyzerCore/scalebar.py
    ZebrafishEmbryoAnalyzerCore/seg.py
    ZebrafishEmbryoAnalyzerCore/seg_helper.py
  RESOURCES
    Resources/Icons/${MODULE_NAME}.png
)

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
