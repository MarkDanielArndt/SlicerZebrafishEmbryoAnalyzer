cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ZebrafishEmbryoAnalyzer)

set(EXTENSION_HOMEPAGE "https://github.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer")
set(EXTENSION_CATEGORY "Analysis")
set(EXTENSION_CONTRIBUTORS "Jona Richter, Mark Daniel Arndt")
set(EXTENSION_DESCRIPTION "Zebrafish body length, curvature, and eye metrics from 2D microscopy images.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/master/ZebrafishEmbryoAnalyzer/Resources/Icons/ZebrafishEmbryoAnalyzer.png")
set(EXTENSION_SCREENSHOTURLS
  "https://raw.githubusercontent.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/master/Documentation_images/gallery.png"
  "https://raw.githubusercontent.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/master/Documentation_images/detail.png"
  "https://raw.githubusercontent.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/master/Documentation_images/setup_dialog.png"
  "https://raw.githubusercontent.com/MarkDanielArndt/SlicerZebrafishEmbryoAnalyzer/master/Documentation_images/manual_correction.png"
)
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(ZebrafishEmbryoAnalyzer)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
