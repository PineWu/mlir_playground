add_mlir_library(
  # Library called MyExtension.
  TestExtension

  # Built from the following source files.
  MyExtension.cpp
  ToyTransformPass.cpp
  TestAffineMapPass.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionCh2IncGen
  AIRRtOpsIncGen
  MLIREmitCConversionPassIncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PUBLIC
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRPass
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRTestDialect
)

add_subdirectory(Conversion/LinalgToEmitC)
