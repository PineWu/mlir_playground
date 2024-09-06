add_mlir_library(
  TestPrintNesting

  # Built from the following source files.
  TestPrintNesting.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include


  LINK_LIBS PUBLIC
  MLIRPass
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRTestDialect
)
