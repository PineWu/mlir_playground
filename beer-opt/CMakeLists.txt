# For a better top-level template to copy, see examples/standalone.

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        )

#add_subdirectory(include)
#add_subdirectory(lib)

#add_llvm_example(transform-opt-ch2
#  transform-opt/transform-opt.cpp)
add_llvm_executable(beer-opt beer-opt.cpp)
llvm_update_compile_flags(beer-opt)

target_link_libraries(beer-opt
  PRIVATE
  ${LIBS}
  MLIRIR
  MLIRMlirOptMain
  MLIRSideEffectInterfaces
  TestExtension
  TestPrintNesting
  MLIRLinalgToEmitC
)
