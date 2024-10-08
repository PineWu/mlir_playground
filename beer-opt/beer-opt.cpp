//===-- beer-opt.cpp - Out-of-tree MLIR play ground entry point --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top-level file for the Transform dialect tutorial chapter 2.
//
//===----------------------------------------------------------------------===//

#include "MyExtension.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include <cstdlib>

// Forward declarations of test passes that used in this chapter for
// illustrative purposes. Test passes are not directly exposed for use in
// binaries other than mlir-opt, which is too big to serve as an example.
namespace mlir {
void registerTestPrintNestingPass();
void registerToyTransformPass();
void registerLinalgToEmitCPass();
void registerTestAffineMapPass();
} // namespace mlir

namespace mlir::test {
void registerTestTransformDialectEraseSchedulePass();
void registerTestTransformDialectInterpreterPass();
} // namespace mlir::test

namespace test {
void registerTestTransformDialectExtension(mlir::DialectRegistry &);
} // namespace test

int main(int argc, char **argv) {
  // Register all "core" dialects and our transform dialect extension.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registerMyExtension(registry);
  registry.insert<mlir::transform::TransformDialect>();

  // Register a handful of cleanup passes that we can run to make the output IR
  // look nicer.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  // Register transform interpreter pass.
  mlir::transform::registerInterpreterPass();

  // Register the test passes.
#ifdef MLIR_INCLUDE_TESTS
  mlir::test::registerTestTransformDialectEraseSchedulePass();
  // mlir::test::registerTestTransformDialectInterpreterPass();
  test::registerTestTransformDialectExtension(registry);
#else
  llvm::errs() << "warning: MLIR built without test passes, interpreter "
                  "testing will not be available\n";
#endif // MLIR_INCLUDE_TESTS

  // Register the printing nesting IR pass
  mlir::registerTestPrintNestingPass();
  mlir::registerToyTransformPass();
  mlir::registerLinalgToEmitCPass();
  mlir::registerTestAffineMapPass();

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "beer-opt", registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
