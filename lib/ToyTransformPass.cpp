//===- ToyTransformPass.cpp - Passes to add Transfrom OP to Payload IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {
/// This pass demos how to inersert Transform OP to the end of Payload IR..
struct ToyTransformPass
    : public PassWrapper<ToyTransformPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyTransformPass)

  StringRef getArgument() const final { return "toy-transform-pass"; }
  StringRef getDescription() const final { return "Insert a transform pass to payload IR."; }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
      registry.insert<transform::TransformDialect>();
  }

  // entry point for the pass.
  void runOnOperation() override {
    //Operation *op = getOperation();
    //Location loc = op->getLoc();
    ModuleOp moduleOp = getOperation();

    // Create an OpBuilder and set insert_point to the end of Block
    // Assume that the ModuleOp has single region and single block
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      createTransformSequence(funcOp);
    }
    // Region& region = moduleOp->getRegion(0);
    // Block& block = region.front();
    // block.setAttr("toy.test_attr", StringAttr::get(moduleOP.getContext(), "foo_bar"));
  }

  // Create a transform.SequenceOP to hold other transform ops
  void createTransformSequence(
      func::FuncOp funcop) {
    //OpBuilder builder = OpBuilder::atBlockEnd(&block);
    MLIRContext *ctx = funcop.getContext();
    Location loc = funcop.getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfter(funcop);

    // auto topLevelTransformModule = builder.create<ModuleOp>(loc);
    // Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
    // builder.setInsertionPointToStart(&topLevelTransformRegion.front());
  
    auto anyOpType = transform::AnyOpType::get(ctx);
    auto sequence = builder.create<transform::SequenceOp>(
        loc, TypeRange{}, transform::FailurePropagationMode::Propagate, anyOpType,
        [&](OpBuilder &b, Location loc, Value variantH) {
          //TODO:
          //ImplicitLocOpBuilder ib(loc, b);
          //buildStrategy(ib, variantH);
          b.create<transform::YieldOp>(loc);
        });
    (void)sequence;
  
    Region& region = funcop.getRegion();
    Block& block = region.front();
    for (Operation &op : block.getOperations()) {
      op.setAttr("toy.test_attr", StringAttr::get(ctx, "foo_bar"));
    }
  }

};
} // namespace

namespace mlir {
void registerToyTransformPass() {
  PassRegistration<ToyTransformPass>();
}
} // namespace mlir
