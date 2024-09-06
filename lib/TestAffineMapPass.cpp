//===- TestAffineMapPass.cpp - Passes to Test AffineMap and TilingInterface  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
struct TestAffineMapPass : public PassWrapper<TestAffineMapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAffineMapPass)

  StringRef getArgument() const final { return "test-affinemap-tilinginterface"; }
  StringRef getDescription() const final { return "Test AffineMap and TilingInterface."; }
  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    testAffineMap(op);
  }

  void testAffineMap(Operation *op) {
    MLIRContext *ctx = op->getContext();
    ModuleOp module = dyn_cast<ModuleOp>(op);
    auto builder = OpBuilder(ctx);

    AffineMap map1 = AffineMap::get(ctx);
    llvm::outs() << "AffineMap 1: " << map1 << "\n";

    AffineMap map2 = AffineMap::getMinorIdentityMap(6, 3, ctx);
    llvm::outs() << "AffineMap 2: " << map2 << "\n";

    AffineMap map3 = AffineMap::getMultiDimIdentityMap(4, ctx);
    llvm::outs() << "AffineMap 3: " << map3 << "\n";

    AffineMap map4 = AffineMap::get(4, 3, ctx);
    llvm::outs() << "AffineMap 4: " << map4 << "\n";

    AffineMap map5 = AffineMap::get(4, 3, getAffineDimExpr(2, ctx));
    llvm::outs() << "AffineMap 5: " << map5 << "\n";

    llvm::ArrayRef<AffineExpr> exprs = {getAffineDimExpr(0, ctx), getAffineDimExpr(3, ctx)};
    AffineMap map6 = AffineMap::get(4, 3, exprs, ctx);
    llvm::outs() << "AffineMap 6: " << map6 << "\n";

    llvm::ArrayRef<AffineExpr> symbols = {getAffineSymbolExpr(0, ctx), getAffineSymbolExpr(2, ctx)};
    AffineMap map7 = AffineMap::get(4, 3, symbols, ctx);
    llvm::outs() << "AffineMap 7: " << map7 << "\n";

    AffineExpr bianryExpr =
        getAffineBinaryOpExpr(AffineExprKind::Add, getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx));
    AffineMap map8 = AffineMap::get(4, 3, bianryExpr, ctx);
    llvm::outs() << "AffineMap 8: " << map8 << "\n";

    AffineMap map9 = AffineMap::getMultiDimIdentityMap(4, ctx);
    llvm::outs() << "AffineMap 9: " << map9 << "\n";

    llvm::ArrayRef<AffineExpr> exprs1 = {
        getAffineDimExpr(0, ctx),
        getAffineBinaryOpExpr(AffineExprKind::Add, getAffineDimExpr(1, ctx),
                              getAffineConstantExpr(9, ctx))};
    AffineMap mapMin = AffineMap::get(2, 0, exprs1, ctx);

    auto foldMin = affine::makeComposedFoldedAffineMin(
        builder, op->getLoc(), mapMin,
        {OpFoldResult(builder.getIndexAttr(333)),
         OpFoldResult(builder.getIndexAttr(213))});
    llvm::outs() << "AffineMap Min: " << mapMin
                 << " , folded Min resulst: " << foldMin << "\n";
    // create a affine.min OP and fold it
    Value cst1 =
        builder.create<arith::ConstantIndexOp>(op->getLoc(), 333).getResult();
    Value cst2 =
        builder.create<arith::ConstantIndexOp>(op->getLoc(), 213).getResult();
    affine::AffineMinOp minOp = builder.create<affine::AffineMinOp>(
        op->getLoc(), mapMin, ValueRange{cst1, cst2});
    llvm::outs() << "=====> AffineMinOp: " << minOp << "\n";

    SmallVector<OpFoldResult> foldResults;
    minOp->fold(foldResults);
    llvm::outs() << "=====> folded with default operands: "
                 << foldResults.front() << "\n";

    SmallVector<OpFoldResult> foldResults_1;
    SmallVector<Attribute> constOperands{builder.getIndexAttr(77),
                                         builder.getIndexAttr(9)};
    minOp->fold(constOperands, foldResults_1);
    llvm::outs() << "=====> folded with new operands: " << foldResults_1.front()
                 << "\n";

    // concat map3 and map9
    AffineMap concatMap = concatAffineMaps(llvm::ArrayRef<AffineMap>{map3, map9});
    llvm::outs() << "Concatenated map: " << concatMap << "\n";

    // infer from exprs
    auto inferList = AffineMap::inferFromExprList(
        ArrayRef<ArrayRef<AffineExpr>>{{getAffineDimExpr(3, ctx)},
                                       {getAffineDimExpr(0, ctx)}},
        ctx);
    for (auto inferItem : inferList) {
      llvm::outs() << "Infered map: " << inferItem << "\n";
    }

    // call lingal.MatmulOp TilingInterface
    for (auto func : module.getOps<func::FuncOp>()) {
      func.walk([&](Operation *op) {
        if (isa<linalg::LinalgOp>(op)) {
          linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
          llvm::outs() << "Linalg Matmul getIndexingMaps: " << linalgOp.getIndexingMaps() << "\n";
          llvm::outs() << "Linalg Matmul getLoopsToShapesMap: " << linalgOp.getLoopsToShapesMap() << "\n";
          llvm::outs() << "Linalg Matmul getShapesToLoopsMap: " << linalgOp.getShapesToLoopsMap() << "\n";
          for (auto indexingMap : linalgOp.getIndexingMapsArray()) {
            llvm::outs() << "Linalg Matmul indexingMap: " << indexingMap << "\n";
          }
          auto opDims =
              linalgOp.createFlatListOfOperandDims(builder, op->getLoc());
          for (auto shape : opDims) {
            llvm::outs() << "Linalg Matmul operand shape: " << shape << "\n";
          }
          auto foldRet = affine::makeComposedFoldedAffineApply(
              builder, op->getLoc(), inferList.front(), opDims);
          llvm::outs() << "Linalg Matmul composed folded affine apply: "
                       << foldRet << "\n";
          // insert an affine applyOp into IR
          llvm::SmallVector<Value> valueOperands;
          auto foldmap = mlir::foldAttributesIntoMap(builder, inferList.front(),
                                                     opDims, valueOperands);
          llvm::outs() << "foldmap: " << foldmap << "\n";
          auto applyOp = affine::makeComposedAffineApply(
              builder, op->getLoc(), inferList.front(), opDims);
          llvm::outs() << "AffineApplyOp: " << applyOp << "\n";

          for (auto ret : linalgOp.getShapesToLoopsMap().getResults()) {
            llvm::outs() << "Linalg Matmul result of getShapesToLoopsMap: " << ret << "\n";
          }

          // test tiling interface
          auto tilingInterface = cast<TilingInterface>(op);
          for (auto range : tilingInterface.getIterationDomain(builder)) {
            llvm::outs() << "Linalg Matmul iteration domain: " << range.offset << ", " << range.size << ", "
                         << range.stride << "\n";
          }
          for (auto itertype : tilingInterface.getLoopIteratorTypes()) {
            llvm::outs() << "Linalg Matmul iterator type: " << itertype << "\n";
          }
        }
        return WalkResult::advance();
      });
    }
  }

};
} // namespace mlir

namespace mlir {
void registerTestAffineMapPass() { PassRegistration<TestAffineMapPass>(); }
} // namespace mlir
