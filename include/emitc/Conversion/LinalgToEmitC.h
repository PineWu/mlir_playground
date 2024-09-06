#ifndef EMITC_CONVSERSION_LINALGTOEMITC_H
#define EMITC_CONVSERSION_LINALGTOEMITC_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
//using mlir::func::FuncOp;

namespace emitc {

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createConvertLinalgToEmitCPass();

} // namespace emitc
} // namespace mlir

#endif //EMITC_CONVSERSION_LINALGTOEMITC_H

