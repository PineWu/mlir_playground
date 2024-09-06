#include "mlir/Dialect/EmitC/IR/EmitC.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "emitc/Conversion/LinalgToEmitC.h"

using namespace mlir;
using namespace mlir::emitc;

#define GEN_PASS_CLASSES
#include "emitc/Conversion/Passes.h.inc"

class MatmulOpConversion : public OpConversionPattern<linalg::MatmulOp> {
    using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

public:
    MatmulOpConversion(MLIRContext *ctx)
        : OpConversionPattern<linalg::MatmulOp>(ctx) {}

private:
    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp, OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {

        // get memory address
        auto lhs = matmulOp.getOperand(0);
        auto rhs = matmulOp.getOperand(1);
        auto result = matmulOp.getOperand(0);

        // substract shape of input and output matrix
        auto lhsShape = lhs.getType().cast<ShapedType>().getShape();
        auto rhsShape = rhs.getType().cast<ShapedType>().getShape();
        auto resShape = result.getType().cast<ShapedType>().getShape();
        auto elementType = lhs.getType().cast<ShapedType>().getElementType();

        //auto ptrType = rewriter.getType<LLVM::LLVMPointerType>(lhs.getType().cast<MemRefType>().getElementType());

        auto loc = matmulOp.getLoc();
        /*
        auto callOperands = ValueRange{
            rewriter.create<emitc::ConstantOp>(loc, elementType, rewriter.getIndexAttr(lhsShape[0])),
            rewriter.create<emitc::ConstantOp>(loc, elementType, rewriter.getIndexAttr(lhsShape[1])),
            rewriter.create<emitc::ConstantOp>(loc, elementType, rewriter.getIndexAttr(rhsShape[1])),
        };
        */

        StringRef funcName = "beer::genmatmul";
        StringAttr callee = rewriter.getStringAttr(funcName);

        ArrayAttr args;
        ArrayAttr templateArgs;

        rewriter.setInsertionPoint(matmulOp);
        auto callOp = rewriter.create<emitc::CallOp>(
                matmulOp.getLoc(), matmulOp.getResultTypes(),
                //callee, args, templateArgs,
                callee,
                ValueRange{
                    rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(lhsShape[0])),
                    rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(lhsShape[1])),
                    rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(rhsShape[1])),
                }
            );
        rewriter.eraseOp(matmulOp);

        //rewriter.replaceOp(matmulOp, callOp.getResult(0));
        //rewriter.replaceOpWithNewOp<emitc::CallOp>(matmulOp, matmulOp.getOperands().getType(),
        //        callee, args, templateArgs, adaptor.getOperands());
        return success();
    }
};



void populateLinalgToEmitCPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<MatmulOpConversion>(ctx);
}

namespace {

struct ConvertLinalgToEmitCPass
    : public ConvertLinalgToEmitCBase<ConvertLinalgToEmitCPass> {
    /// Perform the lowering to EmitC dialect.
    void runOnOperation() override {
        ConversionTarget target(getContext());

        target.addLegalDialect<emitc::EmitCDialect>();
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addLegalDialect<arith::ArithDialect>();

        target.addIllegalOp<linalg::MatmulOp>();

        RewritePatternSet patterns(&getContext());
        populateLinalgToEmitCPatterns(&getContext(), patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
        
    }
};

} // namespace


std::unique_ptr<OperationPass<func::FuncOp>>
mlir::emitc::createConvertLinalgToEmitCPass(){
    return std::make_unique<ConvertLinalgToEmitCPass>();
}

namespace mlir {
    void registerLinalgToEmitCPass() {
        PassRegistration<ConvertLinalgToEmitCPass>();
    }
}
