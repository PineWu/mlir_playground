#ifndef EMITC_CONVERSION_PASSES_TD
#define EMITC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertLinalgToEmitC : Pass<"convert-linalg-to-emitc", "func::FuncOp"> {
    let summary = "Convert Linalg dialect to EmitC dialect.";
    let constructor = "createConvertLinalgToEmitCPass()";
    let dependentDialects = ["EmitCDialect"];
}

#endif // EMITC_CONVERSION_PASSES_TD
