add_mlir_library(MLIRLinalgToEmitC
    LinalgToEmitC.cpp

    DEPENDS
    MLIREmitCDialect
    MLIREmitCConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)
