#ifndef EMITC_CONVERSION_PASSES_H
#define EMITC_CONVERSION_PASSES_H

#include "emitc/Conversion/LinalgToEmitC.h"

namespace mlir {
namespace emitc { 

#define GEN_PASS_REGISTRATION
#include "emitc/Conversion/Passes.h.inc"

} // namespace emitc
} // namesapce mlir

#endif // EMITC_CONVERSION_PASSES_H

