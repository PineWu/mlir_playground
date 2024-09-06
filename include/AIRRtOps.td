#ifndef AIRRT_OPS
#define AIRRT_OPS

include "mlir/IR/OpBase.td"

def AIRRt_Dialect : Dialect {
  let name = "airrt";
  let cppNamespace = "xilinx::airrt";
  let description = [{
The `airrt` dialect is an IR for interacting with the AIR runtime.
It enapsulates data structures and functions available in the runtime and
can be lowered to a combination of standard and LLVM dialects.
  }];
  let useDefaultTypePrinterParser = 1;
}

def AIRRt_Event : DialectType<
  AIRRt_Dialect, CPred<"$_self.isa<xilinx::airrt::EventType>()">, "event type">,
             BuildableType<"xilinx::airrt::EventType::get($_builder.getContext())">;

class AIRRt_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIRRt_Dialect, mnemonic, traits> {
}

def AIRRt_HerdMetadataOp
    : AIRRt_Op<"herd_metadata", [
        HasParent<"SegmentMetadataOp">,
        SingleBlockImplicitTerminator<"HerdMetadataTerminatorOp">
      ]> {
  let summary = "Runtime metadata for a single herd";
  let description = [{
    Runtime metadata for a single herd.

  }];
  let arguments = (ins
    StrAttr:$sym_name
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // #ifndef AIRRT_OPS