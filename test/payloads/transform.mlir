func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.  
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  
  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // transform.test_print_remark_at_operand %arg1, "matmul"
  //     : !transform.op<"linalg.matmul">
  // transform.test_print_remark_at_operand %arg2, "elemwise_binaries"
  //     : !transform.op<"linalg.elemwise_binary">

  // Since the %arg2 handle is associated with both elementwise operations,
  // we need to split it into two handles so we can target only the second
  // elementwise operation.
  %add, %max = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">)
      -> (!transform.any_op, !transform.any_op)

  // The actual tiling transformation takes tile sizes as attributes. It produces a
  // handle to the loop generated during tiling.
  %loop, %tiled = transform.structured.tile_to_forall_op %max tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // We can now fuse the other operations into the loop. Here, we fuse
  // operations one-by-one. This requires the operation that is being fused
  // to define the value used within the loop, so the order of such fusions
  // is important. We could also use "transform.merge_handles" to obtain
  // a single handle to all operations and give it to `fuse_into_containing_op`
  // that would take care of the ordering in this case.
  %add_fused, %new_forall = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused,%new_forall2 = transform.structured.fuse_into_containing_op %arg1 into %loop
      : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Tile again to get the desired size. Note that this time this tiles the
  // "add" operation and fuses matmul into the loop, but doesn't affect the
  // "max" operation. This illustrates the precise targeting with the transform
  // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
  // of which having the same kind.
  %loop_2, %tiled_2 = transform.structured.tile_to_forall_op %add_fused tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused_2, %new_forall3 = transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Since outlining is currently only implemented for region-holding operations
  // such as loops, use tiling to size 1 to materialize the outer loop that is
  // going to be outlined.
  %outline_target, %_ = transform.structured.tile_to_forall_op %tiled_2 tile_sizes [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
   transform.structured.fuse_into_containing_op %matmul_fused_2 into %outline_target
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

  transform.yield
}