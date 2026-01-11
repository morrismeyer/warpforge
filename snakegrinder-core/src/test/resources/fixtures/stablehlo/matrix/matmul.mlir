module @main {
  func.func public @forward(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x8x4xf32>) -> (tensor<1x4x4xf32>) {
    %1 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1x4x8xf32>, tensor<1x8x4xf32>) -> tensor<1x4x4xf32>
    stablehlo.return %1 : tensor<1x4x4xf32>
  }
}