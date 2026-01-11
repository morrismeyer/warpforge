module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x4xf32>) {
    // Linear layer: fc
    %1_weight = stablehlo.constant dense<0.0> : tensor<4x8xf32>  // placeholder for weight
    %1_matmul = stablehlo.dot_general %arg0, %1_weight, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]> : (tensor<1x8xf32>, tensor<4x8xf32>) -> tensor<1x4xf32>
    %1_bias = stablehlo.constant dense<0.0> : tensor<4xf32>  // placeholder for bias
    %1 = stablehlo.add %1_matmul, %1_bias : tensor<1x4xf32>
    stablehlo.return %1 : tensor<1x4xf32>
  }
}