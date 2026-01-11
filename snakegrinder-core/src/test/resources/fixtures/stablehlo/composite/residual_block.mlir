module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
    // Linear layer: fc1
    %1_weight = stablehlo.constant dense<0.0> : tensor<8x8xf32>  // placeholder for weight
    %1_matmul = stablehlo.dot_general %arg0, %1_weight, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]> : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
    %1_bias = stablehlo.constant dense<0.0> : tensor<8xf32>  // placeholder for bias
    %1 = stablehlo.add %1_matmul, %1_bias : tensor<1x8xf32>
    %2_zero = stablehlo.constant dense<0.0> : tensor<1x8xf32>
    %2 = stablehlo.maximum %1, %2_zero : tensor<1x8xf32>
    // Linear layer: fc2
    %3_weight = stablehlo.constant dense<0.0> : tensor<8x8xf32>  // placeholder for weight
    %3_matmul = stablehlo.dot_general %2, %3_weight, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]> : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
    %3_bias = stablehlo.constant dense<0.0> : tensor<8xf32>  // placeholder for bias
    %3 = stablehlo.add %3_matmul, %3_bias : tensor<1x8xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<1x8xf32>
    stablehlo.return %4 : tensor<1x8xf32>
  }
}