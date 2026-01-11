module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
    %1_zero = stablehlo.constant dense<0.0> : tensor<1x8xf32>
    %1 = stablehlo.maximum %arg0, %1_zero : tensor<1x8xf32>
    stablehlo.return %1 : tensor<1x8xf32>
  }
}