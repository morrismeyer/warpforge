module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
    %1 = stablehlo.constant dense<1.0> : tensor<1x8xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<1x8xf32>
    stablehlo.return %2 : tensor<1x8xf32>
  }
}