module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
    %1 = stablehlo.tanh %arg0 : tensor<1x8xf32>
    stablehlo.return %1 : tensor<1x8xf32>
  }
}