module @main {
  func.func public @forward(%arg0: tensor<2x8xf32>) -> (tensor<4x4xf32>) {
    %1 = stablehlo.reshape %arg0 : tensor<2x8xf32> -> tensor<4x4xf32>
    stablehlo.return %1 : tensor<4x4xf32>
  }
}