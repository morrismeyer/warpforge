module @main {
  func.func public @forward(%arg0: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
    %1 = stablehlo.transpose %arg0, dims = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
    stablehlo.return %1 : tensor<8x4xf32>
  }
}