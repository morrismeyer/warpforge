module @main {
  func.func public @forward(%arg0: tensor<1x4x8xf32>) -> (tensor<4x4x8xf32>) {
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<1x4x8xf32>) -> tensor<4x4x8xf32>
    stablehlo.return %1 : tensor<4x4x8xf32>
  }
}