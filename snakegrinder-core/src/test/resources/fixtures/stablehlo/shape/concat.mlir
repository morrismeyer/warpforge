module @main {
  func.func public @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> (tensor<1x8xf32>) {
    %1 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x8xf32>
    stablehlo.return %1 : tensor<1x8xf32>
  }
}