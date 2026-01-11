module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x2xf32>) {
    %1 = stablehlo.slice %arg0, starts = [0, 0], limits = [1, 2], strides = [1, 1] : tensor<1x8xf32> -> tensor<1x2xf32>
    stablehlo.return %1 : tensor<1x2xf32>
  }
}