module @main {
  func.func public @forward(%arg0: tensor<1x4x8x8xf32>) -> (tensor<1x4x4x4xf32>) {
    // MaxPool2d layer: pool
    %1_init = stablehlo.constant dense<-3.40282e+38> : tensor<f32>
    %1 = stablehlo.reduce_window %arg0, %1_init, window = [1, 1, 2, 2], strides = [1, 1, 2, 2], padding_low = [0, 0, 0, 0], padding_high = [0, 0, 0, 0], reducer = max : (tensor<1x4x8x8xf32>, tensor<f32>) -> tensor<1x4x4x4xf32>
    stablehlo.return %1 : tensor<1x4x4x4xf32>
  }
}