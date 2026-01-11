module @main {
  func.func public @forward(%arg0: tensor<2x8xf32>) -> (tensor<2xf32>) {
    %1_init = stablehlo.constant dense<-3.40282e+38> : tensor<f32>
    %1 = stablehlo.reduce %arg0, %1_init, dims = [1], reducer = max : (tensor<2x8xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.slice %1, starts = [0, 0], limits = [1, 2], strides = [1, 1] : tensor<f32> -> tensor<2xf32>
    stablehlo.return %2 : tensor<2xf32>
  }
}