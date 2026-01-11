module @main {
  func.func public @forward(%arg0: tensor<2x8xf32>) -> (tensor<2xf32>) {
    %1_init = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.reduce %arg0, %1_init, dims = [1], reducer = add : (tensor<2x8xf32>, tensor<f32>) -> tensor<2xf32>
    stablehlo.return %1 : tensor<2xf32>
  }
}