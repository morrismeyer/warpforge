module @main {
  func.func public @forward(%arg0: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
    %1_exp = stablehlo.exponential %arg0 : tensor<1x8xf32>
    // Note: Full softmax requires reduction and broadcast - simplified here
    %1 = stablehlo.divide %1_exp, %1_exp : tensor<1x8xf32>  // placeholder
    stablehlo.return %1 : tensor<1x8xf32>
  }
}