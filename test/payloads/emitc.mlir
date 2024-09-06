module {
    func.func @matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<128x256xf32>) {
            %c42 = arith.constant 42 : index
                emitc.call "emitc::genmatmul"(%c42) : (index) -> ()
                    return
                      
    }
    
}
