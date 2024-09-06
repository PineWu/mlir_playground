module {
    func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x256xf32>, %C: memref<128x256xf32>) {
        linalg.matmul
            ins(%A, %B: memref<128x128xf32>, memref<128x256xf32>)
            outs(%C: memref<128x256xf32>)
        return
    }
}
