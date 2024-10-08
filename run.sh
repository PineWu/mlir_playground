LLVM_BUILD_DIR="/home/lixiang/Workspace/llvm-project/build"

if [ -d "./build" ]; then
    echo "build is already exists;"
else
    mkdir build
fi

cd build

cmake -G Ninja .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit

cmake --build .

#cmake --build . --target mlir-doc

cd -

## test
echo "============ Test print IR nesting ================"
./build/bin/beer-opt --test-print-nesting ./test/payloads/hello_matmul.mlir

echo "============ Test AffineMap and TilingInterface ================"
./build/bin/beer-opt --test-affinemap-tilinginterface ./test/payloads/hello_matmul.mlir
