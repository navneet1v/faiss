## Compiling Faiss For M3 Mac
```
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_CUVS=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON  -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"
```

### Running Tutorial
Add the below details
```
find_package(OpenMP REQUIRED)
target_link_libraries(6-HNSW PRIVATE faiss OpenMP::OpenMP_CXX)
```

Now you can run via CLION

### Add this in CMake setting of CLion
```
-G "Unix Makefiles" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_CUVS=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON  -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"
```

Build directory:
```
build
```

Threads:
```
-j 10
```