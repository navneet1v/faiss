
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(1-Flat EXCLUDE_FROM_ALL 1-Flat.cpp)
target_link_libraries(1-Flat PRIVATE faiss)

add_executable(2-IVFFlat EXCLUDE_FROM_ALL 2-IVFFlat.cpp)
target_link_libraries(2-IVFFlat PRIVATE faiss)

add_executable(3-IVFPQ EXCLUDE_FROM_ALL 3-IVFPQ.cpp)
target_link_libraries(3-IVFPQ PRIVATE faiss)

add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
target_link_libraries(4-GPU PRIVATE faiss)

add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
target_link_libraries(5-Multiple-GPUs PRIVATE faiss)


add_executable(6-HNSW EXCLUDE_FROM_ALL 6-HNSW.cpp)
target_link_libraries(6-HNSW PRIVATE faiss)

add_executable(7-PQFastScan EXCLUDE_FROM_ALL 7-PQFastScan.cpp)
target_link_libraries(7-PQFastScan PRIVATE faiss)

add_executable(8-PQFastScanRefine EXCLUDE_FROM_ALL 8-PQFastScanRefine.cpp)
target_link_libraries(8-PQFastScanRefine PRIVATE faiss)

add_executable(navneet-test EXCLUDE_FROM_ALL navneet-test.cpp)
target_link_libraries(navneet-test PRIVATE faiss)

# Link OMP as it is required for HNSW
find_package(OpenMP REQUIRED)


add_executable(cagra-cpu-test EXCLUDE_FROM_ALL cagra-cpu-test.cpp)
target_link_libraries(cagra-cpu-test PRIVATE faiss)
target_link_libraries(cagra-cpu-test PRIVATE OpenMP::OpenMP_CXX)


add_executable(cagra-cpu-check-id-map-working EXCLUDE_FROM_ALL cagra-cpu-check-id-map-working.cpp)
target_link_libraries(cagra-cpu-check-id-map-working PRIVATE faiss)
target_link_libraries(cagra-cpu-check-id-map-working PRIVATE OpenMP::OpenMP_CXX)


option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)

if(FAISS_ENABLE_RAFT)
    find_package(CUDAToolkit REQUIRED)
    ## Find the raft package
    find_package(raft COMPONENTS compiled distributed)
    add_executable(cagra-test EXCLUDE_FROM_ALL cagra-test.cpp)
    ## Add the raft dependencies
    target_link_libraries(cagra-test PRIVATE faiss CUDA::cudart raft::raft raft::compiled)

    add_executable(check-id-map-working EXCLUDE_FROM_ALL check-id-map-working.cpp)
    target_link_libraries(check-id-map-working PRIVATE faiss CUDA::cudart raft::raft raft::compiled)

endif()


# You can create a commons link where you link all your libraries
add_library(commons commons.cpp)
# Make sure others can see these links
target_link_libraries(commons PUBLIC OpenMP::OpenMP_CXX faiss)
# HNSW Testing
add_executable(hnsw-test EXCLUDE_FROM_ALL hnsw-test.cpp)
target_link_libraries(hnsw-test commons)



# Some random tests
target_link_libraries(navneet-test PRIVATE OpenMP::OpenMP_CXX gtest_main)

# This is for OMP
add_executable(omptest EXCLUDE_FROM_ALL omptest.cpp)
target_link_libraries(omptest PRIVATE faiss)

target_link_libraries(omptest PRIVATE OpenMP::OpenMP_CXX gtest_main)
