/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
Building the code:
cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_OPT_LEVEL=generic \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=$CONDA/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DFAISS_ENABLE_CUVS=ON \
    -DCUDAToolkit_ROOT="/usr/local/cuda/lib64" \
    -DBUILD_TESTING=ON \
    -DFAISS_ENABLE_EXTRAS=ON \
    .

make -C build -j6 faiss fp_16_test

*/


#include <iostream>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_factory.h>
#include <cuda_fp16.h>

using idx_t = faiss::idx_t;
std::string data_path = "/home/ec2-user/faiss/tutorial/cpp/data/sift";

bool read_fvecs_file(
        const std::string& filepath,
        std::vector<float>& out_data,
        size_t& out_num_vecs,
        size_t& out_dim) {
    std::ifstream input(filepath, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open file: " << filepath << "\n";
        return false;
    }

    // Read dimension of first vector
    int dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));

    // Calculate number of vectors
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    size_t entry_size = sizeof(int) + dim * sizeof(float);
    if (file_size % entry_size != 0) {
        std::cerr << "File size not divisible by vector size in: " << filepath
                  << "\n";
        return false;
    }
    size_t num_vecs = file_size / entry_size;

    // Read all vectors
    input.seekg(0, std::ios::beg);
    out_data.resize(num_vecs * dim);

    for (size_t i = 0; i < num_vecs; ++i) {
        int cur_dim;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "Inconsistent vector dimension at index " << i
                      << " in file: " << filepath << "\n";
            return false;
        }
        input.read(
                reinterpret_cast<char*>(&out_data[i * dim]),
                dim * sizeof(float));
    }

    out_dim = static_cast<size_t>(dim);
    out_num_vecs = num_vecs;
    return true;
}

std::vector<std::vector<int>> read_ground_truth(const std::string& filepath) {
    std::vector<std::vector<int>> vectors;
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filepath << "\n";
        return vectors;
    }

    int k = 100;
    int value;
    int print = 0;
    while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        std::vector<int> current_vector;
        if(print <=2) {
            std::cout<<"First value is is : "<<value<<std::endl;
            print += 1;
        }
        //current_vector.push_back(value);
        for (int i = 0; i < k; ++i) {
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            current_vector.push_back(value);
        }
        vectors.push_back(current_vector);
    }

    file.close();
    return vectors;
}

float calculate_recall(std::vector<std::vector<int>> results, std::vector<std::vector<int>> GT, int k, size_t num_query) {
    float neighbors_found = 0;
    for(int i = 0 ; i < num_query ; i ++) {
        std::vector<int> query_gt = GT[i];
        for(int j = 0 ; j < k ; j++) {
            int result_id = results[i][j];
            if (std::find(query_gt.begin(), query_gt.begin() + k, result_id) != query_gt.end()) {
                neighbors_found += 1;
            }
        }
    }
    return neighbors_found/(num_query * k);
}

int main() {
    size_t graph_degree = 64;
    size_t intermediate_graph_degree = 128;
    auto metric = faiss::METRIC_L2;
    auto build_algo = faiss::gpu::graph_build_algo::IVF_PQ;
    int k = 100;

    // === Load train and query vectors for fp32 and also convert to fp16 ===
    std::vector<float> trainVecs, queryVecs;
    size_t n_train, n_query, n_dim;

    read_fvecs_file(data_path + "/sift_base.fvecs", trainVecs, n_train, n_dim);
    read_fvecs_file(data_path + "/sift_query.fvecs", queryVecs, n_query, n_dim);
    
    std::cout << "Train: " << n_train << " x " << n_dim << std::endl;
    std::cout << "Query: " << n_query << " x " << n_dim << std::endl;
    
    std::vector<std::vector<int>> GT = read_ground_truth(data_path + "/sift_groundtruth.ivecs");

    std::vector<__half> trainVecs_half(trainVecs.size());
    std::vector<__half> queryVecs_half(queryVecs.size());

    for (size_t i = 0; i < trainVecs.size(); ++i) {
        trainVecs_half[i] = __float2half(trainVecs[i]);
    }
    for (size_t i = 0; i < queryVecs.size(); ++i) {
        queryVecs_half[i] = __float2half(queryVecs[i]);
    }

    std::cout << "Train and search gpu index for FP16\n";
    faiss::gpu::GpuIndexCagraConfig config;
    config.device = 0;
    config.graph_degree = graph_degree;
    config.intermediate_graph_degree = intermediate_graph_degree;
    config.build_algo = build_algo;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();

    faiss::gpu::GpuIndexCagra gpuIndex(&res, n_dim, metric, config);

    gpuIndex.train(n_train, static_cast<void*>(trainVecs_half.data()), faiss::NumericType::Float16);

    faiss::IndexHNSWCagra hnswCagra;
    hnswCagra.base_level_only = true;
    gpuIndex.copyTo(&hnswCagra, faiss::NumericType::Float16);
    
    std::vector<float> D(n_query * k);
    std::vector<idx_t> I(n_query * k);

    faiss::SearchParametersHNSW hnsw_params;
    hnsw_params.efSearch = 100;

    hnswCagra.search(n_query, queryVecs.data(), k, D.data(), I.data(), &hnsw_params);
    std::cout<<"Total Queries: "<<n_query<<std::endl;

    std::vector<std::vector<int>> results;
    for(int i = 0 ; i < n_query; i++) {
        std::vector<int> res;
        for(int j = 0; j < k; j++) {
            res.push_back(I[(i * k) + j])
        }
        results.push_back(res);
    }

    float recall_at_1 = calculate_recall(results, GT, 1, n_query);
    std::cout <<"Recall at 1 is: "<< recall_at_1 <<std::endl;

    float recall_at_k = calculate_recall(results, GT, k, n_query);
    std::cout <<"Recall at k is: "<< recall_at_k <<" where k is : "<< k <<std::endl;
    return 0;
}