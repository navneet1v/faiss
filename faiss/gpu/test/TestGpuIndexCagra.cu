// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/neighborhood_recall.cuh>

struct Options {
    Options() {
        numTrain = 2 * faiss::gpu::randVal(2000, 5000);
        dim = faiss::gpu::randVal(4, 10);
        numAdd = faiss::gpu::randVal(1000, 3000);

        graphDegree = faiss::gpu::randSelect({32, 64});
        intermediateGraphDegree = faiss::gpu::randSelect({64, 98});
        buildAlgo = faiss::gpu::randSelect(
                {faiss::gpu::graph_build_algo::IVF_PQ,
                 faiss::gpu::graph_build_algo::NN_DESCENT});

        numQuery = faiss::gpu::randVal(32, 100);
        k = faiss::gpu::randVal(10, 30);

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "CAGRA device " << device << " numVecs " << numTrain << " dim "
            << dim << " graphDegree " << graphDegree
            << " intermediateGraphDegree " << intermediateGraphDegree
            << "buildAlgo " << static_cast<int>(buildAlgo) << " numQuery "
            << numQuery << " k " << k;

        return str.str();
    }

    int numTrain;
    int numAdd;
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    faiss::gpu::graph_build_algo buildAlgo;
    int numQuery;
    int k;
    int device;
};

void queryTest(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // train gpu index
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        // test quality of searches
        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                testDistance.data(),
                testIndices.data());

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                test_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                test_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_Query_L2) {
    queryTest(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float32_Query_IP) {
    queryTest(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void queryTestFP16(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // train gpu index
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        gpuIndex.train_ex(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        // test quality of searches
        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        // Create half vector
        std::vector<__half> queryVecs_half(queryVecs.size());

        for (size_t i = 0; i < queryVecs.size(); ++i) {
            queryVecs_half[i] = __float2half(queryVecs[i]);
        }
        gpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                testDistance.data(),
                testIndices.data());

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                test_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                test_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float16_Query_L2) {
    queryTestFP16(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float16_Query_IP) {
    queryTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void copyToTest(
        faiss::MetricType metric,
        double expected_recall,
        bool base_level_only) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numAdd, opt.dim, addVecs.data());
        }

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // train gpu index and copy to cpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        faiss::IndexHNSWCagra copiedCpuIndex(
                opt.dim, opt.graphDegree / 2, metric);
        copiedCpuIndex.base_level_only = base_level_only;
        gpuIndex.copyTo(&copiedCpuIndex);
        copiedCpuIndex.hnsw.efConstruction = opt.k * 2;

        // add more vecs to copied cpu index
        if (!base_level_only) {
            copiedCpuIndex.add(opt.numAdd, addVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // add more vecs to cpu index
        if (!base_level_only) {
            cpuIndex.add(opt.numAdd, addVecs.data());
        }

        // query indexes
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        std::vector<float> copyRefDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> copyRefIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParamstwo;
        cpuSearchParamstwo.efSearch = opt.k * 2;
        copiedCpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyRefDistance.data(),
                copyRefIndices.data(),
                &cpuSearchParamstwo);

        // test quality of search
        auto gpuRes = res.getResources();

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto copyRefDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto copyRefIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto copy_ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                copyRefDistanceDev.data(), opt.numQuery, opt.k);
        auto copy_ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_ref_dis_mds);
        auto copy_ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyRefIndicesDev.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_ref_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                copy_ref_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_CopyTo_L2) {
    copyToTest(faiss::METRIC_L2, 0.98, false);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_L2_BaseLevelOnly) {
    copyToTest(faiss::METRIC_L2, 0.98, true);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_IP) {
    copyToTest(faiss::METRIC_INNER_PRODUCT, 0.98, false);
}

TEST(TestGpuIndexCagra, Float32_CopyTo_IP_BaseLevelOnly) {
    copyToTest(faiss::METRIC_INNER_PRODUCT, 0.98, true);
}

void copyToTestFP16(
        faiss::MetricType metric,
        double expected_recall,
        bool base_level_only) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numAdd, opt.dim, addVecs.data());
        }

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // train gpu index and copy to cpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train_ex(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

        faiss::IndexHNSWCagra copiedCpuIndex(
                opt.dim,
                opt.graphDegree / 2,
                metric,
                faiss::NumericType::Float16);
        copiedCpuIndex.base_level_only = base_level_only;
        gpuIndex.copyTo(&copiedCpuIndex);
        copiedCpuIndex.hnsw.efConstruction = opt.k * 2;

        // train cpu index
        faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        // query indexes
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        std::vector<float> refDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParams;
        cpuSearchParams.efSearch = opt.k * 2;
        cpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                refDistance.data(),
                refIndices.data(),
                &cpuSearchParams);

        std::vector<float> copyRefDistance(opt.numQuery * opt.k, 0);
        std::vector<faiss::idx_t> copyRefIndices(opt.numQuery * opt.k, -1);
        faiss::SearchParametersHNSW cpuSearchParamstwo;
        cpuSearchParamstwo.efSearch = opt.k * 2;
        copiedCpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyRefDistance.data(),
                copyRefIndices.data(),
                &cpuSearchParamstwo);

        // test quality of search
        auto gpuRes = res.getResources();

        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                refIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto copyRefDistanceDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefDistance,
                gpuRes->getDefaultStreamCurrentDevice());
        auto copyRefIndicesDev = faiss::gpu::toDeviceTemporary(
                gpuRes.get(),
                copyRefIndices,
                gpuRes->getDefaultStreamCurrentDevice());

        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                refDistanceDev.data(), opt.numQuery, opt.k);
        auto ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        ref_dis_mds);
        auto ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        refIndicesDev.data(), opt.numQuery, opt.k);

        auto copy_ref_dis_mds = raft::make_device_matrix_view<const float, int>(
                copyRefDistanceDev.data(), opt.numQuery, opt.k);
        auto copy_ref_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_ref_dis_mds);
        auto copy_ref_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyRefIndicesDev.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_ref_ind_mds,
                ref_ind_mds,
                recall_score.view(),
                copy_ref_dis_mds_opt,
                ref_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

// For fp16, only base level copy is supported
TEST(TestGpuIndexCagra, Float16_CopyTo_L2_BaseLevelOnly) {
    copyToTestFP16(faiss::METRIC_L2, 0.98, true);
}

TEST(TestGpuIndexCagra, Float16_CopyTo_IP_BaseLevelOnly) {
    copyToTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98, true);
}

void copyFromTest(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWCagra cpuIndex(opt.dim, opt.graphDegree / 2, metric);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, opt.dim, metric);
        copiedGpuIndex.copyFrom(&cpuIndex);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        gpuIndex.train(opt.numTrain, trainVecs.data());

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> copyTestDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        copiedGpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                copyTestDistance.data(),
                copyTestIndices.data());

        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                testDistance.data(),
                testIndices.data());

        // test quality of searches
        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        auto copy_test_dis_mds =
                raft::make_device_matrix_view<const float, int>(
                        copyTestDistance.data(), opt.numQuery, opt.k);
        auto copy_test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_test_dis_mds);

        auto copy_test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyTestIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_test_ind_mds,
                test_ind_mds,
                recall_score.view(),
                copy_test_dis_mds_opt,
                test_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float32_CopyFrom_L2) {
    copyFromTest(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float32_CopyFrom_IP) {
    copyFromTest(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void copyFromTestFP16(faiss::MetricType metric, double expected_recall) {
    for (int tries = 0; tries < 5; ++tries) {
        Options opt;
        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
            metric == faiss::METRIC_INNER_PRODUCT) {
            continue;
        }

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
        }

        // train cpu index
        faiss::IndexHNSWCagra cpuIndex(
                opt.dim,
                opt.graphDegree / 2,
                metric,
                faiss::NumericType::Float16);
        cpuIndex.hnsw.efConstruction = opt.k * 2;
        cpuIndex.add(opt.numTrain, trainVecs.data());

        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // convert to gpu index
        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, opt.dim, metric);
        copiedGpuIndex.copyFrom_ex(&cpuIndex, faiss::NumericType::Float16);

        // train gpu index
        faiss::gpu::GpuIndexCagraConfig config;
        config.device = opt.device;
        config.graph_degree = opt.graphDegree;
        config.intermediate_graph_degree = opt.intermediateGraphDegree;
        config.build_algo = opt.buildAlgo;

        // faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
        // gpuIndex.train(opt.numTrain, trainVecs.data());

        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);

        // Create half vector
        std::vector<__half> trainVecs_half(trainVecs.size());

        for (size_t i = 0; i < trainVecs.size(); ++i) {
            trainVecs_half[i] = __float2half(trainVecs[i]);
        }

        gpuIndex.train_ex(
                opt.numTrain,
                static_cast<void*>(trainVecs_half.data()),
                faiss::NumericType::Float16);

        // query
        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
        if (metric == faiss::METRIC_INNER_PRODUCT) {
            faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
        }

        // Create half vector
        std::vector<__half> queryVecs_half(queryVecs.size());

        for (size_t i = 0; i < queryVecs.size(); ++i) {
            queryVecs_half[i] = __float2half(queryVecs[i]);
        }

        auto gpuRes = res.getResources();
        auto devAlloc = faiss::gpu::makeDevAlloc(
                faiss::gpu::AllocType::FlatData,
                gpuRes->getDefaultStreamCurrentDevice());
        faiss::gpu::DeviceTensor<float, 2, true> copyTestDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        copiedGpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                copyTestDistance.data(),
                copyTestIndices.data());

        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
        gpuIndex.search_ex(
                opt.numQuery,
                queryVecs_half.data(),
                faiss::NumericType::Float16,
                opt.k,
                testDistance.data(),
                testIndices.data());

        // test quality of searches
        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();

        auto test_dis_mds = raft::make_device_matrix_view<const float, int>(
                testDistance.data(), opt.numQuery, opt.k);
        auto test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        test_dis_mds);

        auto test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        testIndices.data(), opt.numQuery, opt.k);

        auto copy_test_dis_mds =
                raft::make_device_matrix_view<const float, int>(
                        copyTestDistance.data(), opt.numQuery, opt.k);
        auto copy_test_dis_mds_opt =
                std::optional<raft::device_matrix_view<const float, int>>(
                        copy_test_dis_mds);

        auto copy_test_ind_mds =
                raft::make_device_matrix_view<const faiss::idx_t, int>(
                        copyTestIndices.data(), opt.numQuery, opt.k);

        double scalar_init = 0;
        auto recall_score = raft::make_host_scalar(scalar_init);

        raft::stats::neighborhood_recall(
                raft_handle,
                copy_test_ind_mds,
                test_ind_mds,
                recall_score.view(),
                copy_test_dis_mds_opt,
                test_dis_mds_opt);
        ASSERT_TRUE(*recall_score.data_handle() > expected_recall);
    }
}

TEST(TestGpuIndexCagra, Float16_CopyFrom_L2) {
    copyFromTestFP16(faiss::METRIC_L2, 0.98);
}

TEST(TestGpuIndexCagra, Float16_CopyFrom_IP) {
    copyFromTestFP16(faiss::METRIC_INNER_PRODUCT, 0.98);
}

void testIDSelectorCagra(faiss::MetricType metricType) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    if (metricType == faiss::METRIC_INNER_PRODUCT) {
        faiss::fvec_renorm_L2(opt.numTrain, opt.dim, trainVecs.data());
    }

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = opt.device;
    config.graph_degree = opt.graphDegree;
    config.intermediate_graph_degree = opt.intermediateGraphDegree;
    // Use only IVF_PQ to avoid NN_DESCENT + IP combination
    config.build_algo = faiss::gpu::graph_build_algo::IVF_PQ;

    faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metricType, config);
    gpuIndex.train(opt.numTrain, trainVecs.data());

    auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
    if (metricType == faiss::METRIC_INNER_PRODUCT) {
        faiss::fvec_renorm_L2(opt.numQuery, opt.dim, queryVecs.data());
    }

    faiss::gpu::TestIDSelectorStruct selector_struct(opt.numTrain);
    faiss::gpu::SearchParametersCagra search_params;
    for (auto& [selectorName, selector] : selector_struct.selector_map) {
        search_params.sel = selector.get();
        faiss::gpu::testIDSelectorSearch(
                &gpuIndex,
                &search_params,
                queryVecs,
                opt.numQuery,
                opt.k,
                selectorName);
    }
}

TEST(TestGpuIndexCagra, IDSelector_L2) {
    testIDSelectorCagra(faiss::METRIC_L2);
}

TEST(TestGpuIndexCagra, IDSelector_IP) {
    testIDSelectorCagra(faiss::METRIC_INNER_PRODUCT);
}

//
// Upper-level projection: GpuIndexCagra::copyTo(...,
// CagraUpperLevelProjectionParams)
//

namespace {

// Well-separated clusters: the exact k-NN graph of this data is disconnected
// across clusters, so a CAGRA graph built from it splits into many components.
constexpr faiss::idx_t kProjN = 20000;
constexpr int kProjDim = 64;
constexpr int kProjClusters = 100;
constexpr int kProjGraphDegree = 32;
constexpr int kProjM = 8; // HNSW m: layer 1 with 2 * m links, promotion 1/(2m)

std::vector<float> clusteredVecs(
        faiss::idx_t n,
        int d,
        int nclusters,
        float sigma,
        std::vector<int>* labels,
        unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, nclusters - 1);
    std::vector<float> centers(size_t(nclusters) * d);
    for (auto& v : centers) {
        v = normal(rng);
    }
    faiss::fvec_renorm_L2(d, nclusters, centers.data());
    std::vector<float> x(size_t(n) * d);
    if (labels) {
        labels->resize(n);
    }
    for (faiss::idx_t i = 0; i < n; i++) {
        int c = pick(rng);
        if (labels) {
            (*labels)[i] = c;
        }
        for (int j = 0; j < d; j++) {
            x[i * d + j] = centers[c * d + j] +
                    sigma / std::sqrt(float(d)) * normal(rng);
        }
    }
    return x;
}

faiss::gpu::GpuIndexCagraConfig projectionConfig() {
    faiss::gpu::GpuIndexCagraConfig config;
    config.graph_degree = kProjGraphDegree;
    config.intermediate_graph_degree = kProjGraphDegree;
    config.build_algo = faiss::gpu::graph_build_algo::IVF_PQ;
    // the projection reads the caller's host array: store_dataset must be false
    config.store_dataset = false;
    return config;
}

faiss::gpu::CagraUpperLevelProjectionParams projectionParams() {
    faiss::gpu::CagraUpperLevelProjectionParams params;
    params.M = kProjM;
    return params;
}

// level-0 neighbour rows of a CPU index, [ntotal][nb0]
std::vector<std::vector<int32_t>> level0Rows(
        const faiss::IndexHNSWCagra& index) {
    std::vector<std::vector<int32_t>> rows(index.ntotal);
    for (faiss::idx_t i = 0; i < index.ntotal; i++) {
        size_t b, e;
        index.hnsw.neighbor_range(i, 0, &b, &e);
        rows[i].assign(
                index.hnsw.neighbors.begin() + b,
                index.hnsw.neighbors.begin() + e);
    }
    return rows;
}

// number of nodes reachable from `start` following level-0 out-edges
faiss::idx_t reachableFrom(
        const std::vector<std::vector<int32_t>>& rows,
        faiss::idx_t start) {
    std::vector<char> seen(rows.size(), 0);
    std::queue<faiss::idx_t> q;
    q.push(start);
    seen[start] = 1;
    faiss::idx_t count = 1;
    while (!q.empty()) {
        faiss::idx_t a = q.front();
        q.pop();
        for (int32_t b : rows[a]) {
            if (b >= 0 && !seen[b]) {
                seen[b] = 1;
                count++;
                q.push(b);
            }
        }
    }
    return count;
}

// recall@k of base-level-only search vs exact search
double baseLevelRecall(
        faiss::IndexHNSWCagra& index,
        const std::vector<float>& base,
        const std::vector<float>& queries,
        faiss::idx_t nq,
        int k) {
    std::vector<float> refD(nq * k), D(nq * k);
    std::vector<faiss::idx_t> refI(nq * k), I(nq * k);
    faiss::knn_L2sqr(
            queries.data(),
            base.data(),
            kProjDim,
            nq,
            kProjN,
            k,
            refD.data(),
            refI.data());
    faiss::SearchParametersHNSW sp;
    sp.efSearch = 64;
    index.search(nq, queries.data(), k, D.data(), I.data(), &sp);
    size_t hits = 0;
    for (faiss::idx_t q = 0; q < nq; q++) {
        std::unordered_set<faiss::idx_t> ref(
                refI.begin() + q * k, refI.begin() + (q + 1) * k);
        for (int j = 0; j < k; j++) {
            hits += ref.count(I[q * k + j]);
        }
    }
    return double(hits) / double(nq * k);
}

} // namespace

TEST(TestGpuIndexCagra, UpperLevelProjection_ConnectsClusteredGraph) {
    std::vector<float> base =
            clusteredVecs(kProjN, kProjDim, kProjClusters, 0.3f, nullptr, 1);
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, kProjDim, faiss::METRIC_L2, projectionConfig());
    gpuIndex.train(kProjN, base.data());

    faiss::IndexHNSWCagra plain, projected;
    plain.base_level_only = true;
    projected.base_level_only = true;
    gpuIndex.copyTo(&plain);
    auto params = projectionParams();
    faiss::gpu::CagraUpperLevelProjectionStats stats;
    gpuIndex.copyTo(&projected, false, &params, &stats);

    auto plainRows = level0Rows(plain);
    auto projectedRows = level0Rows(projected);
    // precondition: the data really fragments the plain CAGRA graph
    ASSERT_LT(reachableFrom(plainRows, 0), kProjN / 2);
    // with the projection every sampled start reaches every node
    std::mt19937 rng(7);
    std::uniform_int_distribution<faiss::idx_t> pick(0, kProjN - 1);
    for (int t = 0; t < 10; t++) {
        EXPECT_EQ(reachableFrom(projectedRows, pick(rng)), kProjN);
    }
    EXPECT_GT(stats.promoted, 0);
    EXPECT_GT(stats.added_by_overwrite, 0);

    // search quality: fresh queries from the same clusters (same seed => the
    // same centers and the same first kProjN points, then nq new points)
    faiss::idx_t nq = 200;
    std::vector<float> fresh = clusteredVecs(
            kProjN + nq, kProjDim, kProjClusters, 0.3f, nullptr, 1);
    std::vector<float> queries(fresh.begin() + kProjN * kProjDim, fresh.end());
    double plainRecall = baseLevelRecall(plain, base, queries, nq, 10);
    double projectedRecall = baseLevelRecall(projected, base, queries, nq, 10);
    EXPECT_GE(projectedRecall, 0.9);
    EXPECT_GT(projectedRecall, plainRecall);
}

TEST(TestGpuIndexCagra, UpperLevelProjection_RowInvariantsAndStats) {
    std::vector<float> base =
            clusteredVecs(kProjN, kProjDim, kProjClusters, 0.3f, nullptr, 2);
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, kProjDim, faiss::METRIC_L2, projectionConfig());
    gpuIndex.train(kProjN, base.data());

    faiss::IndexHNSWCagra plain, projected;
    plain.base_level_only = true;
    projected.base_level_only = true;
    gpuIndex.copyTo(&plain);
    auto params = projectionParams();
    faiss::gpu::CagraUpperLevelProjectionStats stats;
    gpuIndex.copyTo(&projected, false, &params, &stats);

    auto plainRows = level0Rows(plain);
    auto projectedRows = level0Rows(projected);
    const int nb0 = plainRows[0].size();
    const int cap = int(params.max_share * nb0);
    size_t changed = 0;
    for (faiss::idx_t i = 0; i < kProjN; i++) {
        int diff = 0;
        std::unordered_set<int32_t> seen;
        bool plainHasDup = false;
        {
            std::unordered_set<int32_t> s;
            for (int32_t v : plainRows[i]) {
                if (v >= 0 && !s.insert(v).second) {
                    plainHasDup = true;
                }
            }
        }
        for (int j = 0; j < nb0; j++) {
            int32_t v = projectedRows[i][j];
            ASSERT_GE(v, -1);
            ASSERT_LT(v, kProjN);
            diff += v != plainRows[i][j];
            if (!plainHasDup && v >= 0) {
                EXPECT_TRUE(seen.insert(v).second)
                        << "duplicate link in row " << i;
            }
        }
        EXPECT_LE(diff, cap) << "row " << i;
        changed += diff;
    }
    EXPECT_EQ(changed, stats.added_by_overwrite + stats.added_into_empty);

    // promotion is faiss's HNSW level draw for an HNSW with 2 * M links
    faiss::HNSW draw(2 * kProjM);
    draw.assign_probas.clear();
    draw.cum_nneighbor_per_level.clear();
    draw.set_default_probas(2 * kProjM, 1.0f / std::log(2 * kProjM));
    faiss::idx_t expectedPromoted = 0;
    for (faiss::idx_t i = 0; i < kProjN; i++) {
        expectedPromoted += draw.random_level() >= 1;
    }
    EXPECT_EQ(stats.promoted, expectedPromoted);

    // protect_last_in_edge: the projection never makes a node unreachable
    EXPECT_EQ(stats.in_degree0_after, stats.in_degree0_before);
    EXPECT_EQ(stats.rows_not_full, 0); // CAGRA rows are full
}

TEST(TestGpuIndexCagra, UpperLevelProjection_MaxShareZeroAndNullParams) {
    std::vector<float> base =
            clusteredVecs(kProjN, kProjDim, kProjClusters, 0.3f, nullptr, 3);
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, kProjDim, faiss::METRIC_L2, projectionConfig());
    gpuIndex.train(kProjN, base.data());

    faiss::IndexHNSWCagra plain, nullParams, zeroShare;
    plain.base_level_only = nullParams.base_level_only =
            zeroShare.base_level_only = true;
    gpuIndex.copyTo(&plain);
    gpuIndex.copyTo(&nullParams, false, nullptr);
    auto params = projectionParams();
    params.max_share = 0;
    faiss::gpu::CagraUpperLevelProjectionStats stats;
    gpuIndex.copyTo(&zeroShare, false, &params, &stats);

    // no projection params: identical to the existing copyTo
    EXPECT_EQ(level0Rows(nullParams), level0Rows(plain));
    // max_share = 0 on full rows: nothing is overwritten
    EXPECT_EQ(level0Rows(zeroShare), level0Rows(plain));
    EXPECT_EQ(stats.added_by_overwrite, 0);
    EXPECT_GT(stats.skipped_cap, 0);
}

TEST(TestGpuIndexCagra, UpperLevelProjection_InvalidParamsThrow) {
    std::vector<float> base =
            clusteredVecs(5000, kProjDim, 20, 0.3f, nullptr, 4);
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, kProjDim, faiss::METRIC_L2, projectionConfig());
    gpuIndex.train(5000, base.data());

    {
        faiss::IndexHNSWCagra cpu;
        cpu.base_level_only = true;
        auto params = projectionParams();
        params.M = 0; // M is required
        EXPECT_THROW(
                gpuIndex.copyTo(&cpu, false, &params), faiss::FaissException);
    }
    {
        faiss::IndexHNSWCagra cpu;
        cpu.base_level_only = true;
        auto params = projectionParams();
        params.max_share = 1.5f;
        EXPECT_THROW(
                gpuIndex.copyTo(&cpu, false, &params), faiss::FaissException);
    }
    {
        faiss::IndexHNSWCagra cpu;
        cpu.base_level_only =
                false; // the projection keeps the index base-level only
        auto params = projectionParams();
        EXPECT_THROW(
                gpuIndex.copyTo(&cpu, false, &params), faiss::FaissException);
    }
    {
        auto config = projectionConfig();
        config.store_dataset = true; // not supported with the projection
        faiss::gpu::GpuIndexCagra storing(
                &res, kProjDim, faiss::METRIC_L2, config);
        storing.train(5000, base.data());
        faiss::IndexHNSWCagra cpu;
        cpu.base_level_only = true;
        auto params = projectionParams();
        EXPECT_THROW(
                storing.copyTo(&cpu, false, &params), faiss::FaissException);
    }
}

TEST(TestGpuIndexCagra, UpperLevelProjection_Float16AndSkipStorage) {
    std::vector<float> base =
            clusteredVecs(kProjN, kProjDim, kProjClusters, 0.3f, nullptr, 5);
    std::vector<half> baseHalf(base.size());
    for (size_t i = 0; i < base.size(); i++) {
        baseHalf[i] = __float2half(base[i]);
    }
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, kProjDim, faiss::METRIC_L2, projectionConfig());
    gpuIndex.train_ex(kProjN, baseHalf.data(), faiss::NumericType::Float16);

    for (bool skipStorage : {false, true}) {
        faiss::IndexHNSWCagra cpu;
        cpu.base_level_only = true;
        auto params = projectionParams();
        faiss::gpu::CagraUpperLevelProjectionStats stats;
        gpuIndex.copyTo(&cpu, skipStorage, &params, &stats);
        auto rows = level0Rows(cpu);
        EXPECT_EQ(reachableFrom(rows, 0), kProjN)
                << "skip_storage=" << skipStorage;
        EXPECT_GT(stats.added_by_overwrite, 0);
        EXPECT_EQ(stats.in_degree0_after, stats.in_degree0_before);
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
