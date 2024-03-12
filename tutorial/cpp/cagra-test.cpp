//
// Created by Verma, Navneet on 3/11/24.
//

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <cstddef>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <optional>
#include <vector>
#include "faiss/IndexIDMap.h"

#include <raft/core/resource/cuda_stream.hpp>
#include "faiss/IndexHNSW.h"
#include "faiss/index_io.h"
#include <iostream>

long s_seed = 1;

struct Options {
    Options() {
        //numTrain = 2 * faiss::gpu::randVal(2000, 5000);
        numTrain = 2 * 2000;
        //dim = faiss::gpu::randVal(4, 10);
        dim = 10;

        //graphDegree = faiss::gpu::randSelect({32, 64});
        graphDegree = 32;
        intermediateGraphDegree = 64;
        /*buildAlgo = faiss::gpu::randSelect(
                {faiss::gpu::graph_build_algo::IVF_PQ,
                 faiss::gpu::graph_build_algo::NN_DESCENT});*/
        buildAlgo = faiss::gpu::graph_build_algo::NN_DESCENT;

        numQuery = 100;
        k = 10;

        device = faiss::gpu::getNumDevices() - 1;
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
    int dim;
    size_t graphDegree;
    size_t intermediateGraphDegree;
    faiss::gpu::graph_build_algo buildAlgo;
    int numQuery;
    int k;
    int device;
};


std::vector<float> randVecs(size_t num, size_t dim) {
    std::vector<float> v(num * dim);

    faiss::float_rand(v.data(), v.size(), s_seed);
    // unfortunately we generate separate sets of vectors, and don't
    // want the same values
    ++s_seed;

    return v;
}

int main() {
    Options opt;
    std::vector<float> trainVecs = randVecs(opt.numTrain, opt.dim);

    // train gpu index
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexCagraConfig config;
    config.device = opt.device;
    config.graph_degree = opt.graphDegree;
    config.intermediate_graph_degree = opt.intermediateGraphDegree;
    config.build_algo = opt.buildAlgo;

    faiss::gpu::GpuIndexCagra gpuIndex(
            &res, opt.dim, faiss::METRIC_L2, config);
    faiss::IndexIDMap idMapIndex = faiss::IndexIDMap(&gpuIndex);

    std::vector<faiss::idx_t> ids;
    for(int i = 0 ; i < opt.numTrain; i++) {
        ids.push_back(opt.numTrain - i);
    }

    idMapIndex.add_with_ids(opt.numTrain, trainVecs.data(), ids.data());

    //gpuIndex.train(opt.numTrain, trainVecs.data());

    faiss::IndexHNSWCagra cpuCagraIndex;

    faiss::gpu::GpuIndexCagra *mappedGpuIndex = dynamic_cast<faiss::gpu::GpuIndexCagra*>(idMapIndex.index);
    mappedGpuIndex->copyTo(&cpuCagraIndex);
    idMapIndex.index = &cpuCagraIndex;

    auto *indexToBeWritten = dynamic_cast<faiss::Index*>(&idMapIndex);
    faiss::write_index(indexToBeWritten, "/tmp/cagraindex.txt");
    std::cout<<"index is written"<<std::endl;
    return 0;
}

/*
working code
Options opt;

std::vector<float> trainVecs = randVecs(opt.numTrain, opt.dim);

// train cpu index
//    faiss::IndexHNSWFlat cpuIndex(opt.dim, opt.graphDegree / 2);
//    cpuIndex.hnsw.efConstruction = opt.k * 2;
//    cpuIndex.add(opt.numTrain, trainVecs.data());

// train gpu index
faiss::gpu::StandardGpuResources res;
res.noTempMemory();

faiss::gpu::GpuIndexCagraConfig config;
config.device = opt.device;
config.graph_degree = opt.graphDegree;
config.intermediate_graph_degree = opt.intermediateGraphDegree;
config.build_algo = opt.buildAlgo;

faiss::gpu::GpuIndexCagra gpuIndex(
        &res, opt.dim, faiss::METRIC_L2, config);
faiss::IndexIDMap idMapIndex = faiss::IndexIDMap(&gpuIndex);

std::vector<faiss::idx_t> ids;
for(int i = 0 ; i < opt.numTrain; i++) {
    ids.push_back(opt.numTrain - i);
}

idMapIndex.add_with_ids(opt.numTrain, trainVecs.data(), ids.data());

//gpuIndex.train(opt.numTrain, trainVecs.data());

faiss::IndexHNSWCagra cpuCagraIndex;

faiss::gpu::GpuIndexCagra *mappedGpuIndex = dynamic_cast<faiss::gpu::GpuIndexCagra*>(idMapIndex.index);


mappedGpuIndex->copyTo(&cpuCagraIndex);

idMapIndex.index = &cpuCagraIndex;

// Type cast the Index pointer for writing data in file.
std::unique_ptr<faiss::Index> indexWriter;
indexWriter.reset(&idMapIndex);

faiss::write_index(indexWriter.get(), "/tmp/cagraindex.txt");
std::cout<<"index is written"<<std::endl;
return 0;
 */


// query
/*auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);

std::vector<float> refDistance(opt.numQuery * opt.k, 0);
std::vector<faiss::idx_t> refIndices(opt.numQuery * opt.k, -1);

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
        testIndices.data());*/
