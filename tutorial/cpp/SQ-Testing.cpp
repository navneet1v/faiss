//
// Created by Verma, Navneet on 3/10/25.
//
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c_api/faiss_c.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <random>

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_factory.h>

using idx_t = faiss::idx_t;
 
int main() {
    // SQ: HNSW8,SQfp16
    int d = 128;      // dimension
    int nb = 1000; // database size
    int nq = 1;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    faiss_idx_t* ids = new faiss_idx_t[nb];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
        ids[i] = i;
    }

    // for (int i = 0; i < nq; i++) {
    //     for (int j = 0; j < d; j++)
    //         xq[d * i + j] = distrib(rng);
    //     xq[d * i] += i / 1000.;
    // }
    // printf("Printing first vector of the dataset ");
    // for(int i = 0 ; i < d ; i ++ ) {
    //     printf("%f ", xb[i]);
    // }

    printf("\n");

    int k = 4;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra cagraIndex(&res, d);
    std::vector<__half> xb_half(nb);

    for (size_t i = 0; i < nb; ++i) {
        xb_half[i] = __float2half(xb[i]);
    }

    cagraIndex.train(nb, static_cast<void*>(xb_half.data()), faiss::NumericType::Float16);

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        for(int i = 0 ; i < nq * d ; i ++) {
            std::cout<<xq[i];
        }
        //cagraIndex.search(nq, xq, k, D, I);

        printf("I= ");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D= ");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5f ", D[i * k + j]);
            printf("\n");
        }

        printf("Number of distances computed %lu\n", faiss::hnsw_stats.ndis);
        printf("Number of edges Traversed %lu\n", faiss::hnsw_stats.nhops);

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
 