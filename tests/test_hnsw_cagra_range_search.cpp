/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "faiss/IndexHNSW.h"
#include <algorithm>
#include <gtest/gtest.h>

/// Copy the base layer graph from src to a new IndexHNSWCagra (dst)
/// that only has level 0. Modeled after GpuIndexCagra::copyTo().
void copyBaseLevelOnly(
        const faiss::IndexHNSWCagra& src,
        faiss::IndexHNSWCagra& dst) {
    auto n = src.ntotal;
    auto d = src.d;
    auto M = src.hnsw.nb_neighbors(0) / 2;
    auto graph_degree = src.hnsw.nb_neighbors(0);

    // Setup storage
    if (dst.storage && dst.own_fields) {
        delete dst.storage;
    }
    dst.storage = new faiss::IndexFlatL2(d);
    dst.own_fields = true;
    dst.d = d;
    dst.metric_type = src.metric_type;
    dst.is_trained = true;
    dst.keep_max_size_level0 = true;

    // Reset and reconfigure HNSW structure
    dst.hnsw.reset();
    dst.hnsw.assign_probas.clear();
    dst.hnsw.cum_nneighbor_per_level.clear();
    dst.hnsw.set_default_probas(M, 1.0 / log(M));

    // Only build level 0 — no upper levels
    dst.hnsw.prepare_level_tab(n, false);

    // Copy vectors into storage
    auto src_flat = dynamic_cast<faiss::IndexFlat*>(src.storage);
    dst.storage->add(n, src_flat->get_xb());
    dst.ntotal = n;

    // Copy level 0 neighbors from src to dst
#pragma omp parallel for
    for (faiss::idx_t i = 0; i < n; i++) {
        size_t src_begin, src_end;
        src.hnsw.neighbor_range(i, 0, &src_begin, &src_end);

        size_t dst_begin, dst_end;
        dst.hnsw.neighbor_range(i, 0, &dst_begin, &dst_end);

        for (size_t j = 0; j < graph_degree && j < (dst_end - dst_begin); j++) {
            dst.hnsw.neighbors[dst_begin + j] =
                    src.hnsw.neighbors[src_begin + j];
        }
    }

    // Mark as base_level_only
    dst.base_level_only = true;
}

TEST(IndexHNSWCagra, range_searh) {
    int d = 8;    // dimension
    int nb = 10;  // number of vectors to index
    int nq = 1;   // number of queries
    int k = 3;    // number of nearest neighbors
    int M = 4;    // HNSW parameter
    float radius = 0.7f;

    // Generate random vectors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    std::vector<float> xb(nb * d);
    for (auto& v : xb) {
        v = distrib(rng);
    }

    std::vector<float> xq(nq * d);
    for (auto& v : xq) {
        v = distrib(rng);
    }

    // Build index with full HNSW (base_level_only must be false to add)
    faiss::IndexHNSWCagra original_index(d, M, faiss::METRIC_L2);
    original_index.add(nb, xb.data());
    original_index.num_base_level_search_entrypoints = 2;
    faiss::RangeSearchResult original_index_results(nq);
    original_index.range_search(nq, xq.data(), radius, &original_index_results);


    // Range search on copied index
    faiss::IndexHNSWCagra copied_index;
    copyBaseLevelOnly(original_index, copied_index);
    copied_index.num_base_level_search_entrypoints = 2;
    copied_index.base_level_only = true;

    faiss::RangeSearchResult copied_index_results(nq);

    copied_index.range_search(nq, xq.data(), radius, &copied_index_results);

    size_t original_count = original_index_results.lims[nq];
    size_t copied_count = copied_index_results.lims[nq];

    ASSERT_EQ(original_count, copied_count)
            << "Number of results differ. Original: " << original_count
            << " Copied: " << copied_count;

    // Sort labels for each query before comparing (range_search results are unordered)
    for (int i = 0; i < nq; i++) {
        std::sort(original_index_results.labels + original_index_results.lims[i],
                  original_index_results.labels + original_index_results.lims[i + 1]);
        std::sort(copied_index_results.labels + copied_index_results.lims[i],
                  copied_index_results.labels + copied_index_results.lims[i + 1]);
    }

    for (size_t j = 0; j < original_count; j++) {
        ASSERT_EQ(original_index_results.labels[j], copied_index_results.labels[j])
                << "Label mismatch at index " << j
                << ". Original: " << original_index_results.labels[j]
                << " Copied: " << copied_index_results.labels[j];
    }
}