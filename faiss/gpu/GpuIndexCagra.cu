// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <faiss/IndexHNSW.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <faiss/gpu/impl/CuvsCagra.cuh>
#include <memory>
#include <optional>
#include <type_traits>

namespace faiss {
namespace gpu {

namespace {

inline float to_float(float v) {
    return v;
}
inline float to_float(half v) {
    return __half2float(v);
}
inline float to_float(int8_t v) {
    return static_cast<float>(v);
}

/// Copies rows `ids` of a host [n, d] dataset into a host float array
/// [ids.size(), d]. The dataset must be in host memory: with
/// store_dataset = false (the only configuration the OpenSearch
/// remote-vector-index-builder uses) CuvsCagra keeps a pointer to the caller's
/// host array, which is still alive while copyTo runs.
template <typename T>
std::vector<float> gather_rows_as_float(
        const T* data,
        int d,
        const std::vector<idx_t>& ids) {
    FAISS_THROW_IF_NOT_MSG(
            getDeviceForAddress(data) < 0,
            "GpuIndexCagra::copyTo: upper-level projection needs the dataset "
            "in host memory");
    std::vector<float> out(ids.size() * d);
    for (size_t i = 0; i < ids.size(); i++) {
        const T* row = data + ids[i] * d;
        for (int j = 0; j < d; j++) {
            out[i * d + j] = to_float(row[j]);
        }
    }
    return out;
}

/// Draws every node's level with faiss's HNSW level generator (seed 12345, in
/// id order, as HNSW::prepare_level_tab does) for an HNSW with M links per
/// upper-level node, i.e. promotion probability 1 / M, and returns the nodes
/// at level >= 1.
std::vector<idx_t> draw_promoted(idx_t n, int M) {
    HNSW draw(M);
    draw.assign_probas.clear();
    draw.cum_nneighbor_per_level.clear();
    draw.set_default_probas(M, 1.0f / std::log(M));
    std::vector<idx_t> promoted;
    for (idx_t i = 0; i < n; i++) {
        if (draw.random_level() >= 1) {
            promoted.push_back(i);
        }
    }
    return promoted;
}

/// Builds layer 1: a temporary single-level IndexHNSWFlat over the promoted
/// vectors `xp` (every node preset to level 0 with M slots, so no levels above
/// it are created), using faiss's own HNSW insertion. Only the graph is kept:
/// `xp` and the index's flat storage are freed before returning.
std::unique_ptr<IndexHNSWFlat> build_layer1(
        std::vector<float>& xp,
        int d,
        faiss::MetricType metric,
        int M,
        int ef_construction) {
    const idx_t np = xp.size() / d;
    auto up = std::make_unique<IndexHNSWFlat>(d, M, metric);
    up->hnsw.assign_probas.clear();
    up->hnsw.cum_nneighbor_per_level.clear();
    up->hnsw.cum_nneighbor_per_level.push_back(0);
    up->hnsw.cum_nneighbor_per_level.push_back(M);
    up->hnsw.levels.assign(np, 1); // stored level = max level + 1
    up->hnsw.efConstruction = ef_construction;
    up->add(np, xp.data()); // levels are preset, so IndexHNSW::add uses them
    std::vector<float>().swap(xp);
    delete up->storage;
    up->storage = nullptr;
    up->own_fields = false;
    return up;
}

/// Directed links to add to level 0, one row per promoted node (CSR layout):
/// the targets of promoted[r] are links[row_start[r] .. row_start[r + 1]).
/// Both ends of every layer-1 link are promoted, so only promoted rows change.
struct RowLinks {
    std::vector<size_t> row_start;
    std::vector<HNSW::storage_idx_t> links;
};

/// Turns every layer-1 link i-j (positions in `promoted`) into i->j and j->i,
/// grouped by source with a stable counting sort, so each row keeps the order
/// a sequential pass over the promoted nodes would use. Targets are global ids.
RowLinks group_links_by_row(
        const HNSW& layer1,
        const std::vector<idx_t>& promoted) {
    const size_t np = promoted.size();
    std::vector<HNSW::storage_idx_t> src, dst; // positions in `promoted`
    for (size_t i = 0; i < np; i++) {
        size_t b, e;
        layer1.neighbor_range(i, 0, &b, &e);
        for (size_t k = b; k < e; k++) {
            HNSW::storage_idx_t j = layer1.neighbors[k];
            if (j < 0 || size_t(j) == i) {
                continue;
            }
            src.push_back(i);
            dst.push_back(j);
            src.push_back(j);
            dst.push_back(i);
        }
    }
    RowLinks out;
    out.row_start.assign(np + 1, 0);
    for (auto r : src) {
        out.row_start[r + 1]++;
    }
    for (size_t r = 0; r < np; r++) {
        out.row_start[r + 1] += out.row_start[r];
    }
    out.links.resize(src.size());
    std::vector<size_t> cursor(out.row_start.begin(), out.row_start.end() - 1);
    for (size_t k = 0; k < src.size(); k++) {
        out.links[cursor[src[k]]++] = promoted[dst[k]];
    }
    return out;
}

using InDegrees = std::unique_ptr<std::atomic<int32_t>[]>;

/// Level-0 in-degree of every node (atomics, shared between rows).
InDegrees count_in_degrees(const HNSW& h, idx_t n) {
    InDegrees indeg(new std::atomic<int32_t>[n]);
    for (idx_t i = 0; i < n; i++) {
        indeg[i].store(0, std::memory_order_relaxed);
    }
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        size_t b, e;
        h.neighbor_range(i, 0, &b, &e);
        for (size_t j = b; j < e; j++) {
            if (h.neighbors[j] >= 0) {
                indeg[h.neighbors[j]].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return indeg;
}

idx_t count_zero_in_degree(const InDegrees& indeg, idx_t n) {
    idx_t zero = 0;
#pragma omp parallel for reduction(+ : zero)
    for (idx_t i = 0; i < n; i++) {
        zero += indeg[i].load(std::memory_order_relaxed) == 0;
    }
    return zero;
}

/// Removes one in-edge of a node; with protect_last_in_edge, refuses to remove
/// the last one (compare-and-swap, so concurrent rows cannot orphan it).
inline bool release_in_edge(std::atomic<int32_t>& indeg, bool protect) {
    if (!protect) {
        indeg.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    int32_t cur = indeg.load(std::memory_order_relaxed);
    while (cur > 1) {
        if (indeg.compare_exchange_weak(
                    cur, cur - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

/// Per-thread projection counters, merged with an OpenMP reduction.
struct ProjectionCounters {
    size_t added_into_empty = 0;
    size_t added_by_overwrite = 0;
    size_t already_present = 0;
    size_t skipped_cap = 0;
    size_t protected_slots_skipped = 0;
    size_t skipped_no_free_slot = 0;
    size_t rows_not_full = 0;

    ProjectionCounters& operator+=(const ProjectionCounters& o) {
        added_into_empty += o.added_into_empty;
        added_by_overwrite += o.added_by_overwrite;
        already_present += o.already_present;
        skipped_cap += o.skipped_cap;
        protected_slots_skipped += o.protected_slots_skipped;
        skipped_no_free_slot += o.skipped_no_free_slot;
        rows_not_full += o.rows_not_full;
        return *this;
    }
};

#pragma omp declare reduction( \
                merge_counters:ProjectionCounters : omp_out += omp_in)

/// Adds links[begin, end) to one level-0 row: skips links already present,
/// fills empty slots first, then overwrites the row's tail (CAGRA's
/// lowest-ranked links) up to `cap` times (no cap for rows that are not full
/// to begin with), never removing a node's last in-edge when `protect`.
inline void add_links_to_row(
        HNSW::storage_idx_t* row,
        int nb0,
        int cap,
        const HNSW::storage_idx_t* begin,
        const HNSW::storage_idx_t* end,
        std::atomic<int32_t>* indeg,
        bool protect,
        ProjectionCounters& c) {
    int row_cap = cap;
    for (int j = 0; j < nb0; j++) {
        if (row[j] < 0) {
            row_cap = nb0;
            c.rows_not_full++;
            break;
        }
    }
    int replaced = 0;
    int tail = nb0 - 1; // next tail slot to try overwriting
    for (const HNSW::storage_idx_t* it = begin; it != end; ++it) {
        const HNSW::storage_idx_t b = *it;
        int empty = -1;
        bool present = false;
        for (int j = 0; j < nb0; j++) {
            if (row[j] == b) {
                present = true;
                break;
            }
            if (row[j] < 0 && empty < 0) {
                empty = j;
            }
        }
        if (present) {
            c.already_present++;
            continue;
        }
        if (empty >= 0) {
            row[empty] = b;
            indeg[b].fetch_add(1, std::memory_order_relaxed);
            c.added_into_empty++;
            continue;
        }
        if (replaced >= row_cap) {
            c.skipped_cap++;
            continue;
        }
        // this slot may hold its target's last in-edge: then keep it
        while (tail >= 0 && !release_in_edge(indeg[row[tail]], protect)) {
            tail--;
            c.protected_slots_skipped++;
        }
        if (tail < 0) {
            c.skipped_no_free_slot++;
            continue;
        }
        row[tail] = b;
        indeg[b].fetch_add(1, std::memory_order_relaxed);
        tail--;
        replaced++;
        c.added_by_overwrite++;
    }
}

/// Adds the grouped links to the level-0 rows of the promoted nodes of `h`, in
/// parallel: one thread per row, so row writes never race; in-degrees are
/// shared atomics. Promoted nodes come from a random draw and are spread
/// evenly over ids, so a static split is balanced.
ProjectionCounters add_links(
        HNSW& h,
        const std::vector<idx_t>& promoted,
        const RowLinks& links,
        InDegrees& indeg,
        float max_share,
        bool protect) {
    size_t begin0, end0;
    h.neighbor_range(0, 0, &begin0, &end0);
    const int nb0 = static_cast<int>(end0 - begin0);
    const int cap = static_cast<int>(max_share * nb0);
    ProjectionCounters total;
#pragma omp parallel for schedule(static) reduction(merge_counters : total)
    for (size_t r = 0; r < promoted.size(); r++) {
        if (links.row_start[r] == links.row_start[r + 1]) {
            continue;
        }
        size_t rb, re;
        h.neighbor_range(promoted[r], 0, &rb, &re);
        add_links_to_row(
                h.neighbors.data() + rb,
                nb0,
                cap,
                links.links.data() + links.row_start[r],
                links.links.data() + links.row_start[r + 1],
                indeg.get(),
                protect,
                total);
    }
    return total;
}

/// Builds HNSW layer 1 over the promoted vectors `xp` (released as soon as it
/// is built) and projects its links into the level-0 lists of `index`.
void project_upper_level(
        faiss::IndexHNSWCagra* index,
        const std::vector<idx_t>& promoted,
        std::vector<float>& xp,
        int d,
        faiss::MetricType metric,
        int M,
        int ef_construction,
        float max_share,
        bool protect_last_in_edge,
        CagraUpperLevelProjectionStats* stats_out) {
    CagraUpperLevelProjectionStats stats;
    stats.promoted = promoted.size();
    const idx_t n = index->ntotal;
    if (promoted.size() >= 2) {
        double t0 = getmillisecs();
        std::unique_ptr<IndexHNSWFlat> layer1 =
                build_layer1(xp, d, metric, M, ef_construction);
        stats.layer1_build_ms = getmillisecs() - t0;

        t0 = getmillisecs();
        RowLinks links = group_links_by_row(layer1->hnsw, promoted);
        layer1.reset();
        InDegrees indeg = count_in_degrees(index->hnsw, n);
        stats.in_degree0_before = count_zero_in_degree(indeg, n);
        ProjectionCounters c = add_links(
                index->hnsw,
                promoted,
                links,
                indeg,
                max_share,
                protect_last_in_edge);
        stats.in_degree0_after = count_zero_in_degree(indeg, n);
        stats.projection_ms = getmillisecs() - t0;

        stats.added_into_empty = c.added_into_empty;
        stats.added_by_overwrite = c.added_by_overwrite;
        stats.already_present = c.already_present;
        stats.skipped_cap = c.skipped_cap;
        stats.protected_slots_skipped = c.protected_slots_skipped;
        stats.skipped_no_free_slot = c.skipped_no_free_slot;
        stats.rows_not_full = c.rows_not_full;
    }
    if (stats_out) {
        *stats_out = stats;
    }
}

} // namespace

GpuIndexCagra::GpuIndexCagra(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexCagraConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          cagraConfig_(config) {
    this->is_trained = false;
}

void GpuIndexCagra::train_ex(idx_t n, const void* x, NumericType numeric_type) {
    numeric_type_ = numeric_type;
    bool index_is_initialized = !std::holds_alternative<std::monostate>(index_);

    DeviceScope scope(config_.device);
    if (this->is_trained) {
        FAISS_ASSERT(index_is_initialized);
        return;
    }

    // CuvsCagra not initialized
    FAISS_ASSERT(!index_is_initialized);

    std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_params =
            std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::search_params> ivf_pq_search_params =
            std::nullopt;
    if (cagraConfig_.ivf_pq_params != nullptr) {
        ivf_pq_params =
                std::make_optional<cuvs::neighbors::ivf_pq::index_params>();
        ivf_pq_params->n_lists = cagraConfig_.ivf_pq_params->n_lists;
        ivf_pq_params->kmeans_n_iters =
                cagraConfig_.ivf_pq_params->kmeans_n_iters;
        ivf_pq_params->kmeans_trainset_fraction =
                cagraConfig_.ivf_pq_params->kmeans_trainset_fraction;
        ivf_pq_params->pq_bits = cagraConfig_.ivf_pq_params->pq_bits;
        ivf_pq_params->pq_dim = cagraConfig_.ivf_pq_params->pq_dim;
        ivf_pq_params->codebook_kind =
                static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(
                        cagraConfig_.ivf_pq_params->codebook_kind);
        ivf_pq_params->force_random_rotation =
                cagraConfig_.ivf_pq_params->force_random_rotation;
        ivf_pq_params->conservative_memory_allocation =
                cagraConfig_.ivf_pq_params->conservative_memory_allocation;
    }
    if (cagraConfig_.ivf_pq_search_params != nullptr) {
        ivf_pq_search_params =
                std::make_optional<cuvs::neighbors::ivf_pq::search_params>();
        ivf_pq_search_params->n_probes =
                cagraConfig_.ivf_pq_search_params->n_probes;
        ivf_pq_search_params->lut_dtype =
                cagraConfig_.ivf_pq_search_params->lut_dtype;
        ivf_pq_search_params->preferred_shmem_carveout =
                cagraConfig_.ivf_pq_search_params->preferred_shmem_carveout;
        ivf_pq_search_params->max_internal_batch_size =
                cagraConfig_.ivf_pq_search_params->max_internal_batch_size;
    }

    if (numeric_type == NumericType::Float32) {
        index_ = std::make_shared<CuvsCagra<float>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<float>>>(index_)->train(
                n, static_cast<const float*>(x));
    } else if (numeric_type == NumericType::Float16) {
        index_ = std::make_shared<CuvsCagra<half>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<half>>>(index_)->train(
                n, static_cast<const half*>(x));
    } else if (numeric_type == NumericType::Int8) {
        index_ = std::make_shared<CuvsCagra<int8_t>>(
                this->resources_.get(),
                this->d,
                cagraConfig_.intermediate_graph_degree,
                cagraConfig_.graph_degree,
                static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
                cagraConfig_.nn_descent_niter,
                cagraConfig_.store_dataset,
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT,
                ivf_pq_params,
                ivf_pq_search_params,
                cagraConfig_.refine_rate,
                cagraConfig_.guarantee_connectivity);
        std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)->train(
                n, static_cast<const int8_t*>(x));
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::train unsupported data type");
    }

    this->is_trained = true;
    this->ntotal = n;
}

void GpuIndexCagra::train(idx_t n, const float* x) {
    train_ex(n, static_cast<const void*>(x), NumericType::Float32);
}

void GpuIndexCagra::add_ex(idx_t n, const void* x, NumericType numeric_type) {
    train_ex(n, x, numeric_type);
}

void GpuIndexCagra::add(idx_t n, const float* x) {
    add_ex(n, x, NumericType::Float32);
}

bool GpuIndexCagra::addImplRequiresIDs_() const {
    return false;
};

void GpuIndexCagra::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_MSG("adding vectors is not supported by GpuIndexCagra.");
};

void GpuIndexCagra::addImpl_ex_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        const idx_t* ids) {
    GpuIndex::addImpl_ex_(n, x, numeric_type, ids);
}

void GpuIndexCagra::searchImpl_ex_(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(
            this->is_trained &&
            !std::holds_alternative<std::monostate>(index_));
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT_MSG(
            numeric_type == numeric_type_,
            "Inconsistent numeric type for train and search");

    SearchParametersCagra* params;
    if (search_params) {
        params = dynamic_cast<SearchParametersCagra*>(
                const_cast<SearchParameters*>(search_params));
    } else {
        params = new SearchParametersCagra{};
    }

    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(const_cast<idx_t*>(labels), {n, k});

    if (numeric_type == NumericType::Float32) {
        Tensor<float, 2, true> queries(
                const_cast<float*>(static_cast<const float*>(x)), {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<float>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);

    } else if (numeric_type == NumericType::Float16) {
        Tensor<half, 2, true> queries(
                const_cast<half*>(static_cast<const half*>(x)), {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<half>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);
    } else if (numeric_type == NumericType::Int8) {
        Tensor<int8_t, 2, true> queries(
                const_cast<int8_t*>(static_cast<const int8_t*>(x)),
                {n, this->d});

        std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)->search(
                queries,
                k,
                outDistances,
                outLabels,
                params->max_queries,
                params->itopk_size,
                params->max_iterations,
                static_cast<faiss::cagra_search_algo>(params->algo),
                params->team_size,
                params->search_width,
                params->min_iterations,
                params->thread_block_size,
                static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
                params->hashmap_min_bitlen,
                params->hashmap_max_fill_rate,
                params->num_random_samplings,
                params->seed,
                params->sel);
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::searchImpl_ unsupported data type");
    }

    if (not search_params) {
        delete params;
    }
}

void GpuIndexCagra::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    searchImpl_ex_(
            n,
            static_cast<const void*>(x),
            NumericType::Float32,
            k,
            distances,
            labels,
            search_params);
}

void GpuIndexCagra::copyFrom_ex(
        const faiss::IndexHNSWCagra* index,
        NumericType numeric_type) {
    FAISS_ASSERT(index);
    numeric_type_ = numeric_type;

    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    auto hnsw = index->hnsw;

    // copy level 0 to a dense knn graph matrix
    std::vector<idx_t> knn_graph;
    knn_graph.resize(index->ntotal * hnsw.nb_neighbors(0));

#pragma omp parallel for
    for (size_t i = 0; i < index->ntotal; ++i) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            // knn_graph.push_back(hnsw.neighbors[j]);
            knn_graph[i * hnsw.nb_neighbors(0) + (j - begin)] =
                    hnsw.neighbors[j];
        }
    }

    if (numeric_type == NumericType::Float32) {
        auto base_index = dynamic_cast<IndexFlat*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = base_index->get_xb();
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        index_ = std::make_shared<CuvsCagra<float>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
    } else if (numeric_type == NumericType::Float16) {
        auto base_index = dynamic_cast<IndexScalarQuantizer*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = reinterpret_cast<half*>(base_index->codes.data());
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        index_ = std::make_shared<CuvsCagra<half>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
    } else if (numeric_type == NumericType::Int8) {
        auto base_index = dynamic_cast<IndexScalarQuantizer*>(index->storage);
        FAISS_ASSERT(base_index);
        auto dataset = (uint8_t*)base_index->codes.data();
        fprintf(stderr,
                "WARNING: GpuIndexCagra::copyFrom uses non-owning CPU storage. "
                "Keep the source IndexHNSWCagra alive for the lifetime of the "
                "GpuIndexCagra.\n");

        int8_t* decoded_train_dataset = new int8_t[index->ntotal * index->d];
        for (int i = 0; i < index->ntotal * this->d; i++) {
            decoded_train_dataset[i] = dataset[i] - 128;
        }

        index_ = std::make_shared<CuvsCagra<int8_t>>(
                this->resources_.get(),
                this->d,
                index->ntotal,
                hnsw.nb_neighbors(0),
                decoded_train_dataset,
                knn_graph.data(),
                this->metric_type,
                this->metric_arg,
                INDICES_64_BIT);
        delete[] decoded_train_dataset;
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::copyFrom unsupported data type");
    }

    this->is_trained = true;
}

void GpuIndexCagra::copyFrom(const faiss::IndexHNSWCagra* index) {
    copyFrom_ex(index, NumericType::Float32);
}

void GpuIndexCagra::copyTo(
        faiss::IndexHNSWCagra* index,
        bool skip_storage) const {
    copyTo(index, skip_storage, nullptr, nullptr);
}

void GpuIndexCagra::copyTo(
        faiss::IndexHNSWCagra* index,
        bool skip_storage,
        const CagraUpperLevelProjectionParams* projection,
        CagraUpperLevelProjectionStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(
            !projection || index->base_level_only,
            "GpuIndexCagra::copyTo: upper-level projection requires "
            "base_level_only = true");
    FAISS_THROW_IF_NOT_MSG(
            !projection || projection->M > 0,
            "GpuIndexCagra::copyTo: upper-level projection needs M > 0 (the "
            "HNSW m of the index)");
    FAISS_THROW_IF_NOT_MSG(
            !projection ||
                    (projection->max_share >= 0 && projection->max_share <= 1),
            "GpuIndexCagra::copyTo: invalid projection parameters");
    // The projection reads the promoted vectors from the caller's host array,
    // which CuvsCagra only references when store_dataset = false. The
    // OpenSearch remote-vector-index-builder never sets store_dataset = true,
    // so that configuration is not supported.
    FAISS_THROW_IF_NOT_MSG(
            !projection || !cagraConfig_.store_dataset,
            "GpuIndexCagra::copyTo: upper-level projection is not supported "
            "with store_dataset = true");
    FAISS_ASSERT(
            !std::holds_alternative<std::monostate>(index_) &&
            this->is_trained && index);
    DeviceScope scope(config_.device);

    //
    // Index information
    //
    GpuIndex::copyTo(index);
    index->hnsw.is_similarity = is_similarity_metric(this->metric_type);
    // This needs to be zeroed out as this implementation adds vectors to the
    // cpuIndex instead of copying fields
    index->ntotal = 0;
    index->set_numeric_type(numeric_type_);

    idx_t graph_degree;

    if (numeric_type_ == NumericType::Float32) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<float>>>(index_)
                               ->get_knngraph_degree();
    } else if (numeric_type_ == NumericType::Float16) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<half>>>(index_)
                               ->get_knngraph_degree();
    } else if (numeric_type_ == NumericType::Int8) {
        graph_degree = std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)
                               ->get_knngraph_degree();
    } else {
        FAISS_THROW_MSG("GpuIndexCagra::copyTo unsupported data type");
    }

    auto M = graph_degree / 2;
    if (index->storage and index->own_fields) {
        delete index->storage;
    }

    // storage depends on numerictype
    if (numeric_type_ == NumericType::Float32) {
        if (this->metric_type == METRIC_L2) {
            index->storage = new IndexFlatL2(index->d);
        } else if (this->metric_type == METRIC_INNER_PRODUCT) {
            index->storage = new IndexFlatIP(index->d);
        }
    } else if (numeric_type_ == NumericType::Float16) {
        auto qtype = ScalarQuantizer::QT_fp16;
        index->storage =
                new IndexScalarQuantizer(index->d, qtype, this->metric_type);
    } else if (numeric_type_ == NumericType::Int8) {
        auto qtype = ScalarQuantizer::QT_8bit_direct_signed;
        index->storage =
                new IndexScalarQuantizer(index->d, qtype, this->metric_type);
    }

    index->own_fields = true;
    index->keep_max_size_level0 = true;
    index->hnsw.reset();
    index->hnsw.assign_probas.clear();
    index->hnsw.cum_nneighbor_per_level.clear();
    index->hnsw.set_default_probas(M, 1.0 / log(M));

    auto n_train = this->ntotal;
    bool allocation = false;

    // turn off as level 0 is copied from CAGRA graph
    index->init_level0 = false;
    if (skip_storage) {
        index->hnsw.prepare_level_tab(n_train, false);
        index->ntotal = n_train;
    } else if (numeric_type_ == NumericType::Float32) {
        float* train_dataset;
        const float* dataset =
                std::get<std::shared_ptr<CuvsCagra<float>>>(index_)
                        ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new float[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<float*>(dataset);
        }

        if (!index->base_level_only) {
            index->add(n_train, train_dataset);
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            index->storage->add(n_train, train_dataset);
            index->ntotal = n_train;
        }
        if (allocation) {
            delete[] train_dataset;
        }
    } else if (numeric_type_ == NumericType::Float16) {
        half* train_dataset;
        const half* dataset = std::get<std::shared_ptr<CuvsCagra<half>>>(index_)
                                      ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new half[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<half*>(dataset);
        }

        if (!index->base_level_only) {
            FAISS_THROW_MSG(
                    "Only base level copy is supported for FP16 types in GpuIndexCagra::copyTo");
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            index->storage->add_sa_codes(
                    n_train, (uint8_t*)train_dataset, nullptr);
            index->ntotal = n_train;
        }

        if (allocation) {
            delete[] train_dataset;
        }
    } else if (numeric_type_ == NumericType::Int8) {
        int8_t* train_dataset;
        const int8_t* dataset =
                std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)
                        ->get_training_dataset();
        if (getDeviceForAddress(dataset) >= 0) {
            train_dataset = new int8_t[n_train * index->d];
            allocation = true;
            raft::copy(
                    train_dataset,
                    dataset,
                    n_train * index->d,
                    this->resources_->getRaftHandleCurrentDevice()
                            .get_stream());
        } else {
            train_dataset = const_cast<int8_t*>(dataset);
        }

        if (!index->base_level_only) {
            FAISS_THROW_MSG(
                    "Only base level copy is supported for Int8 types in GpuIndexCagra::copyTo");
        } else {
            index->hnsw.prepare_level_tab(n_train, false);
            // Directly update train_dataset with encoding of
            // Quantizer8bitDirectSigned
            for (int64_t i = 0; i < ((int64_t)n_train) * index->d; ++i) {
                train_dataset[i] = static_cast<uint8_t>(
                        static_cast<int>(train_dataset[i]) + 128);
            }

            index->storage->add_sa_codes(
                    n_train,
                    reinterpret_cast<uint8_t*>(train_dataset),
                    nullptr);

            index->ntotal = n_train;
        }

        if (allocation) {
            delete[] train_dataset;
        } else {
            // Recover after appending
            for (int64_t i = 0; i < ((int64_t)n_train) * index->d; ++i) {
                train_dataset[i] = static_cast<int8_t>(
                        static_cast<int>(train_dataset[i]) - 128);
            }
        }
    }

    auto graph = get_knngraph();

#pragma omp parallel for
    for (idx_t i = 0; i < n_train; i++) {
        size_t begin, end;
        index->hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            index->hnsw.neighbors[j] = graph[i * graph_degree + (j - begin)];
        }
    }

    if (projection) {
        // layer 1 as in an HNSW with 2m links per node: promotion 1/(2m)
        const int pM = 2 * projection->M;
        const int pef = projection->ef_construction > 0
                ? projection->ef_construction
                : index->hnsw.efConstruction;
        std::vector<idx_t> promoted = draw_promoted(n_train, pM);
        std::vector<float> xp;
        if (numeric_type_ == NumericType::Float32) {
            xp = gather_rows_as_float(
                    std::get<std::shared_ptr<CuvsCagra<float>>>(index_)
                            ->get_training_dataset(),
                    this->d,
                    promoted);
        } else if (numeric_type_ == NumericType::Float16) {
            xp = gather_rows_as_float(
                    std::get<std::shared_ptr<CuvsCagra<half>>>(index_)
                            ->get_training_dataset(),
                    this->d,
                    promoted);
        } else {
            xp = gather_rows_as_float(
                    std::get<std::shared_ptr<CuvsCagra<int8_t>>>(index_)
                            ->get_training_dataset(),
                    this->d,
                    promoted);
        }
        project_upper_level(
                index,
                promoted,
                xp,
                this->d,
                this->metric_type,
                pM,
                pef,
                projection->max_share,
                projection->protect_last_in_edge,
                stats);
    }

    // turn back on to allow new vectors to be added to level 0
    index->init_level0 = true;
}

void GpuIndexCagra::reset() {
    DeviceScope scope(config_.device);

    if (!std::holds_alternative<std::monostate>(index_)) {
        std::visit(
                [](auto& index_ptr) {
                    using IndexPtrT = std::decay_t<decltype(index_ptr)>;
                    if constexpr (std::is_same_v<IndexPtrT, std::monostate>) {
                        FAISS_THROW_MSG(
                                "CuvsCagra not initialized when calling GpuIndexCagra::reset");
                    } else {
                        return index_ptr->reset();
                    }
                },
                index_);
        this->ntotal = 0;
        this->is_trained = false;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

std::vector<idx_t> GpuIndexCagra::get_knngraph() const {
    FAISS_ASSERT(
            !std::holds_alternative<std::monostate>(index_) &&
            this->is_trained);

    return std::visit(
            [](auto&& index_ptr) -> std::vector<idx_t> {
                using IndexPtrT = std::decay_t<decltype(index_ptr)>;

                if constexpr (std::is_same_v<IndexPtrT, std::monostate>) {
                    FAISS_THROW_MSG(
                            "CuvsCagra not initialized when calling GpuIndexCagra::get_knngraph");
                } else {
                    return index_ptr->get_knngraph();
                }
            },
            index_);
}

faiss::NumericType GpuIndexCagra::get_numeric_type() const {
    return numeric_type_;
}

} // namespace gpu
} // namespace faiss
