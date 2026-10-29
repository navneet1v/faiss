# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

from faiss.contrib import datasets, evaluation
import numpy as np


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestComputeGT(unittest.TestCase):

    def do_compute_GT(self, metric, numeric_type):
        d = 64
        k = 12

        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            # Normalize for inner product to avoid duplicate neighbors
            if metric == faiss.METRIC_INNER_PRODUCT:
                # Normalize database vectors
                data_base = data_base / np.linalg.norm(data_base, axis=1, keepdims=True)
                # Normalize query vectors
                data_query = data_query / np.linalg.norm(
                    data_query, axis=1, keepdims=True
                )
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        Dref, Iref = faiss.knn(data_query, data_base, k, metric)

        res = faiss.StandardGpuResources()

        # attempt to set custom IVF-PQ params
        cagraIndexConfig = faiss.GpuIndexCagraConfig()
        cagraIndexConfig.graph_degree = 32
        cagraIndexConfig.intermediate_graph_degree = 64
        cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
        cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.5
        cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig
        cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ

        index = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
        index.train(data_base_nt, numeric_type=numeric_type)
        Dnew, Inew = index.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_compute_GT_L2(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float32)

    def test_compute_GT_IP(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_compute_GT_L2_FP16(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Float16)

    @unittest.skip(
        "GPU CAGRA inner-product + FP16 search deadlocks on CUDA 12.9; "
        "root cause TBD."
    )
    def test_compute_GT_IP_FP16(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_compute_GT_L2_Int8(self):
        self.do_compute_GT(faiss.METRIC_L2, faiss.Int8)

    def test_compute_GT_IP_Int8(self):
        self.do_compute_GT(faiss.METRIC_INNER_PRODUCT, faiss.Int8)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestInterop(unittest.TestCase):

    def do_interop(self, metric, numeric_type):
        d = 64
        k = 12
        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        index.train(data_base_nt, numeric_type=numeric_type)
        Dnew, Inew = index.search(data_query_nt, k, numeric_type=numeric_type)

        cpu_index = faiss.index_gpu_to_cpu(index)
        # cpu index always search in fp32
        Dref, Iref = cpu_index.search(data_query, k)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

        deserialized_index = faiss.deserialize_index(faiss.serialize_index(cpu_index))

        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
        Dnew2, Inew2 = gpu_index.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dnew2, Inew2, Dnew, Inew, k)

    def test_interop_L2(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float32)

    def test_interop_IP(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_interop_L2_FP16(self):
        self.do_interop(faiss.METRIC_L2, faiss.Float16)

    def test_interop_IP_FP16(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_interop_L2_Int8(self):
        self.do_interop(faiss.METRIC_L2, faiss.Int8)

    def test_interop_IP_Int8(self):
        self.do_interop(faiss.METRIC_INNER_PRODUCT, faiss.Int8)

    def test_base_level_only_range_search(self):
        d = 32
        nb = 1000
        nq = 10
        ds = datasets.SyntheticDataset(d, 0, nb, nq)
        data_base = ds.get_database()
        data_query = ds.get_queries()

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2)
        index.train(data_base, numeric_type=faiss.Float32)

        cpu_index = faiss.index_gpu_to_cpu(index)
        cpu_index.base_level_only = True
        cpu_index.num_base_level_search_entrypoints = 8

        radius = np.float32(1e9)
        lims, _, _ = cpu_index.range_search(data_query, radius)
        counts = lims[1:] - lims[:-1]
        self.assertTrue(np.all(counts > 0))


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestIDMapCagra(unittest.TestCase):

    def do_IDMapCagra(self, metric, numeric_type):
        d = 64
        k = 12
        if numeric_type == faiss.Int8:
            data_base_nt = np.random.randint(-128, 128, size=(10000, d), dtype=np.int8)
            data_query_nt = np.random.randint(-128, 128, size=(100, d), dtype=np.int8)
            data_base = data_base_nt.astype(np.float32)
            data_query = data_query_nt.astype(np.float32)
        else:
            ds = datasets.SyntheticDataset(d, 0, 10000, 100)
            data_base = ds.get_database()  # fp32
            data_query = ds.get_queries()  # fp32
            if numeric_type == faiss.Float16:
                data_base_nt = data_base.astype(np.float16)
                data_query_nt = data_query.astype(np.float16)
            elif numeric_type == faiss.Float32:
                data_base_nt = data_base
                data_query_nt = data_query

        Dref, Iref = faiss.knn(data_query, data_base, k, metric)

        res = faiss.StandardGpuResources()

        index = faiss.GpuIndexCagra(res, d, metric)
        idMapIndex = faiss.IndexIDMap(index)
        idMapIndex.train(data_base_nt, numeric_type=numeric_type)
        ids = np.array([i for i in range(10000)])
        idMapIndex.add_with_ids(data_base_nt, ids, numeric_type=numeric_type)
        Dnew, Inew = idMapIndex.search(data_query_nt, k, numeric_type=numeric_type)

        evaluation.check_ref_knn_with_draws(Dref, Iref, Dnew, Inew, k)

    def test_IDMapCagra_L2(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float32)

    def test_IDMapCagra_IP(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float32)

    def test_IDMapCagra_L2_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Float16)

    def test_IDMapCagra_IP_FP16(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Float16)

    def test_IDMapCagra_L2_Int8(self):
        self.do_IDMapCagra(faiss.METRIC_L2, faiss.Int8)

    def test_IDMapCagra_IP_Int8(self):
        self.do_IDMapCagra(faiss.METRIC_INNER_PRODUCT, faiss.Int8)


def clustered_vectors(n, d, nclusters, sigma, seed):
    """Well-separated clusters: the exact k-NN graph is disconnected across clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((nclusters, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, nclusters, n)
    noise = rng.standard_normal((n, d)).astype(np.float32) * (sigma / np.sqrt(d))
    return np.ascontiguousarray(centers[labels] + noise, dtype=np.float32)


def level0_rows(cpu_index):
    """Level-0 neighbour lists of an IndexHNSWCagra, shape [ntotal, nb0]."""
    hnsw = cpu_index.hnsw
    n = cpu_index.ntotal
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    nb0 = hnsw.nb_neighbors(0)
    return neighbors[offsets[:n, None] + np.arange(nb0)[None, :]]


def reachable_from(rows, start):
    """Number of nodes reachable from `start` following level-0 out-edges."""
    seen = np.zeros(len(rows), dtype=bool)
    seen[start] = True
    frontier = np.array([start])
    while len(frontier):
        nxt = rows[frontier].ravel()
        nxt = np.unique(nxt[nxt >= 0])
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    return int(seen.sum())


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(), "only if cuVS is compiled in"
)
class TestCagraUpperLevelProjection(unittest.TestCase):
    """GpuIndexCagra.copyTo(cpu_index, skip_storage, CagraUpperLevelProjectionParams)"""

    n, d, nclusters, graph_degree, m = 20000, 64, 100, 32, 8

    def build(self, x, numeric_type=faiss.Float32, store_dataset=False):
        cfg = faiss.GpuIndexCagraConfig()
        cfg.graph_degree = self.graph_degree
        cfg.intermediate_graph_degree = self.graph_degree
        cfg.build_algo = faiss.graph_build_algo_IVF_PQ
        # the projection reads the caller's host array: store_dataset must be False
        cfg.store_dataset = store_dataset
        self.res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexCagra(self.res, self.d, faiss.METRIC_L2, cfg)
        gpu_index.train(x, numeric_type=numeric_type)
        return gpu_index

    def params(self, **kwargs):
        p = faiss.CagraUpperLevelProjectionParams()
        p.M = self.m
        for k, v in kwargs.items():
            setattr(p, k, v)
        return p

    @staticmethod
    def copy(gpu_index, params=None, skip_storage=False, stats=None):
        cpu = faiss.IndexHNSWCagra()
        cpu.base_level_only = True
        if params is None:
            gpu_index.copyTo(cpu)
        else:
            gpu_index.copyTo(cpu, skip_storage, params, stats)
        return cpu

    def test_connects_clustered_graph(self):
        xb = clustered_vectors(self.n + 200, self.d, self.nclusters, 0.3, 1)
        xb, xq = xb[: self.n], xb[self.n:]
        x = np.ascontiguousarray(xb)
        gpu_index = self.build(x)
        plain = self.copy(gpu_index)
        stats = faiss.CagraUpperLevelProjectionStats()
        projected = self.copy(gpu_index, self.params(), stats=stats)

        plain_rows, projected_rows = level0_rows(plain), level0_rows(projected)
        # precondition: the data really fragments the plain CAGRA graph
        self.assertLess(reachable_from(plain_rows, 0), self.n // 2)
        for start in np.random.default_rng(7).integers(0, self.n, 10):
            self.assertEqual(reachable_from(projected_rows, int(start)), self.n)
        self.assertGreater(stats.promoted, 0)
        self.assertGreater(stats.added_by_overwrite, 0)

        # base-level-only search recall@10 vs exact
        k = 10
        _, Iref = faiss.knn(xq, x, k)
        params = faiss.SearchParametersHNSW(efSearch=64)

        def recall(index):
            _, I = index.search(xq, k, params=params)
            return np.mean([len(set(I[i]) & set(Iref[i])) / k for i in range(len(xq))])

        projected_recall = recall(projected)
        self.assertGreaterEqual(projected_recall, 0.9)
        self.assertGreater(projected_recall, recall(plain))

    def test_row_invariants_and_stats(self):
        x = clustered_vectors(self.n, self.d, self.nclusters, 0.3, 2)
        gpu_index = self.build(x)
        plain = self.copy(gpu_index)
        stats = faiss.CagraUpperLevelProjectionStats()
        params = self.params()
        projected = self.copy(gpu_index, params, stats=stats)

        plain_rows, projected_rows = level0_rows(plain), level0_rows(projected)
        nb0 = plain_rows.shape[1]
        cap = int(params.max_share * nb0)
        changed = (plain_rows != projected_rows).sum(axis=1)
        self.assertLessEqual(int(changed.max()), cap)
        self.assertEqual(int(changed.sum()), stats.added_by_overwrite + stats.added_into_empty)
        self.assertTrue(((projected_rows >= -1) & (projected_rows < self.n)).all())
        for i in range(self.n):  # no duplicates are introduced
            row = projected_rows[i][projected_rows[i] >= 0]
            plain_row = plain_rows[i][plain_rows[i] >= 0]
            if len(np.unique(plain_row)) == len(plain_row):
                self.assertEqual(len(np.unique(row)), len(row))

        # promotion is faiss's HNSW level draw for an HNSW with 2 * M links
        draw = faiss.HNSW(2 * self.m)
        draw.assign_probas.clear()
        draw.cum_nneighbor_per_level.clear()
        draw.set_default_probas(2 * self.m, 1.0 / np.log(2 * self.m))
        expected = sum(draw.random_level() >= 1 for _ in range(self.n))
        self.assertEqual(stats.promoted, expected)

        # protect_last_in_edge: no node becomes unreachable
        self.assertEqual(stats.in_degree0_after, stats.in_degree0_before)
        self.assertEqual(stats.rows_not_full, 0)  # CAGRA rows are full

    def test_max_share_zero_and_no_params(self):
        x = clustered_vectors(self.n, self.d, self.nclusters, 0.3, 3)
        gpu_index = self.build(x)
        plain_rows = level0_rows(self.copy(gpu_index))

        no_params = faiss.IndexHNSWCagra()
        no_params.base_level_only = True
        gpu_index.copyTo(no_params, False, None)
        np.testing.assert_array_equal(level0_rows(no_params), plain_rows)

        stats = faiss.CagraUpperLevelProjectionStats()
        zero = self.copy(gpu_index, self.params(max_share=0.0), stats=stats)
        np.testing.assert_array_equal(level0_rows(zero), plain_rows)
        self.assertEqual(stats.added_by_overwrite, 0)
        self.assertGreater(stats.skipped_cap, 0)

    def test_invalid_params_raise(self):
        x = clustered_vectors(5000, self.d, 20, 0.3, 4)
        gpu_index = self.build(x)
        with self.assertRaises(RuntimeError):  # M is required
            self.copy(gpu_index, self.params(M=0))
        with self.assertRaises(RuntimeError):
            self.copy(gpu_index, self.params(max_share=1.5))
        cpu = faiss.IndexHNSWCagra()
        cpu.base_level_only = False  # the projection keeps the index base-level only
        with self.assertRaises(RuntimeError):
            gpu_index.copyTo(cpu, False, self.params())
        storing = self.build(x, store_dataset=True)  # not supported with the projection
        with self.assertRaises(RuntimeError):
            self.copy(storing, self.params())

    def test_float16_and_skip_storage(self):
        x = clustered_vectors(self.n, self.d, self.nclusters, 0.3, 5).astype(np.float16)
        gpu_index = self.build(x, numeric_type=faiss.Float16)
        for skip_storage in (False, True):
            stats = faiss.CagraUpperLevelProjectionStats()
            cpu = self.copy(gpu_index, self.params(), skip_storage=skip_storage, stats=stats)
            self.assertEqual(reachable_from(level0_rows(cpu), 0), self.n)
            self.assertGreater(stats.added_by_overwrite, 0)
            self.assertEqual(stats.in_degree0_after, stats.in_degree0_before)
