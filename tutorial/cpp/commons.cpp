//
// Created by Verma, Navneet on 3/14/24.
//

#include <iostream>
#include <vector>
#include <faiss/Index.h>
#include "faiss/index_io.h"
#include "faiss/impl/io.h"
#include <faiss/IndexHNSW.h>
#include "faiss/impl/index_read.cpp"

long s_seed = 1;

std::vector<float> randVecs(size_t num, size_t dim) {
    std::vector<float> v(num * dim);

    faiss::float_rand(v.data(), v.size(), s_seed);
    // unfortunately we generate separate sets of vectors, and don't
    // want the same values
    ++s_seed;

    return v;
}

std::vector<faiss::idx_t> getIds(size_t num) {
    std::vector<faiss::idx_t> ids;
    for(int i = 0 ; i < num; i++) {
        ids.push_back(i);
    }
    return ids;
}