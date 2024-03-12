//
// Created by Verma, Navneet on 3/13/24.
//

#include <iostream>
#include <vector>
#include <faiss/Index.h>
#include "faiss/index_io.h"
#include "faiss/impl/io.h"
#include <faiss/IndexHNSW.h>
#include "faiss/impl/index_read.cpp"

using namespace std;


long s_seed = 1;
std::vector<float> randVecs(size_t num, size_t dim) {
    std::vector<float> v(num * dim);

    faiss::float_rand(v.data(), v.size(), s_seed);
    // unfortunately we generate separate sets of vectors, and don't
    // want the same values
    ++s_seed;

    return v;
}

int main() {
    int dim = 10;
    int k = 10;
    std::vector<float> dis(k);
    std::vector<faiss::idx_t> ids(k);

    faiss::Index* indexReader = faiss::read_index("/Volumes/workplace/faiss/tutorial/cpp/cagraindex.txt");

    auto *index = reinterpret_cast<faiss::IndexIDMap *>(indexReader);
    vector<float> queryVector = randVecs(1, dim);
    index->search(1, queryVector.data(), k, dis.data(), ids.data());

    for(int i = 0 ; i < ids.size(); i++) {
        cout<<"Ids are : "<<ids[i]<<endl;
    }

    cout<<"End of Cagra CPU tests."<<endl;
    return 0;
}
