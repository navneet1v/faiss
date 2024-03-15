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
    int dim = 50;
    int k = 2;
    std::vector<float> dis(k);
    std::vector<faiss::idx_t> ids(k);

    faiss::Index* indexReader = faiss::read_index("/Volumes/workplace/faiss/tutorial/cpp/cagraindex-test-with-ids.txt");

    auto *index = reinterpret_cast<faiss::IndexIDMap *>(indexReader);
    vector<float> queryVector = {90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139};
    index->search(1, queryVector.data(), k, dis.data(), ids.data());

    for(int i = 0 ; i < ids.size(); i++) {
        cout<<"Ids are : "<<ids[i]<<endl;
    }

    cout<<"End of Cagra CPU tests with ids."<<endl;
    return 0;
}
