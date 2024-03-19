//
// Created by Verma, Navneet on 3/14/24.
//

#include "faiss/Index.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/MetaIndexes.h"
#include "faiss/impl/IDSelector.h"
#include <algorithm>
#include <string>
#include <vector>
#include "commons.cpp"



int main() {
    int dim = 4;
    int numVectors = 4;
    std::vector<float> dataset = randVecs(numVectors, dim);
    std::vector<faiss::idx_t> idVector = getIds(numVectors);


    std::string indexDescriptionCpp = "HNSW16,Flat";
    std::unique_ptr<faiss::Index> indexWriter;
    faiss::MetricType metricType = faiss::METRIC_L2;
    indexWriter.reset(faiss::index_factory(dim, indexDescriptionCpp.c_str(), metricType));


    faiss::IndexIDMap idMap = faiss::IndexIDMap(indexWriter.get());
    idMap.add_with_ids(numVectors, dataset.data(), idVector.data());

    std::vector<float> queryVector = {90, 91, 92, 93};
    std::vector<float> dis(3);
    std::vector<faiss::idx_t> ids(3);
    idMap.search(1, queryVector.data(), 3, dis.data(), ids.data());

    // Write the index to disk
    std::string indexPathCpp = "hnsw-index.txt";
    faiss::write_index(&idMap, indexPathCpp.c_str());

    std::cout<<"Index is written to "<<indexPathCpp<<std::endl;
}
