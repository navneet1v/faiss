//
// Created by Verma, Navneet on 5/2/23.
//

#include <cstdio>
#include <iostream>
#include <bitset>
#include <random>

#include <faiss/IndexHNSW.h>
#include "faiss/index_factory.h"
#include <faiss/utils/utils.h>
#include <faiss/impl/HNSW.h>
#include <faiss/IndexIDMap.h>
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetaIndexes.h"
#include "faiss/Index.h"
#include "faiss/impl/IDSelector.h"
#include <memory>
// 64-bit int
using idx_t = faiss::idx_t;

using namespace faiss;
using namespace std;

std::unique_ptr<IDSelector> getIdSelector(idx_t* ids) {
    std::unique_ptr<IDSelectorArray> sel;
    sel.reset(new IDSelectorArray(3, ids));
    return sel;
}


faiss::SearchParameters* createSearchParams(IDSelector* sel) {
//    std::shared_ptr<IDSelectorArray> sel(
//            new IDSelectorArray(ids.size(), ids.data()));
    // IDSelectorArray sel(ids.size(), ids.data());
    static faiss::SearchParametersHNSW searchParametersHnsw;
    searchParametersHnsw.sel = sel;
    return &searchParametersHnsw;
}

void myParam() {
    cout<<"I am in search params"<<endl;
    static faiss::SearchParametersHNSW searchParametersHnsw;
    cout<<"I am returing from search params"<<endl;
}

void myParam_caller() {
    myParam();
}

struct IDSelectorBitmapImproved : IDSelector {
    int n;
    const uint8_t* bitmap;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap id will be selected iff id / 8 < n and bit number
     *               (i%8) of bitmap[floor(i / 8)] is 1.
     */
    IDSelectorBitmapImproved(int n, const uint8_t* bitmap): n(n), bitmap(bitmap) {}
    bool is_member(idx_t id) const final;
    ~IDSelectorBitmapImproved() override {}
};
bool IDSelectorBitmapImproved::is_member(idx_t id) const {
    uint64_t i = id;
    if ((i >> 3) >= n) {
        return false;
    }
    return (bitmap[i >> 3] >> (i & 7)) & 1;
}

// NOt working because of the cast exception from HNSW
/*faiss::SearchParameters createSearchParamsV2(IDSelector* sel) {
    //    std::shared_ptr<IDSelectorArray> sel(
    //            new IDSelectorArray(ids.size(), ids.data()));
    // IDSelectorArray sel(ids.size(), ids.data());
    faiss::SearchParametersHNSW searchParametersHnsw;
    searchParametersHnsw.sel = sel;
    return searchParametersHnsw;
}*/

std::unique_ptr<faiss::SearchParameters> createSearchParamsV2(IDSelector* sel) {
    //    std::shared_ptr<IDSelectorArray> sel(
    //            new IDSelectorArray(ids.size(), ids.data()));
    // IDSelectorArray sel(ids.size(), ids.data());
    std::unique_ptr<faiss::SearchParametersHNSW> searchParametersHnsw;
    searchParametersHnsw.reset(new SearchParametersHNSW());
    searchParametersHnsw.get()->sel = sel;
    return searchParametersHnsw;
}

int main() {

    vector<uint8_t> v;
    cout<<"Hello Size is "<<v.max_size()<<endl;
    cout<<"Size is "<<v.max_size()<<endl;


    vector<int> filterIds;
    filterIds.push_back(3);
    filterIds.push_back(250057);
    int maxIdValue = filterIds[filterIds.size()-1];
    const int bitsetArraySize = (maxIdValue / 8) + 1;
    cout<<"Bitset size is :"<<bitsetArraySize<<endl;
    static std::vector<uint8_t> bitsetVector(bitsetArraySize, 0);
    faiss::read_index("", faiss::IO_FLAG_READ_ONLY);
    /**
     * Coming from Faiss IDSelectorBitmap::is_member function bitmap id will be selected
     * iff id / 8 < n and bit number (i%8) of bitmap[floor(i / 8)] is 1.
     */
    for(int i = 0 ; i < 2 ; i ++) {
        int value = filterIds[i];
        cout<<"Value is "<<value<<endl;
        int bitsetArrayIndex = floor(value/8);
        cout<<"bitsetArrayIndex is "<<bitsetArrayIndex<<endl;
        // this set the bit at required position
        bitsetVector[bitsetArrayIndex] |= 1 << (value % 8);
    }

    myParam_caller();
    vector<int> myvec(1000);

    cout<<"size is " << myvec.size();

    myvec.shrink_to_fit();
    cout<<"\nsize is " << myvec.size()<<endl;

    bitset<8> bits;
    bits.set(7);
    cout<<bits.to_string();



    /*0 => 692
     bitSet.words[33554431] => 4611686018427387904

     {2, 4, 5, 7, 9, 2147483646}

    */
    // Long max value in java 9223372036854775807

    long jLongArray[2] = {0};

    jLongArray[0] = 692;
    jLongArray[1] = 4611686018427387904;





//    bitset<8> mybits;
//    mybits.set(0);
//    cout<<mybits.to_string()<<endl;
//
//
//
//
//    long filterIds[] = {692};
//    long m = 5;
//    cout<<sizeof(m)<<endl; // 4 bit
//    bitset<8> bits;
    vector<int> myIds;
    for(int i=0 ; i < 2; i++) {
        long value = jLongArray[i]; // 8 bit
        std::cout<<"Size of the long is : "<<sizeof(long)<<std::endl;
        if(value != 0) {
            long long mask = 1;
            for (int j = 0; j < sizeof(value) * 8 -1; j++) {
                cout << "mask is : " << (mask << j) << endl;
                if ((mask << j) & value) {
                    myIds.push_back(j);
                    cout << "j is : " << j + 1 << endl;
                    // idsBitset.set(j+1);
                }
            }
        }
    }

    std::cout<<"Printing Ids \n";
    for(int j = 0 ;j < myIds.size(); j++){
        std::cout<<"Ids is "<<myIds[j]<<endl;
    }
    std::cout<<"Printing Ids Completed\n";





    int d = 3;      // dimension
    int nb = 5; // database size
    int nq = 3;  // nb of queries

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = (float)(d * i + j);
        xb[d * i] += i / 1000;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = j;
        xq[d * i] += i / 1000.;
    }

    printf("XB is : \n");
    for(int i=0; i<nb; i++) {
        for (int j = 0; j < d; j++)
            printf("%f ", xb[d * i + j]);
        printf("\n");
    }

    printf("XQ is : \n");
    for(int i=0; i<nq; i++) {
        for (int j = 0; j < d; j++)
            printf("%f ", xq[d * i + j]);
        printf("\n");
    }

   // faiss::IndexHNSWFlat index(d, 16); // call constructor

    std::unique_ptr<faiss::Index> indexWriter;
    // HNSW Flat
    indexWriter.reset(faiss::index_factory(d, "HNSW16,Flat"));
    //faiss::IndexHNSW index(indexWriter.get(), 16);

    //index.add(nb, xb); // add vectors to the index

    //printf("ntotal = %zd\n", index.ntotal);

    faiss::IndexIDMap index = faiss::IndexIDMap(indexWriter.get());



    vector<idx_t> idVector;
    for(int i=0 ;i < nb; i++){
        idVector.push_back(i);
    }
    index.add_with_ids(nb, xb, idVector.data());

    // These are all inserted ids.
    std::vector<idx_t> index_id_map =  index.id_map;
    map<idx_t,idx_t> external_to_internal;
    for(int i = 0; i < index_id_map.size(); i++) {
        external_to_internal[index_id_map[i]] = i;
    }

    //idx_t* filteredIds = static_cast<idx_t*>(malloc(3 * sizeof(idx_t*)));

    unique_ptr<idx_t> filteredIds(static_cast<idx_t*>(malloc(3 * sizeof(idx_t*))));


//    filteredIds.push_back(1);
//    filteredIds.push_back(4);
//    filteredIds.push_back(3);
    filteredIds.get()[0] = 1;
    filteredIds.get()[1] = 4;
    filteredIds.get()[2] = 3;

    {
//        unique_ptr<idx_t> ids(static_cast<idx_t*>(malloc(3 * sizeof(idx_t*))));
//        //std::vector<idx_t>ids;
//
//        ids.get()[0] = 1;
//        ids.get()[1] = 4;
//        ids.get()[2] = 3;
        //std::shared_ptr<faiss::SearchParametersHNSW> searchParametersHnsw = createSearchParams(ids);
        //std::shared_ptr<IDSelectorArray> sel = std::make_shared<IDSelectorArray>(ids.size(), ids.data());

        idx_t* ids = static_cast<idx_t*>(malloc(3 * sizeof(idx_t*)));
        //std::vector<idx_t>ids;

        ids[0] = 1;
        ids[1] = 4;
        ids[2] = 3;
        IDSelectorArray sel2(3, ids);

        std::unique_ptr<IDSelector> sel = getIdSelector(ids);

//        std::shared_ptr<IDSelectorArray> sel(
//                new IDSelectorArray(ids.size(), ids.data()));
        // IDSelectorArray sel(ids.size(), ids.data());
        std::unique_ptr<SearchParameters> searchParametersHnsw = createSearchParamsV2(sel.get());
        //SearchParameters* abc = createSearchParams(ids);
        //unique_ptr<SearchParameters> searchParametersHnsw;
        //searchParametersHnsw.reset(createSearchParams(sel));

        int k = 2;

        { // search xq
            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            index.search(nq, xq, k, D, I, searchParametersHnsw.get());

            // print results
            printf("I (%d first results)=\n", k);
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < k; j++)
                    printf("%5zd ", I[i * k + j]);
                printf("\n");
            }

            printf("D (5 first results)=\n");
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++)
                    printf("%5f ", D[i * k + j]);
                printf("\n");
            }

            delete[] I;
            delete[] D;
        }
        //ids.get_allocator().deallocate(ids.data(), ids.size());
        //free(ids);
        //free(filteredIds);
    }

    cout<<"I am out of scope"<<endl;


    delete[] xb;
    delete[] xq;

    return 0;
}
