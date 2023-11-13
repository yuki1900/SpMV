//
//#define _CRT_SECURE_NO_WARNINGS 1 
//#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
//#include <cusparse.h>         // cusparseSpMV
//#include <stdio.h>            // printf
//#include <stdlib.h>           // EXIT_FAILURE
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <cstring>
//#include <string.h>
//#include <vector>
//#include <filesystem>
//#include <cstdio>
//#include <chrono>
//#include "ReadMatrix.h"
//using namespace std;
//#define floatType double
//#define CUDA_FLOAT_TYPE CUDA_R_64F
//
//#define CHECK_CUDA(func)                                                       \
//{                                                                              \
//    cudaError_t status = (func);                                               \
//    if (status != cudaSuccess) {                                               \
//        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
//               __LINE__, cudaGetErrorString(status), status);                  \
//        return EXIT_FAILURE;                                                   \
//    }                                                                          \
//}
//
//#define CHECK_CUSPARSE(func)                                                   \
//{                                                                              \
//    cusparseStatus_t status = (func);                                          \
//    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
//        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
//               __LINE__, cusparseGetErrorString(status), status);              \
//        return EXIT_FAILURE;                                                   \
//    }                                                                          \
//}
//
//
//
//int main(void) {
//    int	  file_num = 7;
//    char* filenames[] = {
//        (char*)"mtx_files/spal_004.mtx",
//        (char*)"mtx_files/rajat31.mtx",
//        (char*)"mtx_files/ldoor.mtx",
//        (char*)"mtx_files/F1.mtx",
//        (char*)"mtx_files/cage14.mtx",
//        (char*)"mtx_files/bone010.mtx",/**/
//        (char*)"mtx_files/af_1_k101.mtx" };
//
//    // int	  file_num	  = 1;
//    // char *filenames[] = {
//    // 	(char *)"mtx_files/F1.mtx"};
//    int count = 0;
//    freopen("./result.txt", "a", stdout);
//
//
//    for (auto& i : filesystem::directory_iterator("D:\\Code With VS\\CSRspmv\\CUDA_CSRspmv\\matrix_mtx")) {
//        string filename = i.path().string();
//        count++;
//        try {
//            //printf("Tste for %s\n", filename);
//            cout << "Matrix Name: " << filename << endl;
//            // exit(0);
//
//            // Host problem definition
//            int A_num_rows;
//            int A_num_cols;
//            int A_nnz;
//            // char	   filename[] = "F1.mtx";
//            int* hA_csrOffsets;
//            int* hA_columns;
//            floatType* hA_values;
//            floatType  alpha = 1.0f;
//            floatType  beta = 0.0f;
//
//            floatType* vals;
//            int* cols, * rowDelimiters;
//            int		   nItems, numRows, numCols;
//            ReadMatrix::readMatrix(filename, &vals, &cols, &rowDelimiters, &nItems, &numRows, &numCols);
//
//            hA_columns = cols;
//            hA_csrOffsets = rowDelimiters;
//            hA_values = vals;
//            A_num_rows = numRows;
//            A_num_cols = numCols;
//            A_nnz = nItems;
//            cout << A_nnz << endl;
//            floatType* hX = (floatType*)malloc(sizeof(floatType) * numCols);
//            floatType* hY = (floatType*)malloc(sizeof(floatType) * (numRows + 1));
//            floatType* hY_result = (floatType*)malloc(sizeof(floatType) * (numRows + 1));
//            initVector(hX, numCols + 1, 1);
//            initVector(hY, numRows + 1, 0);
//            initVector(hY_result, numRows + 1, 0);
//
//            int* dA_csrOffsets, * dA_columns;
//            floatType* dA_values, * dX, * dY;
//            CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets,
//                (A_num_rows + 1) * sizeof(int)))
//                CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)))
//                CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(floatType)))
//                CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(floatType)))
//                CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(floatType)))
//
//                CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
//                    (A_num_rows + 1) * sizeof(int),
//                    cudaMemcpyHostToDevice))
//                CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
//                    cudaMemcpyHostToDevice))
//                CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(floatType),
//                    cudaMemcpyHostToDevice))
//                CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(floatType),
//                    cudaMemcpyHostToDevice))
//                CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(floatType),
//                    cudaMemcpyHostToDevice))
//                //--------------------------------------------------------------------------
//                // CUSPARSE APIs
//                cusparseHandle_t	 handle = NULL;
//            cusparseSpMatDescr_t matA;
//            cusparseDnVecDescr_t vecX, vecY;
//            void* dBuffer = NULL;
//            size_t				 bufferSize = 0;
//            CHECK_CUSPARSE(cusparseCreate(&handle))
//                // Create sparse matrix A in CSR format
//                CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
//                    dA_csrOffsets, dA_columns, dA_values,
//                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
//                // Create dense vector X
//                CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F))
//                // Create dense vector y
//                CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F))
//                // allocate an external buffer if needed
//                CHECK_CUSPARSE(cusparseSpMV_bufferSize(
//                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
//                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
//                CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
//
//                // execute SpMV
//                int	 times = 3000;
//            auto start = std::chrono::high_resolution_clock::now();
//            for (int i = 0; i < times; i++) {
//                CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
//                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
//            }
//            auto end = std::chrono::high_resolution_clock::now();
//
//            std::chrono::duration<double, std::milli> dur = (end - start);
//
//            auto consume = dur.count() / 1000.0;
//
//            // average time for SpMV
//            double avg_time = consume / times;
//            cout << "time: " << avg_time << endl;
//            // bandwidth Mb/s
//            cout << "bandwidth: " << 1.0e-6 * nItems * 8 / (consume / times) << endl;
//            cout << "GFlops: " << 1.0e-9 * 2 * nItems / (consume / times) << endl;
//
//            // destroy matrix/vector descriptors
//            CHECK_CUSPARSE(cusparseDestroySpMat(matA))
//                CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
//                CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
//                CHECK_CUSPARSE(cusparseDestroy(handle))
//                //--------------------------------------------------------------------------
//                // device result check
//                CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(floatType),
//                    cudaMemcpyDeviceToHost))
//
//                // for (int i = 0; i < numRows + 1; i++) {
//                // 	printf("%lf\n", hY[i]);
//                // }
//                // cout << endl;
//
//                //--------------------------------------------------------------------------
//                // device memory deallocation
//                CHECK_CUDA(cudaFree(dBuffer))
//                CHECK_CUDA(cudaFree(dA_csrOffsets))
//                CHECK_CUDA(cudaFree(dA_columns))
//                CHECK_CUDA(cudaFree(dA_values))
//                CHECK_CUDA(cudaFree(dX))
//                CHECK_CUDA(cudaFree(dY))
//        }
//        catch (exception e) {
//            cout << filename << " failed." << endl;
//        }
//
//    }
//    // ¹Ø±ÕÎÄ¼þ
//    fclose(stdout);
//    /*
//    for (int file_cur = 0; file_cur < file_num; file_cur++) {
//        char* filename = filenames[file_cur];
//        printf("Tste for %s\n", filename);
//
//        // exit(0);
//
//        // Host problem definition
//        int A_num_rows;
//        int A_num_cols;
//        int A_nnz;
//        // char	   filename[] = "F1.mtx";
//        int* hA_csrOffsets;
//        int* hA_columns;
//        floatType* hA_values;
//        floatType  alpha = 1.0f;
//        floatType  beta = 0.0f;
//
//        floatType* vals;
//        int* cols, * rowDelimiters;
//        int		   nItems, numRows, numCols;
//        readMatrix(filename, &vals, &cols, &rowDelimiters, &nItems, &numRows, &numCols);
//
//        hA_columns = cols;
//        hA_csrOffsets = rowDelimiters;
//        hA_values = vals;
//        A_num_rows = numRows;
//        A_num_cols = numCols;
//        A_nnz = nItems;
//
//        floatType* hX = (floatType*)malloc(sizeof(floatType) * numCols);
//        floatType* hY = (floatType*)malloc(sizeof(floatType) * (numRows + 1));
//        floatType* hY_result = (floatType*)malloc(sizeof(floatType) * (numRows + 1));
//        initVector(hX, numCols + 1, 1);
//        initVector(hY, numRows + 1, 0);
//        initVector(hY_result, numRows + 1, 0);
//
//        int* dA_csrOffsets, * dA_columns;
//        floatType* dA_values, * dX, * dY;
//        CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets,
//            (A_num_rows + 1) * sizeof(int)))
//            CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)))
//            CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(floatType)))
//            CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(floatType)))
//            CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(floatType)))
//
//            CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
//                (A_num_rows + 1) * sizeof(int),
//                cudaMemcpyHostToDevice))
//            CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
//                cudaMemcpyHostToDevice))
//            CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(floatType),
//                cudaMemcpyHostToDevice))
//            CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(floatType),
//                cudaMemcpyHostToDevice))
//            CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(floatType),
//                cudaMemcpyHostToDevice))
//            //--------------------------------------------------------------------------
//            // CUSPARSE APIs
//            cusparseHandle_t	 handle = NULL;
//        cusparseSpMatDescr_t matA;
//        cusparseDnVecDescr_t vecX, vecY;
//        void* dBuffer = NULL;
//        size_t				 bufferSize = 0;
//        CHECK_CUSPARSE(cusparseCreate(&handle))
//            // Create sparse matrix A in CSR format
//            CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
//                dA_csrOffsets, dA_columns, dA_values,
//                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
//            // Create dense vector X
//            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F))
//            // Create dense vector y
//            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F))
//            // allocate an external buffer if needed
//            CHECK_CUSPARSE(cusparseSpMV_bufferSize(
//                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
//                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
//            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
//
//            // execute SpMV
//            int	 times = 3000;
//        auto start = std::chrono::high_resolution_clock::now();
//        for (int i = 0; i < times; i++) {
//            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
//                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
//        }
//        auto end = std::chrono::high_resolution_clock::now();
//
//        std::chrono::duration<double, std::milli> dur = (end - start);
//
//        auto consume = dur.count() / 1000.0;
//
//        // average time for SpMV
//        double avg_time = consume / times;
//        cout << "time: " << avg_time << endl;
//        // bandwidth Mb/s
//        cout << "bandwidth: " << 1.0e-6 * nItems * 8 / (consume / times) << endl;
//
//        // destroy matrix/vector descriptors
//        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
//            CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
//            CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
//            CHECK_CUSPARSE(cusparseDestroy(handle))
//            //--------------------------------------------------------------------------
//            // device result check
//            CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(floatType),
//                cudaMemcpyDeviceToHost))
//            CHECK_CUDA(cudaFree(dBuffer))
//            CHECK_CUDA(cudaFree(dA_csrOffsets))
//            CHECK_CUDA(cudaFree(dA_columns))
//            CHECK_CUDA(cudaFree(dA_values))
//            CHECK_CUDA(cudaFree(dX))
//            CHECK_CUDA(cudaFree(dY))
//    }*/
//    return EXIT_SUCCESS;
//}