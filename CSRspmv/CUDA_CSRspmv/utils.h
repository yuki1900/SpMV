#pragma once
#include <sstream>
#include <algorithm>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <string.h>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <chrono>
#include <omp.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}




#define FloatType double
#define CUDA_FLOAT_TYPE CUDA_R_64F

int ITERATION = 5000;

inline void initVector(FloatType* v, int v_len, FloatType init_v)
{
    for (int i = 0; i < v_len; i++) {
        *(v + i) = init_v;
    }
}

inline bool verify(FloatType* y1, FloatType* y2, int v_len)
{
    for (int i = 0; i < v_len; i++) {
        if (fabs(y1[i] - y2[i]) > 1e-6) {
            std::cout << "Wrong answer i = " << i << " " << y1[i] << " " << y2[i] << std::endl;
            return false;
        }
    }
    return true;
}