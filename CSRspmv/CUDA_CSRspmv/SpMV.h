#pragma once
#include"utils.h"
#include<mkl.h>
int cuda_spmv(std::string filename, int numRows, int numCols, int nItems, int *cols, int * rowDelimiters, FloatType* vals) {
    try
    {
        FloatType  alpha = 1.0f;
        FloatType  beta = 0.0f;
        int* hA_columns = cols;
        int* hA_csrOffsets = rowDelimiters;
        FloatType* hA_values = vals;
        int A_num_rows = numRows;
        int A_num_cols = numCols;
        int A_nnz = nItems;
        std::cout << A_nnz << std::endl;
        FloatType* hX = (FloatType*)malloc(sizeof(FloatType) * numCols);
        FloatType* hY = (FloatType*)malloc(sizeof(FloatType) * (numRows + 1));
        FloatType* hY_result = (FloatType*)malloc(sizeof(FloatType) * (numRows + 1));
        initVector(hX, numCols + 1, 1);
        initVector(hY, numRows + 1, 0);
        initVector(hY_result, numRows + 1, 0);

        int* dA_csrOffsets, * dA_columns;
        FloatType* dA_values, * dX, * dY;

        CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(FloatType)))
        CHECK_CUDA(cudaMalloc((void**)&dX, A_num_cols * sizeof(FloatType)))
        CHECK_CUDA(cudaMalloc((void**)&dY, A_num_rows * sizeof(FloatType)))

        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,(A_num_rows + 1) * sizeof(int),cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(FloatType),cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(FloatType),cudaMemcpyHostToDevice))
        CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(FloatType),cudaMemcpyHostToDevice))
            //--------------------------------------------------------------------------
            // CUSPARSE APIs
            cusparseHandle_t	 handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        void* dBuffer = NULL;
        size_t				 bufferSize = 0;
        CHECK_CUSPARSE(cusparseCreate(&handle))
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F))
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F))
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

        // execute SpMV
        int	 times = ITERATION;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        auto consume = dur.count() / 1000.0;

        // average time for SpMV
        double avg_time = consume / times;
        std::cout << "time: " << avg_time << std::endl;
        // BW Mb/s
        std::cout << "BW: " << 1.0e-6 * nItems * 8 / (consume / times) << std::endl;
        std::cout << "GF: " << 1.0e-9 * 2 * nItems / (consume / times) << std::endl;

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(matA))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
        CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
        CHECK_CUSPARSE(cusparseDestroy(handle))
        //--------------------------------------------------------------------------
        // device result check
        CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(FloatType), cudaMemcpyDeviceToHost))
        // for (int i = 0; i < numRows + 1; i++) {
        // 	printf("%lf\n", hY[i]);
        // }
        // cout << endl;
        //--------------------------------------------------------------------------
        // device memory deallocation
        CHECK_CUDA(cudaFree(dBuffer))
        CHECK_CUDA(cudaFree(dA_csrOffsets))
        CHECK_CUDA(cudaFree(dA_columns))
        CHECK_CUDA(cudaFree(dA_values))
        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))
    }
    catch (const std::exception&)
    {
        std::cout << filename << " failed." << std::endl;
    }
    return 1;
}

int spmv_csr_xhit(std::string filename, int numRows, int numCols, int nItems, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y) {
    int	 times = ITERATION;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        #pragma omp parallel for
        for (int i = 0; i < numRows; i++) {
            double sum = 0.0;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
                int xidx = Aj[j];
                sum += x[0] * Ax[j];
            }
            y[i] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    auto consume = dur.count() / 1000.0;
    double avg_time = consume / times;
    std::cout << "hit time: " << avg_time << "\tBW: " << 1.0e-9 * nItems * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nItems / (consume / times) << std::endl;
    return 1;
}

int spmv_csr_omp(std::string filename, int numRows, int numCols, int nItems, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y) {
    int	 times = ITERATION;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        #pragma omp parallel for
        for (int i = 0; i < numRows; i++) {
            double sum = 0.0;
            for (int j = Ap[i]; j < Ap[i + 1]; j++) {
                sum += x[Aj[j]] * Ax[j];
            }
            y[i] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    auto consume = dur.count() / 1000.0;
    double avg_time = consume / times;
    std::cout << "omp time: " << avg_time << "\tBW: " << 1.0e-9 * nItems * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nItems / (consume / times) << std::endl;
    return 1;
}

int spmv_csr_mkl(std::string filename, int nr, int nc, int nnz, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y, int flag = 1) {
    // tbb::task_scheduler_init init(thread_num);
    sparse_status_t stat;
    sparse_matrix_t A;
    struct matrix_descr tt = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER,
                              SPARSE_DIAG_NON_UNIT };
    int tn = mkl_get_max_threads();
    mkl_set_num_threads(tn);
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nr, nc, Ap, Ap + 1, Aj, Ax);
    double alpha = 1.0;
    double beta = 0.0;
    double preprocessing = 0.0;
    if (flag) {
        auto start = std::chrono::high_resolution_clock::now();

        mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, tt, 200);
        mkl_sparse_optimize(A);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        preprocessing = dur.count() / 1000.0;
    }
    int	 times = ITERATION;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, tt, x,
            beta, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    auto consume = dur.count() / 1000.0;
    double avg_time = consume / times;
    if (flag) {
        std::cout << "okl time: " << avg_time << "\tBW: " << 1.0e-9 * nnz * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nnz / (consume / times) <<"\tpreprocessing: "<< preprocessing << std::endl;
    }
    else {
        std::cout << "mkl time: " << avg_time << "\tBW: " << 1.0e-9 * nnz * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nnz / (consume / times) << std::endl;
    }

    return 1;
}

int spmv_csr_mkl_merge(std::string filename, int nr, int nc, int nnz, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y, int flag = 1) {
    // tbb::task_scheduler_init init(thread_num);
    sparse_status_t stat;
    sparse_matrix_t A;
    struct matrix_descr tt = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };
    int tn = mkl_get_max_threads();
    mkl_set_num_threads(tn);
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nr, nc, Ap, Ap + 1, Aj, Ax);
    double alpha = 1.0;
    double beta = 0.0;
    double preprocessing = 0.0;
    if (flag) {
        auto start = std::chrono::high_resolution_clock::now();

        mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, tt, 200);
        mkl_sparse_optimize(A);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        preprocessing = dur.count() / 1000.0;
    }
    int	 times = ITERATION;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        stat = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, tt, x,
            beta, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    auto consume = dur.count() / 1000.0;
    double avg_time = consume / times;
    if (flag) {
        std::cout << "okl time: " << avg_time << "\tBW: " << 1.0e-9 * nnz * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nnz / (consume / times) << "\tpreprocessing: " << preprocessing << std::endl;
    }
    else {
        std::cout << "mkl time: " << avg_time << "\tBW: " << 1.0e-9 * nnz * 8 / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nnz / (consume / times) << std::endl;
    }


    return 1;
}