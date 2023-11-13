
#define _CRT_SECURE_NO_WARNINGS 1 
#include "ReadMatrix.h"
#include "SpMV.h"
using namespace std;
int main(void) {
    int count = 0;
    freopen("./test_result.txt", "a", stdout);

    
    for (auto& i : filesystem::directory_iterator("D:\\Code With VS\\CSRspmv\\CUDA_CSRspmv\\matrix_mtx")) {
        string filename = i.path().string();
        count++;
        try {
            cout << "Matrix Name: " << filename << endl;
            FloatType *Ax;
            int *Aj, *Ap;
            int nnz, nr, nc;
            ReadMatrix::readMatrix(filename, &Ax, &Aj, &Ap, &nnz, &nr, &nc);
            FloatType *x = new FloatType[nc];
            FloatType *y = new FloatType[nr];
            initVector(x, nc, 1);
            initVector(y, nr, 0);
            spmv_csr_mkl(filename, nr, nc, nnz, Aj, Ap, Ax, x, y);
            spmv_csr_mkl(filename, nr, nc, nnz, Aj, Ap, Ax, x, y, 0);
            spmv_csr_xhit(filename, nr, nc, nnz, Aj, Ap, Ax, x, y);
            spmv_csr_omp(filename, nr, nc, nnz, Aj, Ap, Ax, x, y);
            //cuda_spmv(filename, nr, nc, nnz, Aj, Ap, Ax);
        }
        catch (exception e) {
            cout << filename << " failed." << endl;
        }

    }
    // ¹Ø±ÕÎÄ¼þ
    fclose(stdout);
    return EXIT_SUCCESS;
}