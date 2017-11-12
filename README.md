# kblas-gpu

What is KBLAS
=============

KAUST BLAS (KBLAS) is a high performance CUDA library implementing a subset of BLAS as well as Linear Algebra PACKage (LAPACK) routines on NVIDIA GPUs. Using recursive and batch algorithms, KBLAS maximizes the GPU bandwidth, reuses locally cached data and increases device occupancy. KBLAS represents, therefore, a comprehensive and efficient framework versatile to various workload sizes. Located at the bottom of the usual software stack, KBLAS enables higher-level numerical libraries and scientific applications to extract the expected performance from GPU hardware accelerators.

KBLAS is written in CUDA C. It requires CUDA Toolkit for installation.


Current Features of KBLAS
=========================

KBLAS provides highly optimized routines from various levels of BLAS and LAPACK, including:

1. Legacy Level-2 BLAS: (⇟⎐ ⚭ ⚬) SYMV, GEMV, HEMV.
2. Legacy Level-3 BLAS: (⇟⎐ ⚭ ⚬) TRSM, TRMM, GEMM (⚭ only).
3. Batch Level-3 BLAS: (⇟⎏ ⚭ ⚬= ✼) TRSM, TRMM, SYRK.
4. Batch Triangular: (⎏⇞ ⚭ ⚬= ✼) TRTRI, LAUUM. ◆ Batch Symmetric: (⎏⇞ ⚭ ⚬= ✼) POTRF, POTRS, POSV, POTRI, POTI.
5. Batch General: (⎐⇟ ⚭ ⚬= ✼) GESVJ, GERSVD, GEQRF.

⇟ Standard precisions: s/d/c/z.
⇞ Real precisions: s/d.
⎏ Very small matrix sizes.
⎐ Arbitrary sizes.
⚬ Single-GPU support.
⚭ Multi-GPU support.
= Uniform batch sizes.
✼ Non-Strided and Strided variants


Installation
============

KBLAS installation requires a recent **make**.
To build KBLAS, please follow these instructions:

1.  Get KBLAS from git repository

        git clone git@github.com:ecrc/kblas-gpu

    or

        git clone https://github.com/ecrc/kblas-gpu

2.  Go into KBLAS folder

        cd kblas-gpu

3.  Edit file make.inc to:
    - Enable / disable KBLAS sub modules (_SUPPORT_BLAS2_, _SUPPORT_BLAS3_, _SUPPORT_BATCH_TR_, _SUPPORT_SVD_).
    - Enable / disable usage of third party libraries (_USE_MKL_, _USE_MAGMA_) for performance comparsions.
    - Provide path for third party libraries if required (_CUB_DIR_, _MAGMA_ROOT_).
    - Specify CUDA architecture to compile for (_CUDA_ARCH_).

    or

    - Provide equivalent environment variables.

4.  Build KBLAS

        make

5.  Build local documentation (optional)

        make docs


Testing
=======

The folder 'testing' includes a set of sample programs to illustrate the usage of each KBLAS routine, as well as to test the performance and accuracy of such routines against other vendor libraries.


![Handout](docs/KBLAS-brochure.pdf)