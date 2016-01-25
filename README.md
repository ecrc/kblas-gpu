# kblas-gpu
=========================
KBLAS README FILE
=========================

KBLAS is an optimized library for a subset of Basic Linear Algebra Subroutines (BLAS) on NVIDIA GPUs.
KBLAS is written in CUDA C. It requires CUDA Toolkit for installation.

* Installation
  To install KBLAS, you need to have CUDA Toolkit installed (version 5.0 or higher is recommended)
  All that is required is to edit the make.inc file and then type make. Specify the following in 
  your make.inc file:
     - The directory of the CUDA Toolkit installation (default: /usr/local/cuda)
     - The target GPU architecture: currently "fermi" or "kepler". KBLAS was not tested on pre-fermi GPUs
   