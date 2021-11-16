#ifndef __KBLAS_POTRF_H__
#define __KBLAS_POTRF_H__

//###############################################################
// POTRF
//###############################################################

#ifdef __cplusplus
extern "C" {
#endif

int kblasSpotrf(kblasHandle_t handle, char uplo, int n, float *dA, int ldda, int* info);
int kblasDpotrf(kblasHandle_t handle, char uplo, int n, double *dA, int ldda, int* info);

#ifdef __cplusplus
}
#endif 

/** @addtogroup CPP_API
*  @{
*/
#ifdef __cplusplus
/**
* @name TODO
*/
//@{

/**
* @brief TODO
*/
inline int kblas_potrf(kblasHandle_t handle, char uplo, int n, float *dA, int ldda, int* info)
{ return kblasSpotrf(handle, uplo, n, dA, ldda, info); }

/**
* @brief TODO
*/
inline int kblas_potrf(kblasHandle_t handle, char uplo, int n, double *dA, int ldda, int* info)
{ return kblasDpotrf(handle, uplo, n, dA, ldda, info); }
//@}
#endif
/** @} */

#endif 
