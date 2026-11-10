// Regression coverage for the per-operation dense ARA tolerance API.
#include <cublas_v2.h>
#include <magma_v2.h>
#include "kblas.h"
#include "batch_ara.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void kblas_check(int code) {
    require(code == KBLAS_Success, "KBLAS call failed");
}
template<class T> class Device {
public:
    T* p;
    explicit Device(size_t n) { cuda_check(cudaMalloc((void**)&p, n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& v) {
        cuda_check(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        cuda_check(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
struct Context {
    cudaStream_t stream;
    kblasHandle_t handle;
    kblasRandState_t state;
    Context() {
        cuda_check(cudaStreamCreate(&stream));
        kblas_check(kblasCreate(&handle));
        kblas_check(kblasEnableMagma(handle));
        kblasSetStream(handle, stream);
        kblas_check(kblasInitRandState(handle, &state, 32768, 0));
    }
    ~Context() {
        cudaStreamSynchronize(stream);
        kblasDestroyRandState(state);
        kblasDestroy(&handle);
        cudaStreamDestroy(stream);
    }
};
template<class T> struct API;
#define DEFINE_API(T, R, PREFIX) \
template<> struct API<T> { \
    typedef R Real; \
    static int call(Context& ctx, int* rows, int* cols, T** a, int* lda, \
                    T** u, int* ldu, T** v, int* ldv, int* ranks, \
                    R scalar_tol, const R* tolerances, int* maxrank, \
                    int* info, int relative, int count, bool array) { \
        if (array) return kblas_##PREFIX##ara_batch_tol(ctx.handle, \
            rows, cols, a, lda, u, ldu, v, ldv, ranks, tolerances, \
            64, 48, maxrank, 16, 10, info, ctx.state, relative, count); \
        return kblas_##PREFIX##ara_batch(ctx.handle, \
            rows, cols, a, lda, u, ldu, v, ldv, ranks, scalar_tol, \
            64, 48, maxrank, 16, 10, info, ctx.state, relative, count); \
    } \
};
DEFINE_API(float, float, s)
DEFINE_API(double, double, d)
DEFINE_API(cuComplex, float, c)
DEFINE_API(cuDoubleComplex, double, z)
#undef DEFINE_API

template<class T> T entry(double x) { return T(x); }
template<> cuComplex entry(double x) { return make_cuComplex(.8*x, .6*x); }
template<> cuDoubleComplex entry(double x) { return make_cuDoubleComplex(.8*x, .6*x); }
template<class T> std::complex<double> host(T x) { return {double(x), 0}; }
template<> std::complex<double> host(cuComplex x) { return {x.x, x.y}; }
template<> std::complex<double> host(cuDoubleComplex x) { return {x.x, x.y}; }

template<class T> void test() {
    typedef typename API<T>::Real R;
    const int count=6, m=64, n=48, ld=67, maxrank=48;
    const double scales[count]={1e-3,1,1e3,1e-3,1,1e3};
    std::vector<T> a(count*ld*n, entry<T>(0));
    std::vector<R> tolerances(count);
    for (int b=0; b<count; b++) {
        tolerances[b]=R(scales[b]*(b<3 ? 1e-2 : 1e-5));
        for (int j=0; j<n; j++) a[b*ld*n+j+j*ld]=entry<T>(scales[b]*std::pow(10.,-j/6.));
    }
    Device<T> da(a.size()), du(count*m*maxrank), dv(count*n*maxrank);
    Device<T*> dap(count), dup(count), dvp(count);
    Device<int> rows(count), cols(count), lda(count), ldu(count), ldv(count);
    Device<int> maxr(count), ranks(count), info(count);
    Device<R> dtol(count);
    da.put(a); dtol.put(tolerances);
    std::vector<T*> ap(count), up(count), vp(count);
    for (int b=0; b<count; b++) {
        ap[b]=da.p+b*ld*n; up[b]=du.p+b*m*maxrank; vp[b]=dv.p+b*n*maxrank;
    }
    dap.put(ap); dup.put(up); dvp.put(vp);
    rows.put(std::vector<int>(count,m)); cols.put(std::vector<int>(count,n));
    lda.put(std::vector<int>(count,ld)); ldu.put(std::vector<int>(count,m));
    ldv.put(std::vector<int>(count,n)); maxr.put(std::vector<int>(count,maxrank));
    Context ctx;
    kblas_ara_batch_wsquery<T>(ctx.handle,16,count);
    kblas_check(kblasAllocateWorkspace(ctx.handle));
    kblas_check(API<T>::call(ctx,rows.p,cols.p,dap.p,lda.p,dup.p,ldu.p,
        dvp.p,ldv.p,ranks.p,R(0),dtol.p,maxr.p,info.p,0,count,true));
    cuda_check(cudaStreamSynchronize(ctx.stream));
    auto rank=ranks.get(count), status=info.get(count);
    auto u=du.get(count*m*maxrank), v=dv.get(count*n*maxrank);
    for (int b=0; b<count; b++) {
        require(status[b]==KBLAS_Success, "array ARA did not converge");
        require(rank[b]>=0 && rank[b]<=maxrank, "invalid rank");
        double err=0, norm=0;
        for (int j=0; j<n; j++) for (int i=0; i<m; i++) {
            std::complex<double> approx=0;
            for (int k=0; k<rank[b]; k++)
                approx+=host(u[b*m*maxrank+i+k*m])*std::conj(host(v[b*n*maxrank+j+k*n]));
            auto exact=host(a[b*ld*n+i+j*ld]);
            err=std::hypot(err,std::abs(exact-approx)); norm=std::hypot(norm,std::abs(exact));
        }
        require(err/norm < 10*(b<3 ? 1e-2 : 1e-5), "approximation error too large");
        std::cout << "bytes=" << sizeof(T) << " tile=" << b << " rank=" << rank[b]
                  << " relative error=" << err/norm << '\n';
    }
    for (int b=0; b<3; b++) require(rank[b+3]>rank[b], "per-tile tolerance has no effect");

    // Same seed and same thresholds must reproduce scalar API ranks in both modes.
    for (int relative=0; relative<=1; relative++) {
        Context scalar_ctx, array_ctx;
        for (auto c : {&scalar_ctx,&array_ctx}) {
            kblas_ara_batch_wsquery<T>(c->handle,16,count);
            kblas_check(kblasAllocateWorkspace(c->handle));
        }
        const R tol=R(.1);
        dtol.put(std::vector<R>(count,tol));
        kblas_check(API<T>::call(scalar_ctx,rows.p,cols.p,dap.p,lda.p,dup.p,ldu.p,
            dvp.p,ldv.p,ranks.p,tol,nullptr,maxr.p,info.p,relative,count,false));
        cuda_check(cudaStreamSynchronize(scalar_ctx.stream));
        auto scalar_ranks=ranks.get(count), scalar_info=info.get(count);
        kblas_check(API<T>::call(array_ctx,rows.p,cols.p,dap.p,lda.p,dup.p,ldu.p,
            dvp.p,ldv.p,ranks.p,tol,dtol.p,maxr.p,info.p,relative,count,true));
        cuda_check(cudaStreamSynchronize(array_ctx.stream));
        require(scalar_ranks==ranks.get(count), "scalar/array ranks differ");
        require(scalar_info==info.get(count), "scalar/array status differs");
        for (auto s : scalar_info) require(s==KBLAS_Success, "uniform tolerance did not converge");
    }
    // Invalid counts and null pointers are rejected before touching device inputs.
    for (int count_arg : {-1,0,1}) {
        int rc=API<T>::call(ctx,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
            nullptr,nullptr,nullptr,R(0),nullptr,nullptr,nullptr,0,count_arg,true);
        require(rc==(count_arg==0 ? KBLAS_Success : KBLAS_Error_WrongInput), "invalid input handling");
    }
}
} // namespace
int main() {
    if (magma_init() != MAGMA_SUCCESS) return 1;
    try {
        test<float>(); test<double>(); test<cuComplex>(); test<cuDoubleComplex>();
        std::cout << "All per-operation ARA tolerance tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; magma_finalize(); return 1;
    }
    magma_finalize();
}
