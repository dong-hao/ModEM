# CUDA/HIP to SYCL API Mapping Reference

This document provides a quick reference for translating CUDA/HIP code to SYCL for the ModEM Intel GPU implementation.

## Core Concepts Mapping

| Concept | CUDA | HIP | SYCL/DPC++ |
|---------|------|-----|------------|
| Device | `cudaDevice_t` | `hipDevice_t` | `sycl::device` |
| Stream/Queue | `cudaStream_t` | `hipStream_t` | `sycl::queue` |
| Event | `cudaEvent_t` | `hipEvent_t` | `sycl::event` |
| Memory Pointer | `void*` | `void*` | `void*` (USM) or `sycl::buffer` |
| Kernel Launch | `<<<>>>` syntax | `<<<>>>` syntax | `queue.submit()` with lambda |
| Error Type | `cudaError_t` | `hipError_t` | `sycl::exception` |

## Device Management

| Operation | CUDA | SYCL |
|-----------|------|------|
| Get device count | `cudaGetDeviceCount(&count)` | `sycl::device::get_devices().size()` |
| Set device | `cudaSetDevice(id)` | Use specific device in queue creation |
| Get properties | `cudaGetDeviceProperties(&prop, id)` | `device.get_info<sycl::info::device::*>()` |
| Synchronize | `cudaDeviceSynchronize()` | `queue.wait()` |
| Check error | `cudaGetLastError()` | Exception handling |

## Memory Management

| Operation | CUDA | SYCL USM |
|-----------|------|----------|
| Allocate | `cudaMalloc(&ptr, size)` | `sycl::malloc_device<T>(count, queue)` |
| Free | `cudaFree(ptr)` | `sycl::free(ptr, queue)` |
| Copy H→D | `cudaMemcpy(dst, src, size, H2D)` | `queue.memcpy(dst, src, size)` |
| Copy D→H | `cudaMemcpy(dst, src, size, D2H)` | `queue.memcpy(dst, src, size)` |
| Copy D→D | `cudaMemcpy(dst, src, size, D2D)` | `queue.memcpy(dst, src, size)` |
| Async copy | `cudaMemcpyAsync(..., stream)` | `queue.memcpy(...).wait()` |
| Set memory | `cudaMemset(ptr, val, size)` | `queue.memset(ptr, val, size)` |

## Kernel Launch

### CUDA/HIP Style
```cpp
// Define kernel
__global__ void myKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2;
}

// Launch
dim3 blocks(256);
dim3 grids((N + 255) / 256);
myKernel<<<grids, blocks, 0, stream>>>(data, N);
```

### SYCL Style
```cpp
// Define kernel
void myKernel(int *data, int N, sycl::nd_item<1> item) {
    int idx = item.get_global_id(0);
    if (idx < N) data[idx] *= 2;
}

// Launch
sycl::range<1> global(((N + 255) / 256) * 256);
sycl::range<1> local(256);
queue.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(global, local), 
                   [=](sycl::nd_item<1> item) {
        myKernel(data, N, item);
    });
});
```

## Thread Indexing

| Component | CUDA | SYCL |
|-----------|------|------|
| Thread ID in block | `threadIdx.x/y/z` | `item.get_local_id(0/1/2)` |
| Block ID | `blockIdx.x/y/z` | `item.get_group(0/1/2)` |
| Block size | `blockDim.x/y/z` | `item.get_local_range(0/1/2)` |
| Grid size | `gridDim.x/y/z` | `item.get_group_range(0/1/2)` |
| Global ID | `blockIdx.x * blockDim.x + threadIdx.x` | `item.get_global_id(0)` |

### Example: 2D Indexing

**CUDA:**
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

**SYCL:**
```cpp
int x = item.get_global_id(0);
int y = item.get_global_id(1);
// Or
int x = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
int y = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
```

## Synchronization

| Operation | CUDA | SYCL |
|-----------|------|------|
| Block sync | `__syncthreads()` | `item.barrier()` |
| Memory fence | `__threadfence()` | `sycl::atomic_fence()` |
| Grid sync | Not directly supported | `item.barrier(sycl::access::fence_space::global_space)` |

## Atomic Operations

| Operation | CUDA | SYCL |
|-----------|------|------|
| Atomic add | `atomicAdd(&var, val)` | `sycl::atomic_ref<int>(var).fetch_add(val)` |
| Atomic CAS | `atomicCAS(&var, cmp, val)` | `sycl::atomic_ref<int>(var).compare_exchange_strong(cmp, val)` |
| Atomic min | `atomicMin(&var, val)` | `sycl::atomic_ref<int>(var).fetch_min(val)` |
| Atomic max | `atomicMax(&var, val)` | `sycl::atomic_ref<int>(var).fetch_max(val)` |

## Math Functions

| Function | CUDA | SYCL |
|----------|------|------|
| Square root | `sqrt(x)` | `sycl::sqrt(x)` |
| Sine | `sin(x)` | `sycl::sin(x)` |
| Cosine | `cos(x)` | `sycl::cos(x)` |
| Power | `pow(x, y)` | `sycl::pow(x, y)` |
| Exponential | `exp(x)` | `sycl::exp(x)` |
| Double→Float | `__double2float_rn(x)` | `static_cast<float>(x)` |

## BLAS Operations (cuBLAS vs oneMKL)

| Operation | cuBLAS | oneMKL |
|-----------|--------|--------|
| Create handle | `cublasCreate(&handle)` | `N/A` (queue-based) |
| Destroy handle | `cublasDestroy(handle)` | `N/A` |
| Vector dot | `cublasDdot(...)` | `onemkl::blas::dot(queue, ...)` |
| Vector norm | `cublasDnrm2(...)` | `onemkl::blas::nrm2(queue, ...)` |
| AXPY | `cublasDaxpy(...)` | `onemkl::blas::axpy(queue, ...)` |
| GEMV | `cublasDgemv(...)` | `onemkl::blas::gemv(queue, ...)` |

## Sparse Operations (cuSPARSE vs oneMKL)

| Operation | cuSPARSE | oneMKL Sparse |
|-----------|----------|---------------|
| Create handle | `cusparseCreate(&handle)` | `N/A` (queue-based) |
| Create matrix | `cusparseCreateCsr(...)` | `onemkl::sparse::set_csr_data(...)` |
| SpMV | `cusparseSpMV(...)` | `onemkl::sparse::gemv(...)` |
| SpSV | `cusparseSpSV_solve(...)` | `onemkl::sparse::trsv(...)` |
| ILU factorization | `cusparseIlu0(...)` | Custom implementation needed |

## Preprocessor Directives

### Current Code Pattern
```fortran
#if defined(CUDA)
   use cudaFortMap
#elif defined(HIP)
   use hipFortMap
#endif
```

### Updated Pattern
```fortran
#if defined(CUDA)
   use cudaFortMap
#elif defined(HIP)
   use hipFortMap
#elif defined(SYCL)
   use syclFortMap
#endif
```

## Error Handling

### CUDA Style
```cpp
cudaError_t err;
err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

### SYCL Style
```cpp
try {
    auto ptr = sycl::malloc_device<double>(size, queue);
} catch (sycl::exception const &e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
}
```

## Specific Kernel Conversions

### Example 1: Type Conversion Kernel

**CUDA:**
```cpp
__global__ void real64to32(const double *in, float *out, const int N) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos < N) {
        out[pos] = __double2float_rn(in[pos]);
    }
}
```

**SYCL:**
```cpp
void real64to32(const double *in, float *out, const int N,
                sycl::nd_item<1> item) {
    int pos = item.get_global_id(0);
    if (pos < N) {
        out[pos] = static_cast<float>(in[pos]);
    }
}
```

### Example 2: Hadamard Product

**CUDA:**
```cpp
__global__ void hada_cmplx(const double *ina, const double *inb, 
                           double *out, const int N) {
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos < N) {
        if (pos % 2 == 0) {
            out[pos] = ina[pos]*inb[pos] - ina[pos+1]*inb[pos+1];
        } else {
            out[pos] = ina[pos-1]*inb[pos] + ina[pos]*inb[pos-1];
        }
    }
}
```

**SYCL:**
```cpp
void hada_cmplx(const double *ina, const double *inb, 
                double *out, const int N,
                sycl::nd_item<1> item) {
    int pos = item.get_global_id(0);
    if (pos < N) {
        if (pos % 2 == 0) {
            out[pos] = ina[pos]*inb[pos] - ina[pos+1]*inb[pos+1];
        } else {
            out[pos] = ina[pos-1]*inb[pos] + ina[pos]*inb[pos-1];
        }
    }
}
```

## C Wrapper Functions

### CUDA Style
```cpp
extern "C" void kernelc_hadar(const double *a_d, const double *b_d, 
                              double *c_d, int Np, cudaStream_t stream) {
    int N = Np;
    int ngrid = N/256 + 1;
    dim3 grids(ngrid, 1, 1);
    dim3 blocks(32, 8, 1);
    hada_real<<<grids, blocks, 0, stream>>>(a_d, b_d, c_d, N);
}
```

### SYCL Style
```cpp
extern "C" void kernelc_hadar_sycl(const double *a_d, const double *b_d, 
                                   double *c_d, int Np, sycl::queue *q) {
    int N = Np;
    int ngrid = N/256 + 1;
    sycl::range<3> global(ngrid, 8, 32);
    sycl::range<3> local(1, 8, 32);
    
    q->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(global, local),
                       [=](sycl::nd_item<3> item) {
            hada_real(a_d, b_d, c_d, N, item);
        });
    });
}
```

## Fortran Interface

### CUDA Interface
```fortran
interface
   integer(c_int) function cudaMalloc(devPtr, size) bind(C)
      use iso_c_binding
      type(c_ptr) :: devPtr
      integer(c_size_t), value :: size
   end function cudaMalloc
end interface
```

### SYCL Interface
```fortran
interface
   type(c_ptr) function syclMallocDevice(size, queue) bind(C, name='sycl_malloc_device')
      use iso_c_binding
      integer(c_size_t), value :: size
      type(c_ptr), value :: queue
   end function syclMallocDevice
end interface
```

## Key Differences to Remember

1. **No <<<>>> syntax:** SYCL uses `queue.submit()` with lambda functions
2. **Queue-based:** Operations are submitted to queues, not streams
3. **Exception handling:** SYCL uses exceptions instead of error codes
4. **USM preferred:** Unified Shared Memory is the recommended model
5. **Explicit waits:** Use `.wait()` or events for synchronization
6. **Different indexing:** `item.get_*()` methods instead of built-in variables
7. **oneMKL namespace:** All BLAS/Sparse operations in `onemkl::` namespace
8. **No handle creation:** oneMKL uses queue directly, no handle needed

## Tips for Conversion

1. **Start simple:** Convert basic kernels first (type conversion, simple math)
2. **Use dpct:** Intel's compatibility tool can auto-convert CUDA to SYCL
3. **Test incrementally:** Verify each kernel before moving to the next
4. **Match dimensions:** Ensure work-group sizes match CUDA block sizes
5. **Check ordering:** SYCL index ordering is opposite to CUDA (0=x, 1=y, 2=z)
6. **Profile:** Use Intel VTune to optimize performance
7. **Read docs:** oneMKL documentation is essential for sparse operations

## Common Pitfalls

1. **Forgetting .wait():** SYCL operations are asynchronous by default
2. **Wrong index order:** SYCL uses [0,1,2] for [x,y,z], not [x,y,z] directly
3. **Missing queue context:** All device operations need a queue
4. **Buffer vs USM:** Mixing paradigms can cause issues - stick with USM
5. **oneMKL dependencies:** Must link against Intel MKL with SYCL support
6. **Device selection:** Must explicitly select GPU device in queue creation

## Useful SYCL Code Patterns

### Device Selection
```cpp
// Select Intel GPU
auto devices = sycl::device::get_devices();
sycl::device gpu_device;
for (auto &d : devices) {
    if (d.is_gpu() && 
        d.get_info<sycl::info::device::vendor_id>() == 0x8086) {
        gpu_device = d;
        break;
    }
}
sycl::queue queue(gpu_device);
```

### Error Checking
```cpp
queue.submit([&](sycl::handler &h) {
    h.parallel_for(...);
}).wait_and_throw(); // Will throw on errors
```

### Memory Info
```cpp
auto total_mem = device.get_info<sycl::info::device::global_mem_size>();
auto local_mem = device.get_info<sycl::info::device::local_mem_size>();
```

---

**Reference Version:** 1.0  
**Last Updated:** 2025-12-07  
**For:** ModEM Intel GPU Implementation
