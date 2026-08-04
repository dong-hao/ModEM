# Intel GPU Support Implementation Plan for ModEM

## Executive Summary

This document outlines the detailed plan for adding Intel GPU support to ModEM using Intel's oneAPI and SYCL/DPC++ programming model. The implementation will follow the existing patterns used for NVIDIA (CUDA) and AMD (HIP) GPU support in the FWD_SP2 solver.

**Estimated Total LOC:** ~3,500-4,000 lines  
**Estimated Effort:** 4-6 weeks for a senior developer with Intel GPU experience  
**Complexity:** Medium-High (requires GPU programming expertise and understanding of sparse matrix operations)

---

## Current State Analysis

### Existing GPU Implementation
ModEM currently supports:
- **NVIDIA GPUs**: Via CUDA API (kernel_c.cu, cudaFortMap.f90)
- **AMD GPUs**: Via HIP/ROCm API (kernel_c.hip, hipFortMap.f90)

### Key Files Involved
1. **f90/3D_MT/FWD_SP2/kernel_c.cu** (~760 LOC) - CUDA kernels
2. **f90/3D_MT/FWD_SP2/kernel_c.hip** (~650 LOC) - HIP kernels
3. **f90/3D_MT/FWD_SP2/cudaFortMap.f90** (~1,700 LOC) - CUDA-Fortran interface
4. **f90/3D_MT/FWD_SP2/hipFortMap.f90** (~1,710 LOC) - HIP-Fortran interface
5. **f90/3D_MT/FWD_SP2/solver.f90** (~4,700 LOC) - Solver with GPU variants
6. **f90/3D_MT/FWD_SP2/EMsolve3D.f90** (~1,620 LOC) - Forward solver wrapper

### GPU Operations Used
- Sparse matrix operations (SpMV, SpSV, ILU factorization)
- Vector operations (dot products, AXPY, Hadamard products)
- Type conversions (double to single precision)
- Custom BiCGStab solver kernels
- NCCL/RCCL for multi-GPU communication (FG mode)

---

## Implementation Plan

### Phase 1: SYCL Kernel Development (~1,200-1,500 LOC)

#### Task 1.1: Create kernel_c.sycl
**File:** `f90/3D_MT/FWD_SP2/kernel_c.sycl`  
**Estimated LOC:** 800-1,000 lines  
**Effort:** 1-1.5 weeks  

**Description:** Port CUDA/HIP kernels to SYCL/DPC++

**Kernels to implement:**
1. `real64to32` - Double to single precision conversion
2. `real32to64` - Single to double precision conversion
3. `hada_real` - Real Hadamard multiplication
4. `hada_cmplx` - Complex Hadamard multiplication
5. `xpby_real` - Real AXPY operation (y = x + b*y)
6. `xpby_cmplx` - Complex AXPY operation
7. `p_update_real` - BiCGStab p-vector update (real)
8. `p_update_cmplx` - BiCGStab p-vector update (complex)
9. `x_update_cmplx` - BiCGStab x-vector update
10. `reduce_real` - Reduction operation

**Implementation details:**
- Use SYCL nd_range kernels with 2D/3D work-group organization
- Match existing thread block sizes (32x8x1 for most kernels)
- Use SYCL atomic operations for reductions
- Implement device functions using SYCL inline functions
- Use SYCL USM (Unified Shared Memory) for data management

**Example structure:**
```cpp
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <complex>

// Kernel implementation
void real64to32(const double *in, float *out, const int N,
                sycl::nd_item<3> item_ct1) {
    int pos = (item_ct1.get_local_range(2) * item_ct1.get_local_range(1)) *
              item_ct1.get_group(2) + 
              item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
              item_ct1.get_local_id(2);
    if (pos < N) {
        out[pos] = static_cast<float>(in[pos]);
    }
}

// Wrapper function
extern "C" void kernelc_d2s_sycl(const double *a_d, float *b_d, int Np,
                                 sycl::queue *stream) {
    // Implementation
}
```

**Dependencies:**
- Intel oneAPI Base Toolkit
- DPC++ compiler (icpx/dpcpp)
- oneMKL library for BLAS/SPARSE operations

---

#### Task 1.2: Create Device Management Functions
**Part of:** `f90/3D_MT/FWD_SP2/kernel_c.sycl`  
**Estimated LOC:** 200-300 lines  
**Effort:** 2-3 days  

**Functions to implement:**
1. `cf_hookDev_sycl()` - Initialize and select Intel GPU device
2. `cf_resetFlag_sycl()` - Reset device flags (if needed)
3. Device property queries (memory, compute capability, etc.)
4. Device synchronization utilities

**Key differences from CUDA/HIP:**
- Use `sycl::device::get_devices()` instead of `cudaGetDeviceCount()`
- Use `sycl::device::get_info()` for device properties
- Different memory management API (USM instead of cudaMalloc)
- Use `sycl::queue` for command submission

---

#### Task 1.3: Multi-GPU Communication (Optional - FG mode)
**Part of:** `f90/3D_MT/FWD_SP2/kernel_c.sycl`  
**Estimated LOC:** 200-300 lines (if implemented)  
**Effort:** 3-5 days  

**Description:** Implement MPI+GPU communication for multi-GPU support

**Options:**
1. **Option A**: Use Intel MPI with SYCL events for GPU-aware MPI
2. **Option B**: Use CCL (Collective Communications Library) - Intel's alternative to NCCL
3. **Option C**: Use standard MPI with explicit host staging

**Note:** This is complex and may be deferred to a later phase. Current priority is single-GPU support.

---

### Phase 2: SYCL-Fortran Interface Module (~1,700-1,800 LOC)

#### Task 2.1: Create syclFortMap.f90
**File:** `f90/3D_MT/FWD_SP2/syclFortMap.f90`  
**Estimated LOC:** 1,700-1,800 lines  
**Effort:** 1.5-2 weeks  

**Description:** Create Fortran module with ISO_C_BINDING interfaces to SYCL/oneMKL

**Module structure:**
```fortran
module syclFortMap
   use iso_c_binding
   use math_constants
   implicit none
   save
   
   ! SYCL/oneMKL enumerators
   ! Memory copy directions
   integer*8 :: syclMemcpyDeviceToHost
   integer*8 :: syclMemcpyHostToDevice
   integer*8 :: syclMemcpyDeviceToDevice
   
   ! Sparse matrix operations
   integer*4 :: ONEMKL_SPARSE_OPERATION_NON_TRANSPOSE
   ! ... etc
   
   ! Interface declarations
   interface
      ! oneMKL BLAS interfaces
      integer(c_int) function onemklCreate(...) bind(C)
      ! ... etc
   end interface
   
contains
   ! Helper functions for error checking, etc.
end module syclFortMap
```

**Key components:**

1. **Enumerator definitions** (200-300 lines)
   - Memory operations
   - Sparse matrix types and operations
   - Data types and indexing
   - Algorithm selections

2. **oneMKL BLAS interfaces** (400-500 lines)
   - Dense vector operations (dot, axpy, scal, nrm2)
   - Complex arithmetic operations
   - Level 2/3 BLAS if needed

3. **oneMKL Sparse interfaces** (600-700 lines)
   - Sparse matrix creation and destruction
   - SpMV (Sparse Matrix-Vector multiplication)
   - SpSV (Sparse Solve)
   - ILU factorization and solve
   - Matrix format conversions (CSR)

4. **Device management interfaces** (200-300 lines)
   - Device selection and initialization
   - Memory management (allocation, deallocation, copy)
   - Stream/queue management
   - Event synchronization

5. **Custom kernel interfaces** (200-300 lines)
   - Interfaces to custom SYCL kernels from kernel_c.sycl
   - Type conversion kernels
   - Hadamard product kernels
   - BiCGStab update kernels

**Implementation notes:**
- Use `type(c_ptr)` for opaque handles (queue, device, handles)
- Match parameter ordering with existing CUDA/HIP interfaces
- Maintain compatibility with existing solver.f90 code structure
- Include comprehensive error checking wrappers

---

### Phase 3: Solver Integration (~300-400 LOC)

#### Task 3.1: Add SYCL Variants to solver.f90
**File:** `f90/3D_MT/FWD_SP2/solver.f90` (modifications)  
**Estimated LOC:** 300-400 lines (new code, plus modifications)  
**Effort:** 1 week  

**Changes required:**

1. **Module imports** (~10 lines)
```fortran
#elif defined(SYCL)
   use syclFortMap    ! SYCL GPU API bindings for Fortran
#endif
```

2. **Interface definitions** (~20 lines)
```fortran
interface BICG
#if defined(FG)
    module procedure BiCG
    module procedure BiCGfg
#if defined(CUDA) || defined(HIP) || defined(SYCL)
    module procedure syclBiCGfg
#endif 
#else
    module procedure BiCG
#if defined(CUDA) || defined(HIP) || defined(SYCL)
    module procedure syclBiCG
#endif 
#endif
end interface
```

3. **syclBiCG subroutine** (~200-250 lines)
   - Single GPU SYCL-based BiCGStab solver
   - Mirror structure of cuBiCG
   - Use oneMKL sparse operations
   - Custom kernels for BiCGStab updates

4. **syclBiCGfg subroutine** (~150-200 lines, optional)
   - Multi-GPU SYCL-based BiCGStab with CCL
   - Only if multi-GPU support is required

**Key implementation points:**
- Initialize SYCL queue and device
- Create oneMKL sparse matrix descriptors
- Use oneMKL for SpMV and SpSV operations
- Call custom SYCL kernels for BiCGStab updates
- Handle memory transfers between host and device
- Proper error checking and resource cleanup

---

### Phase 4: Build System Integration (~100-200 LOC)

#### Task 4.1: Create Intel GPU Configuration Scripts
**Files:** 
- `f90/CONFIG/Configure.SP2.Intel.GPU`
- Potentially others for different platforms

**Estimated LOC:** 100-150 lines per configuration  
**Effort:** 3-5 days  

**Configuration script structure:**
```bash
#!/bin/sh
# Configuration for SP2 with Intel GPU support

if [ $2 = "MPI" ]; then
    perl fmkmf.pl -f90 mpiifort \
    -opt '-O2 -g -qopenmp' \
    -mpi '-fpp -DMPI -DSYCL' \
    -o './objs/3D_MT/IFortReleaseMPI_SP2_SYCL' \
    -l '-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
         -lmkl_sycl -lsycl -lpthread -qopenmp' \
    -lp '-L${MKLROOT}/lib/intel64 -L${ONEAPI_ROOT}/compiler/lib/intel64' \
    -p .:MPI:$INV_dir:LAPACK:SENS:UTILS:FIELDS/FiniteDiff3D:$SP_path_dir:$MT_path_dir \
    Mod3DMT.f90 > $1
fi
```

**Key changes:**
- Add `-DSYCL` preprocessor flag
- Link against oneMKL SYCL libraries
- Add SYCL runtime library
- Set appropriate compiler (icpx for DPC++ or ifx for Intel Fortran)

#### Task 4.2: Update fmkmf.pl (if needed)
**File:** `f90/fmkmf.pl`  
**Estimated LOC:** 20-50 lines (modifications)  
**Effort:** 1-2 days  

**Changes:**
- Add support for .sycl file extension
- Add SYCL compiler rules
- Handle mixed Fortran/SYCL compilation

#### Task 4.3: Makefile Rules for SYCL Compilation
**Estimated LOC:** 30-50 lines  
**Effort:** 1 day  

**Add rules for:**
```makefile
# SYCL kernel compilation
%.o: %.sycl
	icpx -fsycl -O3 -c $< -o $@

# Linking with SYCL runtime
SYCL_LIBS = -lsycl -lmkl_sycl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
```

---

### Phase 5: Testing and Validation (~200-300 LOC)

#### Task 5.1: Unit Tests for SYCL Kernels
**File:** `f90/3D_MT/FWD_SP2/syclModOpTest.f90` (new)  
**Estimated LOC:** 150-200 lines  
**Effort:** 3-5 days  

**Tests to create:**
1. Type conversion kernel tests
2. Hadamard product tests
3. AXPY operation tests
4. Vector update operation tests
5. Memory transfer tests
6. Sparse matrix operation tests

#### Task 5.2: Integration Tests
**Estimated LOC:** 50-100 lines  
**Effort:** 3-5 days  

**Tests:**
1. Full BiCGStab solver convergence test
2. Comparison with CPU results
3. Comparison with CUDA/HIP results
4. Performance benchmarking scripts

---

### Phase 6: Documentation (~50-100 LOC)

#### Task 6.1: Update README.md
**File:** `README.md`  
**Estimated LOC:** 20-30 lines  
**Effort:** 2-3 hours  

**Updates:**
- Add Intel GPU support to dependencies section
- Update build instructions for Intel oneAPI
- Add citation for Intel GPU implementation

#### Task 6.2: Create Intel GPU User Guide
**File:** `doc/INTEL_GPU_GUIDE.md` (new)  
**Estimated LOC:** 200-300 lines (markdown)  
**Effort:** 1-2 days  

**Contents:**
- Prerequisites and dependencies
- Building ModEM with Intel GPU support
- Running on Intel GPUs
- Performance tuning tips
- Troubleshooting guide
- Known limitations

#### Task 6.3: Code Comments and Inline Documentation
**Effort:** Ongoing throughout development  

---

## Detailed LOC Breakdown

| Component | File(s) | Estimated LOC | Complexity |
|-----------|---------|---------------|------------|
| SYCL Kernels | kernel_c.sycl | 800-1,000 | Medium |
| Device Management | kernel_c.sycl | 200-300 | Medium |
| Multi-GPU (optional) | kernel_c.sycl | 200-300 | High |
| Fortran Interface | syclFortMap.f90 | 1,700-1,800 | Medium-High |
| Solver Integration | solver.f90 | 300-400 | Medium-High |
| Build System | CONFIG/, fmkmf.pl | 150-250 | Low-Medium |
| Testing | Test files | 200-300 | Medium |
| Documentation | Various | 250-350 | Low |
| **TOTAL** | | **3,800-4,700** | **Medium-High** |

---

## Effort Estimation

### Assumptions
- Developer has experience with:
  - GPU programming (CUDA or similar)
  - Fortran and C/C++ interoperability
  - Sparse matrix algorithms
  - Intel oneAPI tools (preferred but not required)
- Access to Intel GPU hardware (Arc, Flex, or Max series)
- Single developer working full-time

### Time Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: Kernel Development | 2-3 weeks | Core GPU kernels |
| Phase 2: Fortran Interface | 1.5-2 weeks | Most time-consuming |
| Phase 3: Solver Integration | 1 week | Depends on phases 1-2 |
| Phase 4: Build System | 3-5 days | Relatively straightforward |
| Phase 5: Testing | 1 week | Critical for correctness |
| Phase 6: Documentation | 2-3 days | Essential for users |
| **TOTAL** | **6-8 weeks** | Full-time effort |

**Note:** If multi-GPU support (FG mode) is required, add 1-2 additional weeks.

---

## Technical Challenges and Considerations

### 1. SYCL Learning Curve
**Challenge:** SYCL syntax differs from CUDA/HIP  
**Mitigation:** 
- Use Intel's DPC++ Compatibility Tool (dpct) to auto-convert CUDA code
- Reference Intel's oneAPI samples and documentation
- Start with simple kernels and gradually increase complexity

### 2. oneMKL API Differences
**Challenge:** oneMKL sparse API differs from cuSPARSE/hipSPARSE  
**Mitigation:**
- Study oneMKL sparse documentation thoroughly
- Create abstraction layer in Fortran interface
- Test each operation independently

### 3. Memory Model Differences
**Challenge:** SYCL uses Unified Shared Memory (USM) model  
**Mitigation:**
- Use USM device allocations (similar to cudaMalloc)
- Maintain explicit memory management for clarity
- Test memory transfers thoroughly

### 4. Performance Optimization
**Challenge:** Achieving performance parity with CUDA/HIP  
**Mitigation:**
- Profile with Intel VTune or similar tools
- Optimize work-group sizes for Intel architecture
- Use Intel's optimization guides
- Consider Intel-specific optimizations (subgroups, etc.)

### 5. Multi-GPU Communication
**Challenge:** Intel's CCL is less mature than NCCL/RCCL  
**Mitigation:**
- Start with single-GPU support
- Evaluate CCL vs GPU-aware MPI
- May defer multi-GPU to future work

### 6. Fortran-SYCL Interoperability
**Challenge:** Limited examples of Fortran calling SYCL  
**Mitigation:**
- Use standard C interoperability (ISO_C_BINDING)
- Create C wrapper functions if needed
- Test interface functions independently

### 7. Hardware Access
**Challenge:** Need Intel GPU hardware for testing  
**Mitigation:**
- Use Intel DevCloud for testing (free access)
- Work with users who have Intel GPUs
- Emulator for initial development (limited)

---

## Dependencies and Prerequisites

### Software Requirements
1. **Intel oneAPI Base Toolkit** (2024.0 or later)
   - DPC++/C++ Compiler (icpx/dpcpp)
   - Intel Fortran Compiler (ifx) or GNU Fortran
   - oneMKL library with SYCL support
   - Intel MPI (optional, for multi-GPU)

2. **Intel GPU Drivers**
   - Level Zero driver
   - OpenCL driver (fallback)

3. **Development Tools**
   - CMake 3.20+ (if migrating from make)
   - Git for version control

### Hardware Requirements
1. **Intel Arc GPUs** (DG2 family)
   - Arc A-series (A770, A750, A580, A380)
   - Recommended: Arc A770 16GB for development

2. **Intel Data Center GPUs**
   - Flex Series (Arctic Sound-M)
   - Max Series (Ponte Vecchio)

3. **System Requirements**
   - PCIe 4.0 x16 slot
   - Adequate system memory (16GB+ recommended)
   - Modern Linux distribution (Ubuntu 22.04, RHEL 8, etc.)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| SYCL API limitations | Medium | High | Research upfront, plan workarounds |
| Performance below expectations | Medium | Medium | Profile and optimize iteratively |
| Hardware unavailability | Low | High | Use Intel DevCloud for testing |
| oneMKL sparse ops issues | Medium | High | Test extensively, consider alternatives |
| Integration complexity | Low | Medium | Follow existing CUDA/HIP patterns |
| Multi-GPU communication | High | Medium | Make optional, defer if needed |
| Compiler bugs/limitations | Low | High | Report to Intel, find workarounds |

---

## Success Criteria

### Minimum Viable Product (MVP)
1. Single GPU support working
2. All core kernels implemented and tested
3. BiCGStab solver converges correctly
4. Results match CPU within numerical tolerance
5. Basic documentation provided

### Full Success
1. MVP achieved
2. Performance within 80-90% of CUDA/HIP implementation
3. Multi-GPU support (FG mode) working
4. Comprehensive test suite passing
5. Complete user documentation
6. Example workflows demonstrated

### Stretch Goals
1. Performance parity or better than CUDA/HIP
2. Support for Intel GPU-specific optimizations
3. Integration with Intel's profiling tools
4. Contributed to Intel oneAPI samples

---

## Maintenance Considerations

### Ongoing Effort
- Update for new oneAPI releases (~1-2 days per release)
- Performance optimization based on user feedback (~1 week per quarter)
- Bug fixes and improvements (variable)

### Testing Matrix
Need to maintain testing on:
- Multiple Intel GPU generations (Arc, Flex, Max)
- Multiple oneAPI toolkit versions
- Multiple OS distributions
- CPU-only, single-GPU, and multi-GPU configurations

---

## Alternative Approaches

### Alternative 1: OpenCL Backend
**Pros:** More mature, wider hardware support  
**Cons:** Less performant, more verbose, deprecated by Intel  
**Recommendation:** Not recommended

### Alternative 2: Pure oneMKL (no custom kernels)
**Pros:** Simpler, more maintainable  
**Cons:** May not support all required operations  
**Recommendation:** Evaluate feasibility during Phase 1

### Alternative 3: SYCL-BLAS Library
**Pros:** Higher-level abstractions  
**Cons:** May not be well-maintained, limited adoption  
**Recommendation:** Monitor but don't rely on

---

## Conclusion

Adding Intel GPU support to ModEM is feasible and follows established patterns from CUDA/HIP implementations. The estimated effort of 6-8 weeks for a full-time experienced developer is realistic for single-GPU support. Multi-GPU support would require additional time.

Key success factors:
1. Access to Intel GPU hardware or Intel DevCloud
2. Familiarity with SYCL/DPC++ programming model
3. Understanding of sparse matrix operations
4. Following existing CUDA/HIP implementation patterns
5. Thorough testing at each phase

The modular structure of ModEM's GPU implementation (separate kernel files, Fortran interface modules, conditional compilation) makes this addition relatively clean and maintainable.

---

## References

1. Intel oneAPI Documentation: https://www.intel.com/content/www/us/en/docs/oneapi/
2. oneMKL Sparse BLAS: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/
3. SYCL Specification: https://registry.khronos.org/SYCL/specs/sycl-2020/
4. Intel DevCloud: https://www.intel.com/content/www/us/en/developer/tools/devcloud/
5. ModEM GPU Paper (Dong et al. 2024): https://doi.org/10.1016/j.cageo.2024.105518

---

## Appendix: File Structure

```
f90/3D_MT/FWD_SP2/
├── kernel_c.cu          (existing CUDA kernels)
├── kernel_c.hip         (existing HIP kernels)
├── kernel_c.sycl        (NEW - SYCL kernels)
├── cudaFortMap.f90      (existing CUDA interface)
├── hipFortMap.f90       (existing HIP interface)
├── syclFortMap.f90      (NEW - SYCL interface)
├── solver.f90           (modified - add SYCL variants)
├── EMsolve3D.f90        (minimal changes if any)
├── modelOperator3D.f90  (no changes expected)
└── syclModOpTest.f90    (NEW - unit tests)

f90/CONFIG/
├── Configure.SP2.Intel.GPU       (NEW)
├── Configure.SP2.HDF5.Intel.GPU  (NEW)
└── (other existing configs)

doc/
└── INTEL_GPU_GUIDE.md   (NEW - user guide)
```

---

**Document Version:** 1.0  
**Date:** 2025-12-07  
**Author:** GitHub Copilot Analysis
