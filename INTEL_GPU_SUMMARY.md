# Intel GPU Support - Quick Summary

## Overview
Add Intel GPU support to ModEM's FWD_SP2 solver using Intel oneAPI and SYCL/DPC++.

## Effort Estimate

| Metric | Estimate |
|--------|----------|
| **Total LOC** | 3,800-4,700 lines |
| **Effort** | 6-8 weeks (full-time) |
| **Complexity** | Medium-High |

## Breakdown by Component

### 1. SYCL Kernels (kernel_c.sycl)
- **LOC:** 1,000-1,300
- **Time:** 2-3 weeks
- **Tasks:** Port 10+ GPU kernels from CUDA to SYCL

### 2. Fortran-SYCL Interface (syclFortMap.f90)
- **LOC:** 1,700-1,800
- **Time:** 1.5-2 weeks
- **Tasks:** Create Fortran bindings for oneMKL and SYCL runtime

### 3. Solver Integration (solver.f90)
- **LOC:** 300-400
- **Time:** 1 week
- **Tasks:** Add syclBiCG and syclBiCGfg variants

### 4. Build System
- **LOC:** 150-250
- **Time:** 3-5 days
- **Tasks:** Configure scripts, Makefile rules

### 5. Testing & Documentation
- **LOC:** 450-650
- **Time:** 1.5 weeks
- **Tasks:** Unit tests, integration tests, user guide

## Key Files to Create

1. `f90/3D_MT/FWD_SP2/kernel_c.sycl` - SYCL GPU kernels
2. `f90/3D_MT/FWD_SP2/syclFortMap.f90` - Fortran-SYCL interface
3. `f90/3D_MT/FWD_SP2/syclModOpTest.f90` - Unit tests
4. `f90/CONFIG/Configure.SP2.Intel.GPU` - Build configuration
5. `doc/INTEL_GPU_GUIDE.md` - User documentation

## Key Files to Modify

1. `f90/3D_MT/FWD_SP2/solver.f90` - Add SYCL solver variants
2. `f90/fmkmf.pl` - Add .sycl file support (if needed)
3. `README.md` - Update with Intel GPU information

## Dependencies

### Software
- Intel oneAPI Base Toolkit (2024.0+)
- DPC++ Compiler (icpx/dpcpp)
- oneMKL library with SYCL support
- Intel GPU drivers (Level Zero)

### Hardware
- Intel Arc GPUs (A770, A750, A580, A380)
- Intel Data Center GPUs (Flex, Max series)
- OR Intel DevCloud (free testing)

## Implementation Phases

1. **Phase 1:** SYCL Kernel Development (2-3 weeks)
   - Port kernels from CUDA/HIP
   - Implement device management
   - Optional: Multi-GPU support

2. **Phase 2:** Fortran Interface (1.5-2 weeks)
   - Create syclFortMap.f90
   - Bind to oneMKL operations
   - Interface to custom kernels

3. **Phase 3:** Solver Integration (1 week)
   - Add syclBiCG variants
   - Test with existing solvers

4. **Phase 4:** Build System (3-5 days)
   - Configuration scripts
   - Makefile rules

5. **Phase 5:** Testing (1 week)
   - Unit tests
   - Integration tests
   - Performance benchmarks

6. **Phase 6:** Documentation (2-3 days)
   - User guide
   - README updates

## Technical Challenges

1. **SYCL Learning Curve** - Use dpct tool to help convert CUDA
2. **oneMKL API Differences** - Study docs, test independently
3. **Memory Management** - Use USM, similar to cudaMalloc
4. **Performance** - Profile with VTune, optimize work-groups
5. **Multi-GPU** - CCL less mature than NCCL (optional feature)

## Success Criteria

### MVP (Minimum Viable Product)
- ✓ Single GPU support
- ✓ All kernels working
- ✓ BiCGStab converges correctly
- ✓ Results match CPU
- ✓ Basic documentation

### Full Success
- ✓ MVP achieved
- ✓ 80-90% of CUDA/HIP performance
- ✓ Multi-GPU support (optional)
- ✓ Complete test suite
- ✓ Comprehensive docs

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API limitations | Medium | High | Research upfront |
| Poor performance | Medium | Medium | Profile & optimize |
| Hardware access | Low | High | Use DevCloud |
| oneMKL issues | Medium | High | Test extensively |
| Multi-GPU comm | High | Medium | Make optional |

## Code Structure Pattern

Following the existing CUDA/HIP pattern:

```fortran
! In solver.f90
#if defined(CUDA)
   use cudaFortMap
#elif defined(HIP)
   use hipFortMap
#elif defined(SYCL)
   use syclFortMap  ! NEW
#endif

interface BICG
#if defined(CUDA)
   module procedure cuBiCG
#elif defined(HIP)
   module procedure hipBiCG  
#elif defined(SYCL)
   module procedure syclBiCG  ! NEW
#endif
end interface
```

## Quick Start for Implementation

1. **Set up environment:**
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. **Use Intel's conversion tool (optional):**
   ```bash
   dpct kernel_c.cu -o kernel_c.sycl
   ```

3. **Create minimal syclFortMap.f90:**
   - Copy cudaFortMap.f90 as template
   - Replace CUDA APIs with SYCL/oneMKL equivalents

4. **Start with simple kernels:**
   - real64to32 (type conversion)
   - hada_real (Hadamard product)
   
5. **Test incrementally:**
   - Build after each kernel
   - Test against CPU results

6. **Add solver integration last:**
   - Only after kernels are working
   - Test thoroughly with real problems

## Resources

- **Main TODO:** See `INTEL_GPU_TODO.md` for detailed plan
- **Intel Docs:** https://www.intel.com/content/www/us/en/docs/oneapi/
- **DevCloud:** https://www.intel.com/content/www/us/en/developer/tools/devcloud/
- **ModEM Paper:** Dong et al. (2024) - Computers & Geosciences

## Notes

- Multi-GPU support (FG mode) is **optional** and can be deferred
- Focus on single-GPU first to establish foundation
- Maintain consistency with existing CUDA/HIP implementations
- Use preprocessor flags (#ifdef SYCL) for conditional compilation
- Keep same function signatures as CUDA/HIP for easy integration

---

For complete details, see **INTEL_GPU_TODO.md**
