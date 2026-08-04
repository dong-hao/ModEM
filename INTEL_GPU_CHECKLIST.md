# Intel GPU Implementation Checklist

This checklist tracks the implementation progress for adding Intel GPU support to ModEM.

## Legend
- [ ] Not Started
- [⚠️] In Progress  
- [✓] Completed
- [🔍] Needs Review
- [❌] Blocked

---

## Phase 1: Environment Setup

### 1.1 Development Environment
- [ ] Install Intel oneAPI Base Toolkit (2024.0+)
- [ ] Verify DPC++ compiler (icpx/dpcpp) installation
- [ ] Install oneMKL library with SYCL support
- [ ] Install Intel GPU drivers (Level Zero)
- [ ] Verify GPU detection: `sycl-ls`
- [ ] Test simple SYCL program compilation
- [ ] Set up Intel VTune (optional, for profiling)

### 1.2 Hardware Access
- [ ] Identify target Intel GPU (Arc/Flex/Max)
- [ ] OR: Set up Intel DevCloud access
- [ ] Verify GPU is accessible from development machine
- [ ] Test GPU compute with simple kernel

### 1.3 Code Repository Setup
- [ ] Create feature branch: `feature/intel-gpu-support`
- [ ] Review existing CUDA implementation (kernel_c.cu)
- [ ] Review existing HIP implementation (kernel_c.hip)
- [ ] Review Fortran interfaces (cudaFortMap.f90, hipFortMap.f90)
- [ ] Understand solver.f90 structure

---

## Phase 2: SYCL Kernel Development

### 2.1 Create kernel_c.sycl
- [ ] Create file: `f90/3D_MT/FWD_SP2/kernel_c.sycl`
- [ ] Add file header and includes
- [ ] Set up namespace and basic structure

### 2.2 Type Conversion Kernels
- [ ] Implement `real64to32()` kernel
- [ ] Implement `real32to64()` kernel
- [ ] Create wrapper `kernelc_d2s_sycl()`
- [ ] Create wrapper `kernelc_s2d_sycl()`
- [ ] Test: Compare output with CUDA/CPU versions

### 2.3 Vector Operation Kernels
- [ ] Implement `hada_real()` - Real Hadamard product
- [ ] Implement `hada_cmplx()` - Complex Hadamard product
- [ ] Create wrapper `kernelc_hadar_sycl()`
- [ ] Create wrapper `kernelc_hadac_sycl()`
- [ ] Test: Verify against CPU implementation

### 2.4 AXPY Operation Kernels
- [ ] Implement `xpby_real()` - Real y = x + b*y
- [ ] Implement `xpby_cmplx()` - Complex y = x + b*y
- [ ] Create wrapper `kernelc_xpbyr_sycl()`
- [ ] Create wrapper `kernelc_xpbyc_sycl()`
- [ ] Test: Verify correctness

### 2.5 BiCGStab Update Kernels
- [ ] Implement `p_update_real()` - Real p update
- [ ] Implement `p_update_cmplx()` - Complex p update
- [ ] Implement `x_update_cmplx()` - Complex x update
- [ ] Create wrapper `kernelc_update_pr_sycl()`
- [ ] Create wrapper `kernelc_update_pc_sycl()`
- [ ] Create wrapper `kernelc_update_xc_sycl()`
- [ ] Test: Verify update operations

### 2.6 Reduction Kernels (if needed)
- [ ] Implement `reduce_real()` kernel
- [ ] Create appropriate wrapper function
- [ ] Test: Verify reduction results

### 2.7 Device Management
- [ ] Implement `cf_hookDev_sycl()` - Device selection
- [ ] Implement `cf_resetFlag_sycl()` - Reset flags
- [ ] Add device property query functions
- [ ] Add error checking utilities
- [ ] Test: Verify device initialization

### 2.8 Kernel Testing
- [ ] Create standalone test program for each kernel
- [ ] Verify numerical accuracy (compare with CPU)
- [ ] Check performance (basic profiling)
- [ ] Fix any bugs found during testing

---

## Phase 3: Fortran-SYCL Interface

### 3.1 Create syclFortMap.f90
- [ ] Create file: `f90/3D_MT/FWD_SP2/syclFortMap.f90`
- [ ] Add module header and ISO_C_BINDING

### 3.2 Define Enumerators
- [ ] Memory copy direction constants
- [ ] Matrix operation constants
- [ ] Matrix type constants
- [ ] Index base constants
- [ ] Fill mode constants
- [ ] Data type constants
- [ ] Algorithm selection constants

### 3.3 Device Management Interfaces
- [ ] Device count and selection
- [ ] Memory allocation/deallocation
- [ ] Memory copy operations (H→D, D→H, D→D)
- [ ] Queue/stream management
- [ ] Event management
- [ ] Synchronization operations

### 3.4 oneMKL BLAS Interfaces
- [ ] Create handle (if needed)
- [ ] Destroy handle (if needed)
- [ ] Set stream/queue
- [ ] Dot product (zdot)
- [ ] Vector norm (znrm2)
- [ ] AXPY (zaxpy)
- [ ] Scale (zscal)
- [ ] Copy (zcopy)

### 3.5 oneMKL Sparse Interfaces
- [ ] Sparse matrix creation (CSR format)
- [ ] Sparse matrix destruction
- [ ] Set matrix data
- [ ] Get matrix data
- [ ] SpMV (sparse matrix-vector multiply)
- [ ] SpSV (sparse solve)
- [ ] ILU factorization setup
- [ ] ILU solve (triangular solves)
- [ ] Buffer size queries

### 3.6 Custom Kernel Interfaces
- [ ] All kernel wrappers from kernel_c.sycl
- [ ] Ensure parameter ordering matches CUDA/HIP
- [ ] Add error checking for each interface

### 3.7 Helper Functions
- [ ] Error code to string conversion
- [ ] Status checking utilities
- [ ] Debugging output functions (optional)

### 3.8 Interface Testing
- [ ] Test each interface independently
- [ ] Verify parameter passing
- [ ] Check error handling
- [ ] Test with simple Fortran program

---

## Phase 4: Solver Integration

### 4.1 Modify solver.f90
- [ ] Add SYCL preprocessor directives
- [ ] Import syclFortMap module
- [ ] Update BICG interface definition

### 4.2 Implement syclBiCG (Single GPU)
- [ ] Create subroutine signature
- [ ] Initialize SYCL queue and device
- [ ] Allocate device memory
- [ ] Create sparse matrix descriptors (A, L, U)
- [ ] Create vector descriptors
- [ ] Setup ILU preconditioner
- [ ] Implement BiCGStab main loop:
  - [ ] rho computation (dot product)
  - [ ] beta computation
  - [ ] p update (custom kernel)
  - [ ] SpMV: v = A*p
  - [ ] alpha computation
  - [ ] s = r - alpha*v
  - [ ] SpMV: t = A*s
  - [ ] omega computation
  - [ ] x update (custom kernel)
  - [ ] r update
  - [ ] Convergence check
- [ ] Handle memory cleanup
- [ ] Error handling and logging

### 4.3 Test syclBiCG
- [ ] Create test case with known solution
- [ ] Verify convergence
- [ ] Compare solution with CPU solver
- [ ] Check iteration count
- [ ] Verify residual norm
- [ ] Test with various matrix sizes
- [ ] Profile performance

### 4.4 Implement syclBiCGfg (Multi-GPU, Optional)
- [ ] Set up MPI+CCL or GPU-aware MPI
- [ ] Implement collective operations
- [ ] Test on multi-GPU system
- [ ] Verify scaling

### 4.5 Integration with EMsolve3D
- [ ] Verify FWDSolve3D calls new solver correctly
- [ ] Test with different solvers (BICG, etc.)
- [ ] Check adjoint solve (if applicable)

---

## Phase 5: Build System Integration

### 5.1 Configuration Scripts
- [ ] Create `f90/CONFIG/Configure.SP2.Intel.GPU`
- [ ] Add compiler flags: `-fsycl -DSYCL`
- [ ] Add library paths for oneMKL and SYCL
- [ ] Add library links: `-lsycl -lmkl_sycl`
- [ ] Test configuration script

### 5.2 Makefile Modifications
- [ ] Update fmkmf.pl for .sycl files (if needed)
- [ ] Add SYCL compilation rules
- [ ] Add linking rules for SYCL/oneMKL
- [ ] Test makefile generation

### 5.3 Build Testing
- [ ] Clean build test
- [ ] Test with different configurations
- [ ] Verify executable runs
- [ ] Check for missing libraries

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests
- [ ] Create `f90/3D_MT/FWD_SP2/syclModOpTest.f90`
- [ ] Test individual kernels
- [ ] Test sparse operations
- [ ] Test solver convergence
- [ ] Document test results

### 6.2 Integration Tests
- [ ] Test with small problem (BLOCK2 example)
- [ ] Compare results with CPU version
- [ ] Compare results with CUDA version (if available)
- [ ] Test forward modeling
- [ ] Test inversion (if applicable)

### 6.3 Performance Testing
- [ ] Benchmark against CPU
- [ ] Benchmark against CUDA (if available)
- [ ] Test with different problem sizes
- [ ] Profile with Intel VTune
- [ ] Identify bottlenecks
- [ ] Optimize if needed

### 6.4 Regression Testing
- [ ] Ensure existing tests still pass
- [ ] Verify no impact on non-GPU code
- [ ] Test fallback to CPU

---

## Phase 7: Documentation

### 7.1 Code Documentation
- [ ] Add comments to kernel_c.sycl
- [ ] Add comments to syclFortMap.f90
- [ ] Document syclBiCG implementation
- [ ] Add function headers with descriptions

### 7.2 User Guide
- [ ] Create `doc/INTEL_GPU_GUIDE.md`
- [ ] Prerequisites section
- [ ] Installation instructions
- [ ] Building instructions
- [ ] Running instructions
- [ ] Troubleshooting section
- [ ] Performance tips
- [ ] Known limitations

### 7.3 Update README
- [ ] Add Intel GPU to supported platforms
- [ ] Update dependencies section
- [ ] Add build instructions for Intel GPU
- [ ] Update citation section (if applicable)

### 7.4 Developer Documentation
- [ ] Document code structure
- [ ] Explain SYCL integration approach
- [ ] Add examples
- [ ] Document testing procedures

---

## Phase 8: Code Review & Finalization

### 8.1 Self Review
- [ ] Review all code for clarity
- [ ] Check for memory leaks
- [ ] Verify error handling
- [ ] Check for edge cases
- [ ] Verify thread safety (if applicable)

### 8.2 Code Quality
- [ ] Run static analysis (if available)
- [ ] Check coding standards
- [ ] Verify consistent naming
- [ ] Remove debug code
- [ ] Clean up commented code

### 8.3 Testing Checklist
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance meets requirements
- [ ] Documentation is complete
- [ ] No compiler warnings

### 8.4 Prepare for Merge
- [ ] Rebase on main branch
- [ ] Resolve any conflicts
- [ ] Squash commits (if needed)
- [ ] Write comprehensive PR description
- [ ] Request code review

---

## Phase 9: Deployment

### 9.1 Release Preparation
- [ ] Create release notes
- [ ] Update version number
- [ ] Tag release
- [ ] Create binary packages (if applicable)

### 9.2 User Communication
- [ ] Announce Intel GPU support
- [ ] Provide usage examples
- [ ] Create tutorial/walkthrough
- [ ] Set up issue tracking for feedback

### 9.3 Maintenance Plan
- [ ] Set up CI for Intel GPU tests
- [ ] Plan for oneAPI updates
- [ ] Monitor user feedback
- [ ] Plan future optimizations

---

## Notes & Issues

### Blockers
_(Document any blocking issues here)_

### Questions
_(Document questions that need answers)_

### Performance Observations
_(Document performance measurements)_

### Future Work
- [ ] Multi-GPU optimization
- [ ] Further performance tuning
- [ ] Support for additional Intel GPU models
- [ ] Integration with Intel's optimization tools

---

## Completion Summary

**Started:** _____________  
**Completed:** _____________  
**Total Time:** _____________  
**Lines Added:** _____________  
**Tests Added:** _____________  

**Performance vs CPU:** _____________  
**Performance vs CUDA:** _____________  

**Key Learnings:**
1. 
2. 
3. 

**Recommendations:**
1. 
2. 
3. 

---

**Checklist Version:** 1.0  
**Last Updated:** 2025-12-07  
**For:** ModEM Intel GPU Implementation
