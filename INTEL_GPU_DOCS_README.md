# Intel GPU Support Documentation

This directory contains comprehensive planning and reference documentation for adding Intel GPU support to ModEM using Intel oneAPI and SYCL/DPC++.

## Documents Overview

### 📋 [INTEL_GPU_CHECKLIST.md](INTEL_GPU_CHECKLIST.md)
**Purpose:** Day-to-day implementation tracking  
**Audience:** Developer implementing the feature  
**Use when:** During active development to track progress

A detailed, actionable checklist organized into 9 phases with checkboxes for tracking completion. Use this to guide daily work and ensure nothing is missed.

**Contains:**
- Environment setup tasks
- Kernel development tasks (with checkboxes)
- Interface development tasks
- Testing requirements
- Documentation tasks
- Progress tracking sections

---

### 📘 [INTEL_GPU_TODO.md](INTEL_GPU_TODO.md)
**Purpose:** Comprehensive implementation plan  
**Audience:** Project planners, technical leads, developers  
**Use when:** Planning the project, understanding scope, estimating resources

The master planning document with complete technical details, estimates, and analysis.

**Contains:**
- Executive summary (LOC: 3,800-4,700, Effort: 6-8 weeks)
- Current state analysis
- Detailed implementation plan (6 phases)
- Per-task LOC estimates and time requirements
- Technical challenges and mitigation strategies
- Risk assessment matrix
- Success criteria
- Alternative approaches
- Complete code examples
- File structure diagrams

**Key Sections:**
1. Phase 1: SYCL Kernel Development (1,000-1,300 LOC, 2-3 weeks)
2. Phase 2: Fortran-SYCL Interface (1,700-1,800 LOC, 1.5-2 weeks)
3. Phase 3: Solver Integration (300-400 LOC, 1 week)
4. Phase 4: Build System (150-250 LOC, 3-5 days)
5. Phase 5: Testing & Validation (200-300 LOC, 1 week)
6. Phase 6: Documentation (250-350 LOC, 2-3 days)

---

### 📄 [INTEL_GPU_SUMMARY.md](INTEL_GPU_SUMMARY.md)
**Purpose:** Quick reference and executive summary  
**Audience:** Project stakeholders, quick reference during development  
**Use when:** Need quick facts, presenting to others, refreshing memory

A condensed summary focusing on key numbers and quick-start information.

**Contains:**
- Effort estimate table
- Component breakdown
- Files to create/modify
- Dependencies list
- Implementation phases (condensed)
- Quick start guide
- Risk assessment table
- Success criteria checklist
- Code structure pattern

---

### 🔄 [CUDA_HIP_SYCL_MAPPING.md](CUDA_HIP_SYCL_MAPPING.md)
**Purpose:** Technical API reference  
**Audience:** Developer writing SYCL code  
**Use when:** Converting CUDA/HIP code to SYCL, looking up API equivalents

A comprehensive API mapping guide showing how to translate CUDA/HIP concepts to SYCL.

**Contains:**
- Side-by-side API comparison tables
- Device management mappings
- Memory operation equivalents
- Kernel launch pattern conversions
- Thread indexing patterns
- BLAS operation mappings (cuBLAS → oneMKL)
- Sparse operation mappings (cuSPARSE → oneMKL)
- Complete kernel conversion examples
- Common pitfalls and solutions
- Useful SYCL code patterns
- Fortran interface examples

---

## Quick Start Guide

### For Project Planning
1. Read **INTEL_GPU_SUMMARY.md** for overview
2. Review **INTEL_GPU_TODO.md** for complete plan
3. Assess risks and resources
4. Decide on timeline

### For Implementation
1. Print or bookmark **INTEL_GPU_CHECKLIST.md**
2. Keep **CUDA_HIP_SYCL_MAPPING.md** open while coding
3. Refer to **INTEL_GPU_TODO.md** for detailed requirements
4. Update checklist daily

### For Code Review
1. Use **INTEL_GPU_CHECKLIST.md** to verify completeness
2. Check against **INTEL_GPU_TODO.md** success criteria
3. Verify SYCL best practices from **CUDA_HIP_SYCL_MAPPING.md**

---

## Implementation Path

### Recommended Reading Order

1. **First:** INTEL_GPU_SUMMARY.md (5 min read)
   - Get the big picture
   - Understand scope and effort

2. **Second:** INTEL_GPU_TODO.md (30-60 min read)
   - Understand detailed requirements
   - Review technical challenges
   - Study code examples

3. **Third:** CUDA_HIP_SYCL_MAPPING.md (20-30 min read)
   - Learn API mappings
   - Study conversion patterns
   - Note common pitfalls

4. **During Implementation:** INTEL_GPU_CHECKLIST.md
   - Track daily progress
   - Ensure nothing is missed
   - Document issues

---

## Key Numbers at a Glance

| Metric | Estimate |
|--------|----------|
| **New Files** | 5-6 files |
| **Modified Files** | 2-3 files |
| **Total LOC** | 3,800-4,700 lines |
| **Kernels to Port** | 10+ kernels |
| **Phases** | 6 phases (9 with checklist detail) |
| **Effort** | 6-8 weeks full-time |
| **Complexity** | Medium-High |

---

## Prerequisites

### Software Requirements
- Intel oneAPI Base Toolkit (2024.0+)
- DPC++/C++ Compiler (icpx/dpcpp)
- Intel Fortran Compiler (ifx) or GNU Fortran
- oneMKL library with SYCL support
- Intel MPI (optional, for multi-GPU)

### Hardware Requirements
- Intel Arc GPUs (A770, A750, A580, A380) OR
- Intel Data Center GPUs (Flex, Max series) OR
- Intel DevCloud access (free)

### Developer Skills
- GPU programming experience (CUDA or similar)
- Fortran and C/C++ interoperability
- Sparse matrix algorithm knowledge
- Intel oneAPI familiarity (helpful but not required)

---

## File Structure

After implementation, the structure will be:

```
f90/3D_MT/FWD_SP2/
├── kernel_c.cu          (existing - CUDA)
├── kernel_c.hip         (existing - HIP)
├── kernel_c.sycl        ⭐ NEW - SYCL kernels
├── cudaFortMap.f90      (existing - CUDA interface)
├── hipFortMap.f90       (existing - HIP interface)
├── syclFortMap.f90      ⭐ NEW - SYCL interface
├── solver.f90           📝 MODIFIED - add SYCL variants
├── EMsolve3D.f90        (minimal/no changes)
├── modelOperator3D.f90  (no changes)
└── syclModOpTest.f90    ⭐ NEW - unit tests

f90/CONFIG/
├── Configure.SP2.Intel.GPU       ⭐ NEW
└── Configure.SP2.HDF5.Intel.GPU  ⭐ NEW (optional)

doc/
└── INTEL_GPU_GUIDE.md   ⭐ NEW - user guide
```

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Single GPU support working
- ✅ All core kernels implemented and tested
- ✅ BiCGStab solver converges correctly
- ✅ Results match CPU within numerical tolerance
- ✅ Basic documentation provided

### Full Success
- ✅ MVP achieved
- ✅ Performance within 80-90% of CUDA/HIP
- ✅ Multi-GPU support working (optional)
- ✅ Comprehensive test suite passing
- ✅ Complete user documentation
- ✅ Example workflows demonstrated

---

## Support and Resources

### Intel Documentation
- oneAPI Documentation: https://www.intel.com/content/www/us/en/docs/oneapi/
- oneMKL Reference: https://www.intel.com/content/www/us/en/docs/onemkl/
- SYCL Specification: https://registry.khronos.org/SYCL/

### Development Tools
- Intel DevCloud: https://www.intel.com/content/www/us/en/developer/tools/devcloud/
- Intel VTune Profiler: For performance analysis
- DPC++ Compatibility Tool (dpct): For CUDA→SYCL conversion

### ModEM References
- Main README: [README.md](README.md)
- GPU Paper: Dong et al. (2024), Computers & Geosciences
  - DOI: https://doi.org/10.1016/j.cageo.2024.105518

---

## Contact and Contribution

For questions or contributions to this implementation:
1. Open an issue on the ModEM GitHub repository
2. Reference these planning documents
3. Follow the checklist for implementation
4. Submit PR when ready for review

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-07 | Initial comprehensive planning documentation |

---

## License

These planning documents are part of the ModEM project and follow the same license terms as ModEM itself. See the main repository LICENSE file for details.

---

**Happy Coding!** 🚀

Remember: This is a well-structured, feasible project. Follow the checklist, refer to the mapping guide, and you'll have Intel GPU support running in no time.
