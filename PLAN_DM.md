# Plan: Joint 3D MT Inversion with Full Galvanic Distortion Matrix in ModEM

**Reference:** Avdeeva, A., Avdeev, D., & Jegen, M. (2015). Three-dimensional inversion of
magnetotelluric impedance tensor data and full distortion matrix. *Geophysical Journal
International*, 202(1), 464–481. https://doi.org/10.1093/gji/ggv144

---

## 1  Background and Mathematical Framework

Galvanic distortion caused by small-scale near-surface heterogeneities or topography
shifts MT impedance tensors in a frequency-independent way.  Instead of removing it as
a pre-processing step, the distortion can be inverted for jointly with the 3-D
conductivity structure.

The joint objective function (Avdeeva et al. 2015, Eq. 2) is:

```
φ(σ, C, λ, ν)  =  φ_d(σ, C)  +  λ φ_s(σ)  +  ν ψ(C)  →  min
```

| Term | Expression | Role |
|------|-----------|------|
| **φ_d** | (1/2) Σ_j Σ_n β_jn tr[ Ā_jn A_jn ] | data misfit |
| **φ_s** | log(σ)ᵀ WᵀW log(σ) | spatial smoothness of conductivity |
| **ψ**  | (1/2) Σ_j tr[ (C_j−I)ᵀ(C_j−I) ] | distortion penalty |

where:

- **A_jn = C_j Z_jn − D_jn** — distorted residual matrix (Eq. 3)
- **C_j** — real 2×2 frequency-independent distortion matrix at site *j* (4 unknowns/site)
- **Z_jn** — complex 2×2 predicted impedance tensor at site *j*, period *n*
- **D_jn** — complex 2×2 observed impedance tensor
- **β_jn** — positive weights derived from estimated data errors
- **λ, ν** — trade-off parameters

Key insight: the gradients ∂φ/∂C_j are **purely algebraic** — no new forward solves
are required.  The σ-gradient requires the same adjoint solves as in the standard
ModEM inversion, but the adjoint sources are modified to include C_j (Eqs. 5–9).

---

## 2  Conductivity Gradient (σ-update)

The gradient ∂φ_d/∂σ_k is (Eq. 5):

```
∂φ_d/∂σ_k  =  ℜ{ Σ_n ∫_{V_k} tr[ u_n^T E_n ] dV }
```

where **E_n** and **H_n** are the forward EM solutions and **u_n**, **v_n** solve the
adjoint system (Eqs. 7a–7b), with modified external sources that include C_j:

```
j_n^ext  =  Σ_j β_jn p^T C_j^T Ā_jn H_jn^{-T} δ(r − r_j)   (Eq. 8)

h_n^ext  =  −(1/iω_n μ) Σ_j β_jn p^T Z_jn^T C_j^T Ā_jn H_jn^{-T} δ(r − r_j)  (Eq. 9)
```

In ModEM terms: the adjoint source assembly lives in `LmultT` (`DataSens.f90`).
Currently it scales sparse row vectors Lz by `conj(residual)`; with distortion it must
instead scale by `C_j^T · conj(A_jn)`.  Similarly `QmultT` handles the magnetic-field
source correction (h_n^ext).

---

## 3  Distortion Gradient (C-update)

The gradient of φ_d with respect to C_xx at site *j* is (Eq. 10):

```
∂φ_d/∂(C_xx)_j  =  ℜ{ Σ_n β_jn tr[ (Z_xx  0 ; Z_yx  0)_jn · Ā_jn ] }
```

Analogous formulae hold for C_xy, C_yx, C_yy.  The full total distortion gradient is:

```
∂φ/∂C_j  =  ∂φ_d/∂C_j  +  ν · (C_j − I)
```

This requires only **Z_jn** (already computed in the forward pass) and the weighted
residual **Ā_jn** — no additional forward or adjoint solves.

---

## 4  Implementation Plan

### Step 1 — New data type: `Full_Impedance_Distorted`

**File:** `f90/3D_MT/DICT/dataTypes.f90`

- Add integer parameter `Full_Impedance_Distorted = 15` (next available after 14).
- Register entry in `setup_typeDict`: name `'Full_Impedance_Distorted'`, abbreviation
  `'ZD'`, `isComplex = .true.`, `nComp = 8`, same component IDs as `Full_Impedance`.
- Update `allocate(typeDict(15), ...)` size in `setup_typeDict`.
- Add `case('Full_Impedance_Distorted','ZD')` to `ImpType`.

**File:** `f90/3D_MT/DICT/txTypes.f90`

- Extend `txTypeDict(MT)%dataTypes` from 6 → 7 entries and add `Full_Impedance_Distorted`.

---

### Step 2 — Per-station distortion matrix storage

**New file:** `f90/3D_MT/DistortionParam.f90`  (new Fortran module `DistortionParam`)

```
module DistortionParam
  use math_constants
  use utilities
  use receivers          ! for nRx (number of receiver sites)
  implicit none
  public :: distortionParam_t, &
            create_distortionParam, deall_distortionParam, &
            zero_distortionParam, identity_distortionParam, &
            dotProd_distortionParam, linComb_distortionParam, &
            scMult_distortionParam, &
            read_distortionParam, write_distortionParam

  type :: distortionParam_t
     integer                                :: nSites = 0
     real(kind=prec), allocatable           :: C(:,:,:)   ! (2,2,nSites)
     integer, allocatable                   :: siteIndex(:) ! rxDict indices
  end type distortionParam_t
  ...
end module DistortionParam
```

Key operations:

| Routine | Description |
|---------|-------------|
| `create_distortionParam(nSites, C)` | allocate and initialise to identity |
| `deall_distortionParam(C)` | deallocate |
| `zero_distortionParam(C)` | set all matrices to zero |
| `identity_distortionParam(C)` | set all matrices to identity |
| `dotProd_distortionParam(C1, C2)` | scalar inner product Σ_j tr(C1_j^T C2_j) |
| `linComb_distortionParam(a,C1,b,C2,C)` | C = a·C1 + b·C2 |
| `scMult_distortionParam(a, C_in, C_out)` | C_out = a · C_in |
| `read_distortionParam(fname, C)` | read ASCII file (site, 2×2 block per site) |
| `write_distortionParam(fname, C)` | write ASCII file |

**File format** (one block per site, one site per line or block):
```
NSITE  <N>
<site_id>  Cxx Cxy Cyx Cyy
...
```

---

### Step 3 — Modified data residual (distorted predicted impedance)

**File:** `f90/3D_MT/DataFunc.f90`

Add a new public subroutine:

```fortran
subroutine dataResp_distorted(ef, Sigma, C_site, iDT, iRX, Resp, Orient, Binv)
  ! Computes distorted predicted response: D_hat = C_j * Z_jn
  ! C_site(2,2): the distortion matrix for this site
  ! Calls dataResp to get Z, then applies C_site
```

Logic:
1. Call existing `dataResp(ef, Sigma, Full_Impedance, iRX, Resp, Orient, Binv)` to
   obtain `Z_jn` stored in `Resp(1:8)` (real/imag interleaved).
2. Reconstruct complex 2×2 matrix `Z` from `Resp`.
3. Compute `D_hat = C_site * Z` (real 2×2 × complex 2×2 matrix multiply).
4. Copy `D_hat` back into `Resp` in the same real/imag interleaved layout.

When `iDT == Full_Impedance_Distorted` the result is the distorted predicted
impedance `Ẑ_jn = C_j Z_jn`; if `C_site = I` this is identical to standard
`Full_Impedance`.

**File:** `f90/SENS/SensComp.f90` (or `DataSens.f90`)

Modify `fwdPred_TX` to call `dataResp_distorted` when
`d%data(j)%dataType == Full_Impedance_Distorted`, passing the relevant 2×2
slice of `distortionParam_t%C(:,:,iSite)`.  A module-level pointer to the
`distortionParam_t` (set before the inversion loop begins) is the cleanest
approach to avoid changing every call signature.

---

### Step 4 — Modified adjoint sources (σ-gradient)

**File:** `f90/SENS/DataSens.f90`

**`LmultT`**: Currently assembles the adjoint comb by accumulating:
```fortran
Z = cmplx(residual_real, -residual_imag, 8)   ! conj of residual
call add_sparseVrhsV(Z, Lz(iFunc), comb)
```

With distortion the weight must be premultiplied by **C_j^T**:
```
weight_func = Σ_i (C_j^T)_{func,i} * conj(A_jn)_i
```

Modification: after extracting the 4 complex residuals for site *j* into a local 2×2
complex matrix `A_bar`, compute `W = C_j^T * A_bar` (real-times-complex 2×2 multiply)
and use `W(iFunc,...)` as the weight for `add_sparseVrhsV`.

Implementation strategy: add an optional module-level `distortionParam_t` pointer
(analogous to how `eAll` is stored as a module variable in `SensComp`).  `LmultT`
checks whether the module pointer is associated; if not, falls back to current
behaviour (C=I path), ensuring **backward compatibility**.

**`QmultT`**: The h_n^ext source has an extra `Z_jn^T C_j^T` factor compared to the
undistorted case.  Because `QmultT` operates on parameter space (not EM solution
space) and involves the derivative of the data functional with respect to conductivity
directly, apply an analogous C_j^T weighting to the residuals before the Qrows
accumulation.

---

### Step 5 — Distortion gradient module

**New file:** `f90/INV/DistortGradient.f90`  (new module `DistortGradient`)

```
module DistortGradient
  use DistortionParam
  use DataSpace
  use dataTypes
  implicit none
  public :: compute_distortion_gradient

Contains

  subroutine compute_distortion_gradient(d_obs, d_pred, distC, nu, gradC)
    ! Inputs:
    !   d_obs   : observed distorted data (dataVectorMTX_t)
    !   d_pred  : predicted distorted data (dataVectorMTX_t) = C * Z
    !   distC   : current distortion matrices (distortionParam_t)
    !   nu      : distortion regularization weight
    ! Output:
    !   gradC   : gradient w.r.t. distortion matrices (distortionParam_t)
    ...
  end subroutine compute_distortion_gradient

end module DistortGradient
```

For each site *j* and frequency *n*:

1. Extract predicted `Z_jn` from `d_pred` (complex 2×2, available after the
   fwdPred call with distortion).
2. Compute weighted residual `Ā_jn = conj(C_j Z_jn − D_jn) / σ²_jn`.
3. Apply Eq. 10 for each element of C_j:
   - `∂φ_d/∂(C_xx)_j += β_jn ℜ{ tr[ (Z_xx 0; Z_yx 0) · Ā_jn ] }`
   - Analogous for C_xy, C_yx, C_yy
4. Sum over all frequencies *n*.
5. Add regularization gradient: `gradC_j += ν · (C_j − I)`.

**Note:** Z_jn must be stored as a separate data vector (undistorted) during the
forward pass so that it is available here.  One approach: store a `d_undistorted`
alongside `d_pred`, or extract Z_jn from the raw forward solution before applying C_j.

---

### Step 6 — Extended inversion core

**File:** `f90/INV/INVcore.f90`

Add the following new public routines (not changing existing ones):

| Routine | Description |
|---------|-------------|
| `func_distorted(lambda, nu, d, m0, mHat, distC, F, mNorm, distNorm, dHat, eAll, RMS)` | full φ including distortion terms |
| `gradient_distorted(lambda, nu, d, m0, mHat, distC, gradM, gradC, dHat, eAll)` | both σ and C gradients |
| `psi_C(distC)` | returns ψ(C) = (1/2) Σ_j ‖C_j−I‖²_F |

`func_distorted` logic:
1. Run `fwdPred` with distortion active to get `dHat = C Z`.
2. Compute residual `res = d − dHat`.
3. Compute `SS = dotProd(res, CdInv·res)`, `Ndata`, `mNorm`, `distNorm = psi_C(distC)`.
4. `F = SS/Ndata + lambda * mNorm/Nmodel + nu * distNorm`.

`gradient_distorted` logic:
1. Compute σ-gradient via `JmultT` with modified adjoint sources (distortion weighting
   injected into the `LmultT`/`QmultT` path, Step 4).
2. Compute C-gradient via `compute_distortion_gradient` (Step 5) — no extra forward
   solves.
3. Return both `gradM` (modelParam_t) and `gradC` (distortionParam_t).

---

### Step 7 — New solver: alternating NLCG

**New file:** `f90/INV/NLCG_distorted.f90`  (new module `NLCG_distorted`)

Implements an **alternating-update** strategy as the primary solver
(simpler to implement, valid 2-block coordinate descent):

```
Outer loop (outer_iter = 1 … maxIter):
  ┌─ σ-update (inner NLCG sweep, nSigmaIter steps):
  │    gradient_distorted → gradM (σ-gradient with distortion-weighted adjoint sources)
  │    NLCG line search on φ(σ, C_fixed, λ) using existing lineSearchCubic
  │
  └─ C-update (nDistIter gradient steps):
       compute_distortion_gradient → gradC
       gradient descent: C_j ← C_j − α_C · gradC_j
       (simple Armijo backtracking sufficient: C-gradient is cheap)

  Converge when RMS ≤ rmsTol or |ΔRMS| < fdiffTol
```

Control type `NLCGDistControl_t` extends `NLCGiterControl_t` with:

| Field | Default | Description |
|-------|---------|-------------|
| `nu` | 1.0 | distortion regularization weight |
| `nSigmaIter` | 5 | σ-update iterations per outer loop |
| `nDistIter` | 3 | C-update iterations per outer loop |
| `alpha_C` | 0.1 | initial step size for C-update |
| `distFname` | `'distortion.dat'` | output distortion file |
| `init_distFname` | `''` | input distortion file (empty = start from I) |

Public entry point:
```fortran
subroutine NLCGsolver_distorted(d, lambda, nu, m0, m, distC, fname)
```

---

### Step 8 — I/O and configuration

**File:** `f90/Mod3DMT.f90`

- Add `use NLCG_distorted` to the use-association block.
- Add a new dispatch branch:
  ```fortran
  elseif (trim(cUserDef%search) == 'NLCG_DIST') then
      write(*,*) 'Starting the NLCG_DIST search (joint distortion)...'
      sigma1 = dsigma
      call NLCGsolver_distorted(allData, cUserDef%lambda, cUserDef%nu_dist, &
                                 sigma0, sigma1, distC, cUserDef%rFile_invCtrl)
  ```
- Read `nu_dist` and optional `rFile_distortion` from the startup file (UserCtrl).

**File:** `f90/3D_MT/DataIO.f90`

- Add recognition of the `'Full_Impedance_Distorted'` / `'ZD'` tag in the data file
  reader so that distorted observed data can be loaded directly.  The on-disk format
  is identical to `Full_Impedance`.

**File:** `f90/UserCtrl.f90` (or the existing startup file reader)

- Add optional fields `nu_dist` (default 1.0) and `rFile_distortion` (default `''`).

---

### Step 9 — Build system

**Files to update:**

| File | Change |
|------|--------|
| `f90/3D_MT/CMakeLists.txt` | add `DistortionParam.f90` |
| `f90/INV/CMakeLists.txt` | add `DistortGradient.f90`, `NLCG_distorted.f90` |
| `f90/Makefile.3D.MF.gnu` | add object files for the same three new sources |
| `f90/Makefile.3D.MF.intel` | same |

---

### Step 10 — Testing

**Verification sequence:**

1. **Regression test (C = I):** Run inversion on any existing benchmark with
   `NLCG_DIST` and distortion matrices initialised to identity.  The result must be
   numerically identical (to machine precision) to a standard `NLCG` run.

2. **Gradient check:** Apply a synthetic 2×2 distortion to the COMMEMI 3D-2A benchmark
   (as in Avdeeva et al. 2015 Section 3.1) and confirm that the analytic C-gradient
   matches a finite-difference gradient.

3. **Recovery test:** Invert the synthetically distorted dataset.  The joint inversion
   should recover both the conductivity model and the distortion matrices; a σ-only
   inversion on the same data should show artefacts.

4. **MT3DINV benchmark:** Apply to the MT3DINV workshop dataset provided by the user
   once available.

---

## 5  Key Design Decisions

| Paper approach (x3Di / L-BFGS) | ModEM adaptation |
|---------------------------------|-----------------|
| Full J stored explicitly | J implicit via Jmult/JmultT; C weighting injected into `LmultT` adjoint sources |
| C_j per-site, frequency-independent | New `distortionParam_t` type, decoupled from `modelParam_t` to preserve all existing solver infrastructure |
| Full joint L-BFGS step | Phase 1: alternating NLCG (2-block coordinate descent) — simpler and extensible |
| φ_d uses trace formulation | Algebraically equivalent to weighted sum-of-squares; maps to existing `dotProd`/`CdInvMult` framework with modified residuals |
| C gradient: analytic, no extra solves | Computed in `DistortGradient.f90` using Z_jn from the forward pass |

---

## 6  File Summary

| File | Status | Description |
|------|--------|-------------|
| `f90/3D_MT/DICT/dataTypes.f90` | **modify** | add `Full_Impedance_Distorted = 15` |
| `f90/3D_MT/DICT/txTypes.f90` | **modify** | register new data type for MT |
| `f90/3D_MT/DistortionParam.f90` | **new** | distortion matrix type and operations |
| `f90/3D_MT/DataFunc.f90` | **modify** | add `dataResp_distorted` |
| `f90/SENS/DataSens.f90` | **modify** | C_j weighting in `LmultT` / `QmultT` |
| `f90/SENS/SensComp.f90` | **modify** | route `Full_Impedance_Distorted` through distorted fwdPred |
| `f90/INV/DistortGradient.f90` | **new** | analytic C-gradient (Eq. 10) |
| `f90/INV/INVcore.f90` | **modify** | `func_distorted`, `gradient_distorted`, `psi_C` |
| `f90/INV/NLCG_distorted.f90` | **new** | alternating NLCG solver |
| `f90/Mod3DMT.f90` | **modify** | dispatch `NLCG_DIST`, read ν and distortion file |
| `f90/3D_MT/DataIO.f90` | **modify** | read/write `Full_Impedance_Distorted` |
| `f90/UserCtrl.f90` | **modify** | add `nu_dist`, `rFile_distortion` fields |
| `f90/3D_MT/CMakeLists.txt` | **modify** | add `DistortionParam.f90` |
| `f90/INV/CMakeLists.txt` | **modify** | add `DistortGradient.f90`, `NLCG_distorted.f90` |
| `f90/Makefile.3D.MF.gnu` | **modify** | add new object files |
| `f90/Makefile.3D.MF.intel` | **modify** | add new object files |
