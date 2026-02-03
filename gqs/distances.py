"""Geometric Quantum States (GQS) toolkit.

This package contains utilities for:
- building spin-coherent product states,
- evolving many-body dynamics (e.g., kicked top),
- extracting environment-conditioned (GQS) ensembles,
- computing ensemble distances (Wasserstein/OT),
- and computing sensitivity diagnostics (Gamma).

Modules are split for clarity and easier reuse.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
try:
    import ot  # POT (Python Optimal Transport)
except ImportError as e:
    ot = None

def Psi_Dist(psi0,psi1):
    f1  = np.dot(psi0,np.transpose(np.conj(psi1)))
    f2  = np.dot(psi1,np.transpose(np.conj(psi0)))
    f3  = np.dot(psi0,np.transpose(np.conj(psi0)))
    f4  = np.dot(psi1,np.transpose(np.conj(psi1)))
    d1 = np.abs(np.arccos(np.sqrt(f1*f2/(f3*f4))) )
    return d1

def Dist_ij(chi_alpha_1, chi_alpha_2):
    len_i = np.shape(chi_alpha_1)[0]
    len_j = np.shape(chi_alpha_2)[0]
    D_ij = np.zeros((len_i,len_j))
    for i in range(len_i):
        for j in range(len_j):
            D_ij[i][j]=Psi_Dist(chi_alpha_1[i],chi_alpha_2[j])
    return D_ij

def _mask_chi_lambda(chi, lam, eps=1e-12, renormalize=True):
    lam = np.asarray(lam).reshape(-1)
    chi = np.asarray(chi)
    if chi.shape[0] != lam.size:
        raise ValueError("chi and lambda must have matching first dimension")

    mask = lam > eps
    chi_m = chi[mask]
    lam_m = lam[mask]

    if renormalize:
        s = lam_m.sum()
        if s > 0:
            lam_m = lam_m / s
    return chi_m, lam_m

def Quantum_EMD(chi_1, lambda_1, chi_2, lambda_2, eps=1e-12):
    # Mask *here* (late), keeping only nonzero-probability support
    chi_1, lambda_1 = _mask_chi_lambda(chi_1, lambda_1, eps=eps, renormalize=True)
    chi_2, lambda_2 = _mask_chi_lambda(chi_2, lambda_2, eps=eps, renormalize=True)

    # Pairwise distances only on the reduced supports
    M = Dist_ij(chi_1, chi_2)

    if ot is None:
        raise ImportError(
            "POT (package 'POT', imported as 'ot') is required for Quantum_EMD. "
            "Install with: pip install POT"
        )

    # Wasserstein distance
    return ot.emd2(lambda_1, lambda_2, M)

def bures_distance(rho, sigma, *, check=True, eps=1e-12):
    """
    Compute the Bures distance between two density matrices rho and sigma.

    D_B(rho, sigma) = sqrt( 2 * (1 - sqrt(F)) )
    where F is the Uhlmann fidelity:
        F(rho, sigma) = (Tr(sqrt( sqrt(rho) sigma sqrt(rho) )))^2

    Parameters
    ----------
    rho, sigma : (d, d) array_like (complex)
        Density matrices (Hermitian, PSD, trace ~ 1).
    check : bool
        If True, run basic sanity checks.
    eps : float
        Numerical tolerance for Hermiticity/trace checks and clipping.

    Returns
    -------
    float
        Bures distance in [0, sqrt(2)].
    """
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)

    if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho and sigma must be square matrices of the same shape.")

    if check:
        # Hermiticity
        if not (np.allclose(rho, rho.conj().T, atol=1e-10) and np.allclose(sigma, sigma.conj().T, atol=1e-10)):
            raise ValueError("rho and sigma must be Hermitian.")
        # Trace ~ 1
        if not (abs(np.trace(rho) - 1) < 1e-8 and abs(np.trace(sigma) - 1) < 1e-8):
            raise ValueError("rho and sigma should have trace 1 (within tolerance).")

    # sqrt(rho)
    sqrt_rho = sqrtm(rho)

    # A = sqrt(rho) sigma sqrt(rho)
    A = sqrt_rho @ sigma @ sqrt_rho

    # sqrt(A)
    sqrt_A = sqrtm(A)

    # fidelity amplitude = Tr(sqrt(A)) should be real for valid density matrices,
    # but due to numerics may have tiny imaginary part
    tr = np.trace(sqrt_A)
    tr_real = float(np.real_if_close(tr, tol=1000).real)

    # Fidelity F = (Tr(sqrt(A)))^2, clamp to [0,1]
    F = tr_real * tr_real
    F = min(1.0, max(0.0, F))

    # Bures distance
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - np.sqrt(F)))))
