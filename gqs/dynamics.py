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
from scipy.linalg import expm

from .operators import sy, sz, Spin

def Hamiltonian_QK(tau, kappa, nqubit, s_y=sy, s_z=sz):
    """
    Kicked-top Hamiltonian pieces in standard angular-momentum convention:

        H(t) = (pi/(2*tau)) * J_y  +  (kappa/(2j)) * J_z^2 * sum_n delta(t - n*tau)

    Returns
    -------
    H1 : ndarray (dim, dim)
         continuous/free part (acts for duration tau)
    H2 : ndarray (dim, dim)
         kick generator (acts instantaneously at kicks)
    """
    # total spin quantum number corresponding to N spin-1/2's
    j = nqubit / 2.0

    _, JY = Spin(nqubit, s_y)
    _, JZ = Spin(nqubit, s_z)

    H1 = (np.pi / (2.0 * tau)) * JY
    H2 = (kappa / (2.0 * j)) * (JZ @ JZ)

    return H1, H2

def floquet_operator_from_H(H1, H2, tau, hbar=1.0, order="free_then_kick"):
    """
    Build the Floquet operator for a delta-kicked Hamiltonian:
        H(t) = H1 + H2 * sum_n delta(t - n*tau)

    One-period map (stroboscopic):
        U_free = exp(-i * H1 * tau / hbar)
        U_kick = exp(-i * H2 / hbar)

    order:
      - "free_then_kick": psi_{n+1} = U_kick @ U_free @ psi_n
      - "kick_then_free": psi_{n+1} = U_free @ U_kick @ psi_n
    """
    U_free = expm((-1j / hbar) * H1 * tau)
    U_kick = expm((-1j / hbar) * H2)

    if order == "free_then_kick":
        return U_kick @ U_free
    elif order == "kick_then_free":
        return U_free @ U_kick
    else:
        raise ValueError("order must be 'free_then_kick' or 'kick_then_free'")
