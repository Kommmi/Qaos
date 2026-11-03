"""
Quantum kicked top in the permutation-symmetric (Dicke) subspace.

Why this works
--------------
The kicked-top Floquet operator

    U_F = exp[-i kappa/(2j) J_z^2] exp[-i (pi/2) J_y],   j = n/2,

is built only from collective spin operators J = sum_i S_i, so it commutes with
every permutation of the n qubits. A spin-coherent product state is permutation
symmetric, so the evolved state stays in the symmetric subspace spanned by the
n+1 Dicke states

    |D_k> = C(n,k)^{-1/2} * sum_{strings with k ones} |s_1 ... s_n>,  k = 0..n.

So we evolve n+1 amplitudes c_k instead of 2^n:

    |psi> = sum_k c_k |D_k>.

Convention (matches gqs.operators / gqs.states): |0> has S_z = +1/2, so
|D_k> has J_z = n/2 - k, and site 0 is the leftmost tensor factor.

Projected ensemble of one qubit
-------------------------------
Measure the other n-1 qubits in the computational basis. Every outcome with m
ones gives the same conditional state, so the ensemble has at most n distinct
members:

    |psi_m>  ∝  ( sqrt((n-m)/n) c_m ,  sqrt((m+1)/n) c_{m+1} ),
    w_m      =  ((n-m)/n) |c_m|^2 + ((m+1)/n) |c_{m+1}|^2 ,       m = 0..n-1,

where w_m is the total probability of all C(n-1, m) outcomes with m ones. No
binomial coefficient larger than n appears, so this is stable for n in the
thousands.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------- operators
def collective_ops(n: int):
    """J_z, J_y, J_x in the Dicke basis k = 0..n (k = number of |1>'s)."""
    j = n / 2.0
    m = j - np.arange(n + 1)                         # J_z eigenvalue of |D_k>
    Jz = np.diag(m).astype(complex)
    # J_+ |j,m> = sqrt(j(j+1) - m(m+1)) |j,m+1>, and m+1 corresponds to k-1
    Jp = np.zeros((n + 1, n + 1), dtype=complex)
    for k in range(1, n + 1):
        mk = m[k]
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - mk * (mk + 1))
    Jm = Jp.conj().T
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)
    return Jz, Jy, Jx


def floquet_operator(n: int, kappa: float, tau: float = 1.0, order: str = "free_then_kick"):
    """
    Kicked-top Floquet operator in the Dicke basis, same convention as
    gqs.dynamics.floquet_operator_from_H(Hamiltonian_QK(tau, kappa, n), tau):

        U_free = exp[-i (pi/2) J_y],   U_kick = exp[-i kappa/(2j) J_z^2].
    """
    j = n / 2.0
    Jz, Jy, _ = collective_ops(n)
    evals, V = np.linalg.eigh(Jy)
    U_free = (V * np.exp(-1j * (np.pi / (2.0 * tau)) * tau * evals)) @ V.conj().T
    U_kick = np.diag(np.exp(-1j * (kappa / (2.0 * j)) * np.real(np.diag(Jz)) ** 2))
    if order == "free_then_kick":
        return U_kick @ U_free
    if order == "kick_then_free":
        return U_free @ U_kick
    raise ValueError("order must be 'free_then_kick' or 'kick_then_free'")


# ------------------------------------------------------------------- states
def _log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def coherent_state(n: int, theta: float, phi: float) -> np.ndarray:
    """
    Dicke amplitudes of (cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>)^{⊗n},
    the same state as gqs.states.Initial_state(n, theta, phi):

        c_k = sqrt(C(n,k)) cos^{n-k}(theta/2) sin^k(theta/2) e^{i k phi}.

    Computed in log space so it is stable for large n.
    """
    k = np.arange(n + 1)
    a, b = np.cos(theta / 2.0), np.sin(theta / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mag = 0.5 * _log_binom(n, k)
        log_mag = log_mag + np.where(n - k > 0, (n - k) * np.log(np.abs(a)), 0.0)
        log_mag = log_mag + np.where(k > 0, k * np.log(np.abs(b)), 0.0)
    mag = np.exp(log_mag)
    sign = np.sign(a) ** (n - k) * np.sign(b) ** k
    sign = np.where((n - k == 0) | (a != 0), sign, 0.0)
    sign = np.where((k == 0) | (b != 0), sign, 0.0)
    c = mag * sign * np.exp(1j * k * phi)
    return c / np.linalg.norm(c)


def evolve(c0: np.ndarray, U: np.ndarray, N_kicks: int):
    """Return the list [c(0), c(1), ..., c(N_kicks)] of Dicke amplitudes."""
    out = [c0]
    for _ in range(N_kicks):
        out.append(U @ out[-1])
    return out


def to_full_state(c: np.ndarray) -> np.ndarray:
    """Expand Dicke amplitudes to the 2^n state vector (small n only, for validation)."""
    n = len(c) - 1
    idx = np.arange(2**n)
    k = np.array([bin(i).count("1") for i in idx])
    return c[k] / np.sqrt(np.exp(_log_binom(n, k)))


# ------------------------------------------------------ projected ensemble
def projected_ensemble(c: np.ndarray, eps: float = 0.0):
    """
    Exact projected ensemble (GQS) of one qubit, all others measured in Z.

    Returns
    -------
    chi : ndarray, shape (n, 2)
        Normalized conditional states |psi_m>, m = number of ones measured.
    w : ndarray, shape (n,)
        Total probability of each class m (sums to 1).
    """
    n = len(c) - 1
    m = np.arange(n)
    amp = np.column_stack([np.sqrt((n - m) / n) * c[:-1], np.sqrt((m + 1) / n) * c[1:]])
    w = np.sum(np.abs(amp) ** 2, axis=1)
    chi = np.zeros_like(amp)
    nz = w > eps
    chi[nz] = amp[nz] / np.sqrt(w[nz])[:, None]
    return chi, w


def reduced_density_matrix(c: np.ndarray) -> np.ndarray:
    """One-qubit reduced density matrix, the average of the projected ensemble."""
    chi, w = projected_ensemble(c)
    return np.einsum("m,mi,mj->ij", w, chi, chi.conj())


# ------------------------------------------------------------- MPS bridge
def to_mps(c: np.ndarray):
    """
    Exact MPS of a symmetric state. The bond index after site i counts how many
    ones have appeared so far, so

        A_[i]^{s}[k, k+s] = 1                     (i < n-1),
        A_[n-1]^{s}[k, 0] = c_{k+s} / sqrt(C(n, k+s)).

    Bond dimensions are at most n, and the true Schmidt rank across the cut after
    site i is min(i+1, n-i-1) + 1, so a symmetric state never needs more than
    n/2 + 1. This is why permutation symmetry keeps entanglement bounded.
    """
    from .mps import MPS

    n = len(c) - 1
    state = MPS(n)
    tensors = []
    for i in range(n - 1):
        A = np.zeros((i + 1, 2, i + 2), dtype=complex)
        for k in range(i + 1):
            A[k, 0, k] = 1.0
            A[k, 1, k + 1] = 1.0
        tensors.append(A)
    A = np.zeros((n, 2, 1), dtype=complex)
    for k in range(n):
        for s in (0, 1):
            K = k + s
            A[k, s, 0] = c[K] / np.sqrt(np.exp(_log_binom(n, K)))
    tensors.append(A)
    state.tensors = tensors
    # Bring into canonical form: sweeping the center from the last site to site 0 makes
    # every site 1..n-1 right-orthonormal, so Schmidt values and sampling are exact.
    state.center = n - 1
    state.move_center(0)
    return state


def hamming_weights(bits: np.ndarray) -> np.ndarray:
    """Number of ones in each measured outcome (rows of a bit array, e.g. from
    gqs.mps.sample_projected_ensemble). Groups samples by class m."""
    return np.asarray(bits).sum(axis=1).astype(np.int64)
