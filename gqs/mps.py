"""
Minimal matrix product state (MPS) circuit simulator for qubit chains.

State representation
--------------------
    |psi> = sum_{s_1..s_n} A[1]^{s_1} A[2]^{s_2} ... A[n]^{s_n} |s_1 ... s_n>

Each tensor A[i] has shape (chi_left, 2, chi_right). Boundary bonds have
dimension 1. Memory is O(n * 2 * chi^2) instead of O(2^n) for a state vector.

The MPS is kept in mixed-canonical form with an orthogonality center
(``self.center``). Tensors left of the center are left-orthonormal and tensors
right of it are right-orthonormal. With this form, truncating the SVD across a
bond is the optimal low-rank approximation of the whole state, and the
discarded singular-value weight equals the infidelity added by that truncation.
"""

from __future__ import annotations

import numpy as np


class MPS:
    def __init__(self, n: int, chi_max: int | None = None, cutoff: float = 1e-12):
        self.n = n
        self.chi_max = chi_max          # None means no truncation (exact)
        self.cutoff = cutoff            # drop singular values below this
        self.tensors = []
        for _ in range(n):              # product state |00...0>
            A = np.zeros((1, 2, 1), dtype=complex)
            A[0, 0, 0] = 1.0
            self.tensors.append(A)
        self.center = 0
        self.discarded = []             # discarded weight per truncation

    # ------------------------------------------------------------------ gauge
    def move_center(self, k: int) -> None:
        """Move the orthogonality center to site k with QR decompositions."""
        while self.center < k:          # sweep right: A[j] = Q, R into A[j+1]
            j = self.center
            l, d, r = self.tensors[j].shape
            Q, R = np.linalg.qr(self.tensors[j].reshape(l * d, r))
            self.tensors[j] = Q.reshape(l, d, Q.shape[1])
            self.tensors[j + 1] = np.einsum("ab,bsc->asc", R, self.tensors[j + 1])
            self.center += 1
        while self.center > k:          # sweep left with an LQ factorization
            j = self.center
            l, d, r = self.tensors[j].shape
            Q, R = np.linalg.qr(self.tensors[j].reshape(l, d * r).T)
            self.tensors[j] = Q.T.reshape(Q.shape[1], d, r)
            self.tensors[j - 1] = np.einsum("asb,bc->asc", self.tensors[j - 1], R.T)
            self.center -= 1

    # ------------------------------------------------------------------ gates
    def apply_1q(self, U: np.ndarray, i: int) -> None:
        """A[i]^{s} <- sum_{s'} U_{s s'} A[i]^{s'}. Bond dimension is unchanged."""
        self.tensors[i] = np.einsum("st,atb->asb", U, self.tensors[i])

    def apply_2q(self, U: np.ndarray, i: int) -> None:
        """Apply a 4x4 gate to neighbouring sites (i, i+1), then SVD and truncate."""
        self.move_center(i)
        A, B = self.tensors[i], self.tensors[i + 1]
        theta = np.einsum("asb,btc->astc", A, B)                 # (l, 2, 2, r)
        U4 = U.reshape(2, 2, 2, 2)                                # (s', t', s, t)
        theta = np.einsum("uvst,astc->auvc", U4, theta)
        l, _, _, r = theta.shape
        X, S, Vh = np.linalg.svd(theta.reshape(l * 2, 2 * r), full_matrices=False)

        keep = int(np.sum(S > self.cutoff))
        if self.chi_max is not None:
            keep = min(keep, self.chi_max)
        keep = max(keep, 1)
        total = np.sum(S**2)
        self.discarded.append(float(np.sum(S[keep:] ** 2) / total))

        X, S, Vh = X[:, :keep], S[:keep], Vh[:keep, :]
        S = S / np.linalg.norm(S)                                 # renormalize
        self.tensors[i] = X.reshape(l, 2, keep)
        self.tensors[i + 1] = (np.diag(S) @ Vh).reshape(keep, 2, r)
        self.center = i + 1

    # -------------------------------------------------------------- analysis
    def bond_dims(self) -> list[int]:
        return [A.shape[2] for A in self.tensors[:-1]]

    def schmidt_values(self, bond: int) -> np.ndarray:
        """Schmidt coefficients across the cut between sites bond and bond+1."""
        self.move_center(bond)
        l, d, r = self.tensors[bond].shape
        return np.linalg.svd(self.tensors[bond].reshape(l * d, r), compute_uv=False)

    def entropy(self, bond: int) -> float:
        p = self.schmidt_values(bond) ** 2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    def expect_1q(self, O: np.ndarray, i: int) -> float:
        """<O_i>, computed locally at the orthogonality center."""
        self.move_center(i)
        A = self.tensors[i]
        return float(np.real(np.einsum("asb,st,atb->", A.conj(), O, A)))

    def estimated_fidelity(self) -> float:
        """Product of (1 - eps_k) over truncations; tracks the true fidelity."""
        return float(np.prod([1.0 - e for e in self.discarded]))

    def to_statevector(self) -> np.ndarray:
        """Contract to a dense vector (only for small n, for validation)."""
        psi = self.tensors[0]
        for A in self.tensors[1:]:
            psi = np.einsum("...a,asb->...sb", psi, A)
        return psi.reshape(-1)


# ---------------------------------------------------------------------------
# Exact state-vector reference (same gate conventions; qubit 0 = leftmost)
# ---------------------------------------------------------------------------
class StateVector:
    def __init__(self, n: int):
        self.n = n
        self.psi = np.zeros((2,) * n, dtype=complex)
        self.psi[(0,) * n] = 1.0

    def apply_1q(self, U, i):
        self.psi = np.moveaxis(np.tensordot(U, self.psi, axes=([1], [i])), 0, i)

    def apply_2q(self, U, i):
        U4 = U.reshape(2, 2, 2, 2)
        out = np.tensordot(U4, self.psi, axes=([2, 3], [i, i + 1]))
        self.psi = np.moveaxis(out, [0, 1], [i, i + 1])

    def vector(self):
        return self.psi.reshape(-1)


# ---------------------------------------------------------------------------
# Trotterized transverse-field Ising chain:  H = J sum Z_i Z_{i+1} + h sum X_i
# ---------------------------------------------------------------------------
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(theta):
    """exp(-i theta X / 2)."""
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * X


def rzz(theta):
    """exp(-i theta Z Z / 2)."""
    zz = np.diag([1, -1, -1, 1]).astype(complex)
    return np.diag(np.exp(-1j * theta / 2 * np.diag(zz)))


def trotter_step(sim, n, J, h, dt):
    """One first-order step: ZZ layer on even then odd bonds, then X layer."""
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            sim.apply_2q(rzz(2 * J * dt), i)
    for i in range(n):
        sim.apply_1q(rx(2 * h * dt), i)


def prepare_initial(sim, n):
    """|+>^n start (an equal superposition) so the dynamics builds entanglement."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for i in range(n):
        sim.apply_1q(H, i)


Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Projected ensembles / geometric quantum states from an MPS by sampling
# ---------------------------------------------------------------------------
def sample_projected_ensemble(mps: MPS, subsystem, M: int, seed=None):
    """
    Sample the projected ensemble of ``subsystem`` (kept, unmeasured sites) when
    every other site is measured in the computational basis.

    Method (sequential "perfect" sampling)
    ---------------------------------------
    1. Move the orthogonality center to site 0, so sites 1..n-1 are
       right-orthonormal:  sum_s A^s A^{s dagger} = I.
    2. Sweep left to right, carrying a boundary tensor L of shape
       (M, 2^k_so_far, chi). Kept sites add an open physical index to L.
       For a measured site j, the conditional Born probabilities are
           p(z_j | earlier outcomes) = ||L A_j^{z_j}||^2 / ||L||^2,
       because everything to the right of j contracts to the identity.
    3. After the last site, L (normalized) is the conditional state of the
       subsystem for the sampled outcome z.

    Cost: O(M * n * 2^k * chi^2). No state vector and no enumeration of outcomes.

    Parameters
    ----------
    mps : MPS
        State to sample. Its gauge (orthogonality center) is changed, not the state.
    subsystem : int or sequence of int
        Site(s) kept as the subsystem A. Ordering of A's basis follows increasing
        site index (first kept site is the most significant bit).
    M : int
        Number of samples.
    seed : int or None
        Random seed.

    Returns
    -------
    chi : ndarray, shape (M, 2^k), complex
        Normalized conditional states |psi_z> of the subsystem (one row per sample).
    p : ndarray, shape (M,)
        Exact Born probability p_z of each sampled outcome.
    z_index : ndarray, shape (M,), int
        Outcome index in the same convention as ``gqs.states.Reduced_state_single_site``
        (measured sites in increasing order, first site = most significant bit).
    """
    if np.isscalar(subsystem):
        subsystem = [int(subsystem)]
    keep = set(int(s) for s in subsystem)
    n = mps.n
    if not keep or not keep.issubset(range(n)):
        raise ValueError("subsystem must be a non-empty set of valid site indices")
    rng = np.random.default_rng(seed)

    mps.move_center(0)
    L = np.ones((M, 1, 1), dtype=complex)        # (samples, subsystem rows, bond)
    p = np.ones(M)
    z_index = np.zeros(M, dtype=np.int64)

    for j, T in enumerate(mps.tensors):          # T has shape (l, 2, r)
        if j in keep:
            Mm, R, _ = L.shape
            L = np.einsum("mra,asb->mrsb", L, T).reshape(Mm, R * 2, T.shape[2])
            continue
        L0 = L @ T[:, 0, :]
        L1 = L @ T[:, 1, :]
        n0 = np.sum(np.abs(L0) ** 2, axis=(1, 2))
        n1 = np.sum(np.abs(L1) ** 2, axis=(1, 2))
        tot = n0 + n1
        z = rng.random(M) * tot >= n0            # True -> outcome 1
        nz = np.where(z, n1, n0)
        p *= nz / tot
        L = np.where(z[:, None, None], L1, L0) / np.sqrt(nz)[:, None, None]
        z_index = 2 * z_index + z

    chi = L[:, :, 0]
    chi /= np.linalg.norm(chi, axis=1, keepdims=True)
    return chi, p, z_index


def bloch_vectors(chi: np.ndarray) -> np.ndarray:
    """Bloch vectors (x, y, z) of single-qubit states, one row per state."""
    a, b = chi[:, 0], chi[:, 1]
    return np.column_stack([2 * np.real(np.conj(a) * b),
                            2 * np.imag(np.conj(a) * b),
                            np.abs(a) ** 2 - np.abs(b) ** 2])


def ensemble_moments(bloch: np.ndarray, weights=None):
    """First moment (mean Bloch vector = RDM) and second moment E[r r^T] of an ensemble."""
    w = np.full(len(bloch), 1.0 / len(bloch)) if weights is None else np.asarray(weights) / np.sum(weights)
    m1 = w @ bloch
    m2 = np.einsum("m,mi,mj->ij", w, bloch, bloch)
    return m1, m2


def fubini_study_matrix(chi_1: np.ndarray, chi_2: np.ndarray) -> np.ndarray:
    """Pairwise Fubini-Study distances arccos|<a|b>| (vectorized Dist_ij)."""
    ov = np.abs(chi_1.conj() @ chi_2.T)
    return np.arccos(np.clip(ov, 0.0, 1.0))


def empirical_ensemble(chi: np.ndarray, z_index: np.ndarray):
    """Merge repeated outcomes: returns unique conditional states and their frequencies."""
    uniq, first, counts = np.unique(z_index, return_index=True, return_counts=True)
    return chi[first], counts / counts.sum(), uniq
