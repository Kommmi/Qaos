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
import pandas as pd
from numpy import linalg as LA
from gqs.distances import Psi_Dist, _mask_chi_lambda

def Initial_state(nqubit, theta0, phi0):
    """
    Product state of n qubits, each initialized to
    cos(theta/2)|0> + e^{-i phi} sin(theta/2)|1>.
    """
    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)

    psi_single = (
        np.cos(theta0 / 2.0) * ket0
        + np.sin(theta0 / 2.0) * np.exp(1j * phi0) * ket1
    )

    Psi = psi_single
    for _ in range(nqubit - 1):
        Psi = np.kron(Psi, psi_single)

    return Psi

def Reduced_state_single_site(d_hilbert, n_chain, system_site, Psi_SE, lambda_E=None, eps=1e-12):
    """
    Compute conditional pure system states |chi_a> for a user-chosen single site in a chain and a
    associated.

    For each environment basis outcome a:
        |chi_a> = (1/sqrt(lambda_E[a])) * sum_k psi_{k,a} |k>
    where lambda_E[a] = sum_k |psi_{k,a}|^2.

    Parameters
    ----------
    d_hilbert : int
        Local Hilbert space dimension (2 for qubits).
    n_chain : int
        Number of subsystems.
    system_site : int
        Index (0-based) of the chosen system site.
    Psi_SE : array-like, complex
        Pure state vector, length d_hilbert**n_chain (or shape (1, d_hilbert**n_chain)).
    lambda_E : array-like or None
        Optional precomputed lambda_E of length d_hilbert**(n_chain-1).
        If None, it is computed internally.
    eps : float
        Threshold to avoid division by ~0.

    Returns
    -------
    chi_S : np.ndarray, shape (d_hilbert**(n_chain-1), d_hilbert), complex
        Row a is the conditional pure state |chi_a> written in the system basis.
        Rows for which lambda_E[a] <= eps are left as all zeros.
    lambda_E : np.ndarray, shape (d_hilbert**(n_chain-1),), float
        The environment probabilities used for normalization.
    """
    d = int(d_hilbert)
    n = int(n_chain)

    if not (0 <= system_site < n):
        raise ValueError(f"system_site must be in [0, {n-1}]")

    Psi = np.asarray(Psi_SE).reshape(-1)
    expected = d**n
    if Psi.size != expected:
        raise ValueError(f"Psi_SE must have length {expected}, got {Psi.size}")

    # (1) Tensor view: one axis per subsystem
    Psi_tensor = Psi.reshape((d,) * n)

    # (2) Put the chosen site first: (system, env...)
    Psi_perm = np.moveaxis(Psi_tensor, system_site, 0)

    # (3) Flatten env indices: A has shape (d, d^(n-1))
    A = Psi_perm.reshape(d, -1)

    # (4) lambda_E[a] = sum_k |A_{k,a}|^2
    if lambda_E is None:
        lambda_E = np.sum(np.abs(A)**2, axis=0)
    else:
        lambda_E = np.asarray(lambda_E).reshape(-1)
        if lambda_E.size != d**(n-1):
            raise ValueError(f"lambda_E must have length {d**(n-1)}, got {lambda_E.size}")

    # (5) chi_a = A[:,a] / sqrt(lambda_E[a])  (column normalization)
    chi = np.zeros_like(A, dtype=complex)  # (d, d^(n-1))
    mask = lambda_E > eps
    chi[:, mask] = A[:, mask] / np.sqrt(lambda_E[mask])

    # Return as (env_state_index, system_dim) like your original chi_S[a]
    chi_S = chi.T  # shape (d^(n-1), d)

    return chi_S, lambda_E

def rho_single_spin(d_hilbert, n_chain, system_site, Psi_SE):
    """
    Single-spin reduced density matrix rho_s for a user-chosen subsystem in an n-spin pure state.

    Computes: rho_s = Tr_env( |Psi><Psi| )

    Parameters
    ----------
    d_hilbert : int
        Local Hilbert space dimension (2 for qubits).
    n_chain : int
        Number of subsystems in the chain.
    system_site : int
        Which subsystem is the "system" (0-based index: 0,...,n_chain-1).
    Psi_SE : array-like, complex
        Pure state vector of length d_hilbert**n_chain.

    Returns
    -------
    rho_s : np.ndarray, shape (d_hilbert, d_hilbert), complex
        Reduced density matrix of the chosen subsystem.
    """
    d = int(d_hilbert)
    n = int(n_chain)

    if not (0 <= system_site < n):
        raise ValueError(f"system_site must be in [0, {n-1}]")

    Psi = np.asarray(Psi_SE).reshape(-1)
    expected = d**n
    if Psi.size != expected:
        raise ValueError(f"Psi_SE must have length {expected}, got {Psi.size}")

    # Reshape to n-index tensor and move system axis to front
    Psi_tensor = Psi.reshape((d,) * n)
    Psi_perm = np.moveaxis(Psi_tensor, system_site, 0)

    # Flatten env indices -> matrix A with shape (d, d^(n-1))
    A = Psi_perm.reshape(d, -1)

    # Partial trace over environment: rho_s = A A^\dagger
    rho_s = A @ A.conj().T

    # Numerical hygiene: enforce Hermiticity (optional)
    rho_s = 0.5 * (rho_s + rho_s.conj().T)

    return rho_s

def random_wavefunction(num_qubits, seed=None):
    """
    Return a normalized random wavefunction for `num_qubits`.

    The state lives in a Hilbert space of dimension 2**num_qubits:
        |Psi_SE> = sum_i psi_i |i>

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in S + E.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    psi : np.ndarray
        Complex normalized wavefunction of shape (2**num_qubits,).
    """
    rng = np.random.default_rng(seed)

    dim = 2**num_qubits

    # Random complex amplitudes
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)

    # Normalize
    psi = psi / np.linalg.norm(psi)

    return psi

def print_wavefunction(psi, tol=1e-12, ncols=4):
    """
    Print wavefunction in computational basis using multiple columns,
    with '+' signs between terms.
    """
    dim = len(psi)
    num_qubits = int(np.log2(dim))

    entries = []

    for i, amp in enumerate(psi):
        if abs(amp) > tol:
            basis = format(i, f"0{num_qubits}b")
            entries.append(f"({amp:.4f}) |{basis}>")

    for i in range(0, len(entries), ncols):
        row = entries[i:i+ncols]

        # Add + between entries in the same row
        line = " + ".join(f"{entry:<28}" for entry in row)

        # Add + at the end of the row if more entries remain
        if i + ncols < len(entries):
            line += " +"

        print(line)

def print_density_matrix(rho, basis_labels=None, precision=4, title=r"rho_S"):
    """
    Print a density matrix nicely with optional basis labels.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    basis_labels : list of str or None
        Basis labels, e.g. ["|0>", "|1>"].
    precision : int
        Number of digits to print.
    title : str
        Name/title for the matrix.
    """
    rho = np.asarray(rho)

    d = rho.shape[0]

    if basis_labels is None:
        basis_labels = [f"|{i}>" for i in range(d)]

    print(f"{title} =")
    print()

    # Header row
    header = " " * 10
    for label in basis_labels:
        header += f"{label:^18}"
    print(header)

    # Matrix rows
    for i in range(d):
        row = f"{basis_labels[i]:<10}"
        for j in range(d):
            row += f"{format_complex(rho[i, j], precision):^18}"
        print(row)

    print()
    print(f"Trace = {np.trace(rho).real:.{precision}f}")
    print(f"Hermitian check = {np.allclose(rho, rho.conj().T)}")


def format_complex(z, precision=4, chop=1e-12):
    """
    Nicely format a complex number.
    """
    z = complex(z)

    real = 0.0 if abs(z.real) < chop else z.real
    imag = 0.0 if abs(z.imag) < chop else z.imag

    if imag == 0:
        return f"{real:.{precision}f}"
    elif real == 0:
        return f"{imag:.{precision}f}j"
    elif real == 0 and imag == 0:
        return f"0.{precision}f"
    elif imag > 0:
        return f"{real:.{precision}f}+{imag:.{precision}f}j"
    else:
        return f"{real:.{precision}f}{imag:.{precision}f}j"


def gqs_single_site_table(
    chi_S,
    lambda_E,
    d_hilbert=2,
    n_chain=None,
    precision=4,
    tol=1e-12,
):
    """
    Return the single-site GQS as a pandas DataFrame.

    Columns:
        1. j
        2. environment state
        3. |chi_j^S>
        4. lambda_E[j]
    """
    chi_S = np.asarray(chi_S)
    lambda_E = np.asarray(lambda_E)

    d_E, d_S = chi_S.shape

    if n_chain is not None:
        n_env = n_chain - 1
    else:
        n_env = int(np.round(np.log(d_E) / np.log(d_hilbert)))

    rows = []

    for j in range(d_E):
        lam = lambda_E[j]

        if lam <= tol:
            continue

        # Environment basis label
        if d_hilbert == 2:
            env_label = f"|{format(j, f'0{n_env}b')}>"
        else:
            env_label = f"|e_{j}>"

        # Conditional subsystem state
        terms = []
        for k in range(d_S):
            amp = chi_S[j, k]

            if abs(amp) > tol:
                amp_str = format_complex(amp, precision=precision)
                terms.append(f"({amp_str}) |{k}>"
)

        chi_str = " + ".join(terms)

        rows.append(
            {
                "j": j,
                "Environment state": env_label,
                r"|χ_j^S⟩": chi_str,
                r"λ_E[j]": np.round(lam, precision),
            }
        )

    return pd.DataFrame(rows)


def print_gqs_single_site(
    chi_S,
    lambda_E,
    d_hilbert=2,
    n_chain=None,
    system_site=None,
    precision=4,
    tol=1e-12,
    title="Geometric Quantum State Q^S"
):
    """
    Print the environment-conditioned GQS ensemble nicely.

    Parameters
    ----------
    chi_S : np.ndarray
        Conditional subsystem states, shape (d_E, d_hilbert).
        Row j is |chi_j^S>.
    lambda_E : np.ndarray
        Environment probabilities, shape (d_E,).
    d_hilbert : int
        Local Hilbert space dimension.
    n_chain : int or None
        Total number of sites. Used only for labeling environment states.
    system_site : int or None
        Chosen subsystem site. Used only for printing context.
    precision : int
        Number of digits to print.
    tol : float
        Threshold for skipping zero-probability environment outcomes.
    title : str
        Title for printed output.
    """
    chi_S = np.asarray(chi_S)
    lambda_E = np.asarray(lambda_E)

    d_E, d_S = chi_S.shape

    if n_chain is not None:
        n_env = n_chain - 1
    else:
        n_env = int(np.log(d_E) / np.log(d_hilbert))

    print(title)
    print("=" * len(title))

    if system_site is not None:
        print(f"Subsystem site: {system_site}")
    print(f"Subsystem dimension d_S = {d_S}")
    print(f"Number of environment outcomes d_E = {d_E}")
    print()

    df_gqs = gqs_single_site_table(
    chi_S,
    lambda_E,
    d_hilbert=2,
    n_chain=3,
    precision=3
    )

    styles = [
        {'selector': 'td, th', 'props': [('border-right', '1px solid #ddd')]}
    ]
    styled_df = df_gqs.style.set_table_styles(styles)
    display(styled_df)
    return
