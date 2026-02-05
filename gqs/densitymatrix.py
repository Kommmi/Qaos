import numpy as np

def bloch_vector_from_density(rho):
    """Return Bloch vector [r_x, r_y, r_z] for a single-qubit density matrix."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must be shape (2,2) for a single qubit")

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    rx = np.trace(rho @ sx).real
    ry = np.trace(rho @ sy).real
    rz = np.trace(rho @ sz).real
    return np.array([rx, ry, rz])

