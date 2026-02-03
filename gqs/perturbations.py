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
from .states import Initial_state

def wrap_phi(phi):
    """Wrap to (-pi, pi]."""
    return (phi + np.pi) % (2*np.pi) - np.pi

def theta_phi_to_vec(theta, phi):
    return np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ], dtype=float)

def vec_to_theta_phi(r):
    x, y, z = r
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    phi = wrap_phi(phi)
    return theta, phi

def rotate_vec(r, axis, angle):
    # Rodrigues rotation formula
    axis = axis / np.linalg.norm(axis)
    return (r*np.cos(angle)
            + np.cross(axis, r)*np.sin(angle)
            + axis*np.dot(axis, r)*(1 - np.cos(angle)))

def perturb_theta_phi_isotropic(nqubit, theta0, phi0, angle_sigma=1e-2, rng=None):
    """
    Perturb (theta, phi) by applying a small random rotation on the Bloch sphere.
    angle_sigma sets typical rotation angle in radians.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = theta_phi_to_vec(theta0, phi0)

    # random axis (uniform on sphere)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)

    # small rotation angle
    angle = rng.normal(scale=angle_sigma)

    r_p = rotate_vec(r, axis, angle)
    theta_p, phi_p = vec_to_theta_phi(r_p)

    Psi_pert = Initial_state(nqubit, theta_p, phi_p)
    return Psi_pert, theta_p, phi_p
