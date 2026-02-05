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
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from .distances import Psi_Dist, _mask_chi_lambda

def bloch_from_chi(chi_S):
    """
    Convert conditional single-qubit pure states chi_S into Bloch coordinates.

    Parameters
    ----------
    chi_S : array-like, shape (N, 2), complex
        Each row is a normalized qubit state |chi_a> = (alpha, beta).

    Returns
    -------
    sx, sy, sz : np.ndarray, shape (N,)
        Bloch-sphere coordinates.
    """
    chi_S = np.asarray(chi_S)

    alpha = chi_S[:, 0]
    beta  = chi_S[:, 1]

    sx = 2.0 * np.real(np.conj(alpha) * beta)
    sy = 2.0 * np.imag(np.conj(alpha) * beta)
    sz = np.abs(alpha)**2 - np.abs(beta)**2

    return sx, sy, sz

def aggregate_bloch(sx, sy, sz, lam, decimals=10):
    """
    Aggregate points with (approximately) identical Bloch coordinates by summing lambda.

    decimals controls tolerance via rounding (higher = stricter).
    """
    lam = np.asarray(lam).reshape(-1)
    pts = np.column_stack([sx, sy, sz])

    # Bucket points by rounded coordinates
    key = np.round(pts, decimals=decimals)

    # Find unique buckets and sum lambdas
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    lam_sum = np.bincount(inv, weights=lam)

    sx_u, sy_u, sz_u = uniq[:, 0], uniq[:, 1], uniq[:, 2]
    return sx_u, sy_u, sz_u, lam_sum

def GQS_Bloch_Sphere_chi(chi_S, qz, fname='none'):
    """
    Plot conditional single-qubit states |chi_a> on the Bloch sphere,
    colored by their associated probabilities qz = lambda_E[a].

    Parameters
    ----------
    chi_S : array-like, shape (N, 2), complex
        Conditional pure states |chi_a>.
    qz : array-like, shape (N,)
        Associated probabilities lambda_E[a].
    fname : str
        Filename prefix for saving the figure ('none' to disable saving).
    """
    chi_S = np.asarray(chi_S)
    qz = np.asarray(qz)

    if chi_S.shape[0] != qz.size:
        raise ValueError("chi_S and qz must have the same length")

    chi_S, qz = _mask_chi_lambda(chi_S, qz,renormalize=True)

    # Bloch coordinates
    sx, sy, sz = bloch_from_chi(chi_S)

    # Aggregate identical points
    sx, sy, sz, qz = aggregate_bloch(sx, sy, sz, qz)

    # Create Bloch sphere
    r = 1.0
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='whitesmoke', alpha=0.3, linewidth=0)

    ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=16)
    ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=16)

    im = ax.scatter(sx, sy, sz,
                    s=20,
                    c=qz,
                    vmin =0,
                    vmax =1,
                    cmap='viridis',
                    alpha=1.0)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.12)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r'$\lambda_E$', rotation=0,
                       labelpad=25, fontsize=16)

    # Axis labels and formatting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax.set_xlabel(r'$x$', labelpad=1, fontsize=16)
    ax.set_ylabel(r'$y$', labelpad=1, fontsize=16)
    ax.set_zlabel(r'$z$', labelpad=1, fontsize=16)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    plt.tick_params(labelsize=12)

    if fname != 'none':
        plt.savefig(fname + '.png', format='png', bbox_inches='tight')

    plt.show()
    return

def GQS_Bloch_Density(chi_S, qz, fname='none', n_phi=20, n_theta=20,
                      normalize=True, log_scale=False):
    """
    Plot a probability density heatmap on the Bloch sphere in (phi, theta) coordinates.

    Parameters
    ----------
    chi_S : array-like, shape (N,2), complex
        Conditional single-qubit states |chi_a>.
    qz : array-like, shape (N,)
        Weights/probabilities associated with each |chi_a> (e.g., lambda_E[a]).
    fname : str
        If not 'none', saves as fname + '.png'.
    n_phi, n_theta : int
        Number of bins in phi (0..2pi) and theta (0..pi).
    normalize : bool
        If True, normalize qz so total mass is 1 before binning.
    log_scale : bool
        If True, plot log10(density + tiny) to reveal low-mass regions.

    Notes
    -----
    - This bins the sphere in (phi, theta). Areas near the poles represent smaller surface area
      per (phi, theta) bin; for a true surface-density per unit area, you can optionally divide
      by sin(theta) (see comment in code).
    """
    chi_S = np.asarray(chi_S)
    qz = np.asarray(qz).reshape(-1)

    if chi_S.shape[0] != qz.size:
        raise ValueError("chi_S and qz must have the same length")

    if normalize:
        s = qz.sum()
        if s > 0:
            qz = qz / s

    # Bloch coordinates
    sx, sy, sz = bloch_from_chi(chi_S)

    # Convert to spherical angles
    # theta in [0, pi], phi in [0, 2pi)
    sz_clip = np.clip(sz, -1.0, 1.0)
    theta = np.arccos(sz_clip)
    phi = np.mod(np.arctan2(sy, sx), 2.0*np.pi)

    # Bin edges
    phi_edges = np.linspace(0.0, 2.0*np.pi, n_phi + 1)
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)

    # Weighted 2D histogram: H[tbin, pbin]
    H, _, _ = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges], weights=qz)

    # Optional: convert "mass per bin" into an approximate surface density per unit area.
    # Each bin's surface area ~ sin(theta_center) dtheta dphi, so density ~ H / sin(theta_center).
    # Uncomment to get closer to density-per-area:
    # theta_centers = 0.5*(theta_edges[:-1] + theta_edges[1:])
    # area_factor = np.sin(theta_centers)[:, None]
    # H = np.divide(H, area_factor, out=np.zeros_like(H), where=area_factor > 1e-12)

    plot_data = H
    if log_scale:
        plot_data = np.log10(H + 1e-16)

    # Plot
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    # Use pcolormesh so edges match bins
    Theta, Phi = np.meshgrid(theta_edges, phi_edges)
    im = ax.pcolormesh(Theta, Phi, plot_data,vmin=0,vmax=1, shading='auto')

    ax.set_ylabel(r'$\phi$', fontsize=18)
    ax.set_xlabel(r'$\theta$', fontsize=18)
    ax.set_title(r'Bloch-sphere density (weighted by $\lambda_E$)', fontsize=12)

    # Helpful ticks
    ax.set_ylim(0, 2*np.pi)
    ax.set_xlim(0, np.pi)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'Binned mass' + (r' (log$_{10}$)' if log_scale else ''), fontsize=14)

    fig.tight_layout()

    if fname != 'none':
        plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')

    plt.show()

def GQS_Bloch_Sphere_two_chi(chi_S1, qz1, chi_S2, qz2, fname='none',
                            marker1='o', marker2='^',
                            s1=20, s2=35):
    """
    Plot TWO sets of conditional single-qubit states on the SAME Bloch sphere.

    Behaves like GQS_Bloch_Sphere_chi:
      - same colorbar (0..1) for both sets
      - both colored by their probabilities
      - second set uses a different marker

    Parameters
    ----------
    chi_S1 : array-like, shape (N1, 2), complex
        First set of conditional pure states |chi_a>.
    qz1 : array-like, shape (N1,)
        Probabilities for the first set.
    chi_S2 : array-like, shape (N2, 2), complex
        Second set of conditional pure states |chi_a>.
    qz2 : array-like, shape (N2,)
        Probabilities for the second set.
    fname : str
        Filename prefix for saving the figure ('none' to disable saving).
    marker1, marker2 : str
        Markers for the first and second set.
    s1, s2 : float
        Marker sizes for the first and second set.
    """
    chi_S1 = np.asarray(chi_S1)
    qz1 = np.asarray(qz1)
    chi_S2 = np.asarray(chi_S2)
    qz2 = np.asarray(qz2)

    if chi_S1.shape[0] != qz1.size:
        raise ValueError("chi_S1 and qz1 must have the same length")
    if chi_S2.shape[0] != qz2.size:
        raise ValueError("chi_S2 and qz2 must have the same length")

    # Apply same masking/renormalization behavior as your original function
    chi_S1, qz1 = _mask_chi_lambda(chi_S1, qz1, renormalize=True)
    chi_S2, qz2 = _mask_chi_lambda(chi_S2, qz2, renormalize=True)

    # Bloch coordinates
    sx1, sy1, sz1 = bloch_from_chi(chi_S1)
    sx2, sy2, sz2 = bloch_from_chi(chi_S2)

    # Aggregate identical points (within each set, consistent with your pipeline)
    sx1, sy1, sz1, qz1 = aggregate_bloch(sx1, sy1, sz1, qz1)
    sx2, sy2, sz2, qz2 = aggregate_bloch(sx2, sy2, sz2, qz2)

    # Create Bloch sphere
    r = 1.0
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='whitesmoke', alpha=0.3, linewidth=0)

    ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=16)
    ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=16)

    # --- Key behavior: same colormap + same vmin/vmax (0..1) for both ---
    im1 = ax.scatter(sx1, sy1, sz1,
                     s=s1,
                     c=qz1,
                     vmin=0, vmax=1,
                     cmap='viridis',
                     alpha=1.0,
                     marker=marker1)

    ax.scatter(sx2, sy2, sz2,
               s=s2,
               c=qz2,
               vmin=0, vmax=1,
               cmap='viridis',
               alpha=1.0,
               marker=marker2)

    fig.tight_layout()

    # Colorbar: use im1, but applies to same (cmap, vmin/vmax) for both sets
    cbar = fig.colorbar(im1, ax=ax, shrink=0.5, pad=0.12)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r'$\lambda_E$', rotation=0,
                       labelpad=25, fontsize=16)

    # Axis labels and formatting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax.set_xlabel(r'$x$', labelpad=1, fontsize=16)
    ax.set_ylabel(r'$y$', labelpad=1, fontsize=16)
    ax.set_zlabel(r'$z$', labelpad=1, fontsize=16)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    plt.tick_params(labelsize=12)

    if fname != 'none':
        plt.savefig(fname + '.png', format='png', bbox_inches='tight')

    plt.show()
    return
