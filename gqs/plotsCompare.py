
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker

from .distances import _mask_chi_lambda
from .gqs import bloch_from_chi, aggregate_bloch
from .states import Reduced_state_single_site, rho_single_spin
from .distances import Quantum_EMD, bures_distance

def Plot_Avg_Separation(data_dist,N_kicks):
  D_global = data_dist['ln_avg_dist_G']
  D_local = data_dist['ln_avg_dist_L']
  D_local_rho = data_dist['ln_avg_dist_R']

  plt.figure(figsize=(12,3))
  ml1, _, _ = plt.stem(range(N_kicks), D_global,linefmt='C0-',markerfmt='C0o',basefmt=' ',label='Global State Distance')
  ml1.set_markersize(2)
  ml2, _, _ = plt.stem(range(N_kicks), D_local,linefmt='C1-',markerfmt='C1^',basefmt=' ',label='Quantum EMD')
  ml2.set_markersize(2)
  ml3, _, _ = plt.stem(range(N_kicks), D_local_rho,linefmt='C2-',markerfmt='C2s',basefmt=' ',label='Bures Distance')
  ml3.set_markersize(2)
  plt.xlabel('n - Number of Floquet Kicks',fontsize=14)
  plt.ylabel(r'$\langle log(\frac{d(n)}{d(0)})\rangle$',fontsize=14)
  plt.title('Comparison of Average Global and Local Distances over Time',fontsize=16)
  plt.legend(fontsize=12)
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()

def _GQS_Bloch_Sphere_two_chi_on_ax(
    ax, chi_S1, qz1, chi_S2, qz2,
    marker1='o', marker2='^',
    s1=20, s2=35,
    sphere_alpha=0.3
):
    """
    Subplot-friendly version of GQS_Bloch_Sphere_two_chi:
    draws onto an existing 3D axis and returns a mappable for colorbar.
    """
    chi_S1 = np.asarray(chi_S1)
    qz1 = np.asarray(qz1)
    chi_S2 = np.asarray(chi_S2)
    qz2 = np.asarray(qz2)

    if chi_S1.shape[0] != qz1.size:
        raise ValueError("chi_S1 and qz1 must have the same length")
    if chi_S2.shape[0] != qz2.size:
        raise ValueError("chi_S2 and qz2 must have the same length")

    # Same masking/renormalization behavior
    chi_S1, qz1 = _mask_chi_lambda(chi_S1, qz1, renormalize=True)
    chi_S2, qz2 = _mask_chi_lambda(chi_S2, qz2, renormalize=True)

    # Bloch coordinates
    sx1, sy1, sz1 = bloch_from_chi(chi_S1)
    sx2, sy2, sz2 = bloch_from_chi(chi_S2)

    # Aggregate identical points within each set
    sx1, sy1, sz1, qz1 = aggregate_bloch(sx1, sy1, sz1, qz1)
    sx2, sy2, sz2, qz2 = aggregate_bloch(sx2, sy2, sz2, qz2)

    # Bloch sphere surface
    r = 1.0
    phi, theta = np.mgrid[0.0:np.pi:80j, 0.0:2.0*np.pi:80j]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='whitesmoke', alpha=sphere_alpha, linewidth=0)

    ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=12)
    ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=12)

    # SAME colormap + SAME vmin/vmax for both
    im1 = ax.scatter(sx1, sy1, sz1,
                     s=s1, c=qz1,
                     vmin=0, vmax=1,
                     cmap='viridis',
                     alpha=1.0, marker=marker1)

    ax.scatter(sx2, sy2, sz2,
               s=s2, c=qz2,
               vmin=0, vmax=1,
               cmap='viridis',
               alpha=1.0, marker=marker2)

    # Formatting
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])

    ax.set_xlabel(r'$x$', labelpad=0, fontsize=12)
    ax.set_ylabel(r'$y$', labelpad=0, fontsize=12)
    ax.set_zlabel(r'$z$', labelpad=0, fontsize=12)

    return im1

def _GQS_Bloch_Sphere_two_rho_on_ax(
    ax, rho1, rho2=None,
    color1='C0', color2='C1',
    marker1='o', marker2='^',
    s1=20, s2=35,
    sphere_alpha=0.3,
    draw_sphere=True,
    annotate_kets=True,
):
    """
    Plot one or two (possibly time-dependent) single-qubit density matrices
    on a Bloch sphere without colorbars.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis to draw on (projection='3d').
    rho1 : (2,2) or (T,2,2)
        First density matrix or time series.
    rho2 : (2,2) or (T,2,2) or None
        Second density matrix or time series.
    color1, color2 : str
        Solid colors for each trajectory.
    marker1, marker2 : str
        Marker styles.
    s1, s2 : float
        Marker sizes.
    sphere_alpha : float
        Transparency of Bloch sphere surface.
    draw_sphere : bool
        Whether to draw Bloch sphere.
    annotate_kets : bool
        Whether to label |0>, |1>.
    """

    def _as_rho_timeseries(rho):
        rho = np.asarray(rho, dtype=complex)
        if rho.shape == (2, 2):
            return rho[None, :, :]
        if rho.ndim == 3 and rho.shape[1:] == (2, 2):
            return rho
        raise ValueError("rho must be shape (2,2) or (T,2,2)")

    def _bloch_from_rho_series(rhos):
        T = rhos.shape[0]
        out = np.empty((T, 3))
        for t in range(T):
            sx = np.array([[0, 1], [1, 0]], dtype=complex)
            sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
            sz = np.array([[1, 0], [0, -1]], dtype=complex)
            out[t, 0] = np.trace(rhos[t] @ sx).real
            out[t, 1] = np.trace(rhos[t] @ sy).real
            out[t, 2] = np.trace(rhos[t] @ sz).real
        return out[:, 0], out[:, 1], out[:, 2]

    # --- Bloch coordinates ---
    rhos1 = _as_rho_timeseries(rho1)
    x1, y1, z1 = _bloch_from_rho_series(rhos1)

    if rho2 is not None:
        rhos2 = _as_rho_timeseries(rho2)
        x2, y2, z2 = _bloch_from_rho_series(rhos2)

    # --- Bloch sphere ---
    if draw_sphere:
        phi, theta = np.mgrid[0.0:np.pi:80j, 0.0:2.0*np.pi:80j]
        xs = np.sin(phi) * np.cos(theta)
        ys = np.sin(phi) * np.sin(theta)
        zs = np.cos(phi)
        ax.plot_surface(xs, ys, zs,
                        color='whitesmoke',
                        alpha=sphere_alpha,
                        linewidth=0)

    if annotate_kets:
        ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=12)
        ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=12)

    # --- Scatter points ---
    ax.scatter(x1, y1, z1,
               s=s1, c=color1,
               alpha=1.0, marker=marker1)

    if rho2 is not None:
        ax.scatter(x2, y2, z2,
                   s=s2, c=color2,
                   alpha=1.0, marker=marker2)

    # --- Formatting ---
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    ax.set_xlabel(r'$x$', fontsize=12, labelpad=0)
    ax.set_ylabel(r'$y$', fontsize=12, labelpad=0)
    ax.set_zlabel(r'$z$', fontsize=12, labelpad=0)


def plot_gqs_and_rho_before_after_kick(
    U_F, Psi_0, Psi_p,
    dhilbert, nqubit, site,
    sphere_alpha=0.25,
    marker1='o', marker2='^',
    s1=20, s2=35,
    fig_scale=(18, 4),
    suptitle=None,
):
    """
    1x4 figure:
      (1) GQS before kick  + QEMD
      (2) GQS after kick   + QEMD
      (3) rho before kick + Bures
      (4) rho after kick  + Bures

    Adds a shared colorbar for GQS panels only.
    """

    def _put_below(ax, text, dy=0.06, fontsize=11):
        bbox = ax.get_position()
        x = 0.5 * (bbox.x0 + bbox.x1)
        y = bbox.y0 - dy
        ax.figure.text(x, y, text, ha='center', va='top', fontsize=fontsize)

    # -------- BEFORE kick --------
    chi_p_b, lam_p_b = Reduced_state_single_site(dhilbert, nqubit, site, Psi_p)
    chi_0_b, lam_0_b = Reduced_state_single_site(dhilbert, nqubit, site, Psi_0)
    d_qemd_before = Quantum_EMD(chi_0_b, lam_0_b, chi_p_b, lam_p_b)

    rho_p_b = rho_single_spin(dhilbert, nqubit, site, Psi_p)
    rho_0_b = rho_single_spin(dhilbert, nqubit, site, Psi_0)
    d_bures_before = bures_distance(rho_0_b, rho_p_b)

    # -------- APPLY kick --------
    Psi_0_next = U_F @ Psi_0
    Psi_p_next = U_F @ Psi_p

    # -------- AFTER kick --------
    chi_p_a, lam_p_a = Reduced_state_single_site(dhilbert, nqubit, site, Psi_p_next)
    chi_0_a, lam_0_a = Reduced_state_single_site(dhilbert, nqubit, site, Psi_0_next)
    d_qemd_after = Quantum_EMD(chi_0_a, lam_0_a, chi_p_a, lam_p_a)

    rho_p_a = rho_single_spin(dhilbert, nqubit, site, Psi_p_next)
    rho_0_a = rho_single_spin(dhilbert, nqubit, site, Psi_0_next)
    d_bures_after = bures_distance(rho_0_a, rho_p_a)

    # -------- FIGURE --------
    fig = plt.figure(figsize=fig_scale)
    if suptitle is not None:
        fig.suptitle(suptitle, y=1.02, fontsize=14)

    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')

    # -------- GQS before --------
    ax1.set_title("GQS (before kick)", fontsize=12)
    mappable = _GQS_Bloch_Sphere_two_chi_on_ax(
        ax1, chi_0_b, lam_0_b, chi_p_b, lam_p_b,
        marker1=marker1, marker2=marker2,
        s1=s1, s2=s2,
        sphere_alpha=sphere_alpha
    )
    _put_below(ax1, rf"Local QEMD = {d_qemd_before:.6g}")

    # -------- GQS after --------
    ax2.set_title("GQS (after kick)", fontsize=12)
    _GQS_Bloch_Sphere_two_chi_on_ax(
        ax2, chi_0_a, lam_0_a, chi_p_a, lam_p_a,
        marker1=marker1, marker2=marker2,
        s1=s1, s2=s2,
        sphere_alpha=sphere_alpha
    )
    _put_below(ax2, rf"Local QEMD = {d_qemd_after:.6g}")

    # -------- rho before --------
    ax3.set_title(r"$\rho$ (before kick)", fontsize=12)
    _GQS_Bloch_Sphere_two_rho_on_ax(
        ax3, rho_0_b, rho2=rho_p_b,
        marker1=marker1, marker2=marker2,
        s1=s1, s2=s2,
        sphere_alpha=sphere_alpha
    )
    _put_below(ax3, rf"Local Bures = {d_bures_before:.6g}")

    # -------- rho after --------
    ax4.set_title(r"$\rho$ (after kick)", fontsize=12)
    _GQS_Bloch_Sphere_two_rho_on_ax(
        ax4, rho_0_a, rho2=rho_p_a,
        marker1=marker1, marker2=marker2,
        s1=s1, s2=s2,
        sphere_alpha=sphere_alpha
    )
    _put_below(ax4, rf"Local Bures = {d_bures_after:.6g}")

    # -------- Shared colorbar for GQS only --------
    cbar = fig.colorbar(
        mappable,
        ax=[ax1, ax2],
        fraction=0.035,
        pad=0.04
    )
    cbar.set_label(r"Eigenvalue $\lambda_j$", fontsize=11)
    cbar.set_ticks([0, 0.5, 1.0])

    # Layout tweaks to make room for text + colorbar
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, right=0.92)

    plt.show()

    return (
        Psi_0_next, Psi_p_next,
        d_qemd_before, d_qemd_after,
        d_bures_before, d_bures_after
    )



