
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

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


def plot_density_matrix_from_gqs(
    states,
    weights=None,
    ax=None,
    title="Density matrix",
    show_bottom_text=False,
    bottom_text="Ensembles of particles\nAny length vectors",
    rhoS_label=None,
    rhoS_label_pos=(0.5, -0.1,0),
    arrow_color="#756ef9",
    sphere_alpha=0.10,
    elev=20,
    azim=-35
):
    """
    Plot the density matrix corresponding to a single-qubit GQS on a Bloch sphere.

    Parameters
    ----------
    states : ndarray, shape (N, 2) or (N, 3)
        GQS states. Two allowed formats:

        1. Pure state kets:
           states[j] = [alpha_j, beta_j], complex, normalized or unnormalized.
           Then rho = sum_j w_j |psi_j><psi_j|.

        2. Bloch vectors:
           states[j] = [x_j, y_j, z_j], real, with ||r_j|| <= 1.
           Then rho_j = (I + r_j . sigma)/2 and rho = sum_j w_j rho_j.

    weights : array-like, shape (N,), optional
        Ensemble weights. If None, uniform weights are used.

    ax : matplotlib 3D axis, optional
        If None, a new figure and axis are created.

    title : str
        Plot title.

    show_bottom_text : bool
        Whether to place text under the sphere.

    bottom_text : str
        Bottom annotation text.

    arrow_color : str
        Color of the Bloch vector arrow.

    sphere_alpha : float
        Transparency of Bloch sphere surface.

    elev, azim : float
        Viewing angles for 3D plot.

    Returns
    -------
    rho : ndarray, shape (2, 2)
        The density matrix.

    r : ndarray, shape (3,)
        The Bloch vector of rho.

    fig, ax : matplotlib figure and axis
        Figure/axis containing the plot.
    """
    states = np.asarray(states)
    N = len(states)

    if weights is None:
        weights = np.ones(N) / N
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # Build rho
    if states.ndim != 2:
        raise ValueError("states must have shape (N,2) for kets or (N,3) for Bloch vectors.")

    if states.shape[1] == 2:
        # states are kets
        rho = np.zeros((2, 2), dtype=complex)
        for w, psi in zip(weights, states):
            psi = np.asarray(psi, dtype=complex)
            norm = np.linalg.norm(psi)
            if norm == 0:
                raise ValueError("Encountered zero vector in ket input.")
            psi = psi / norm
            rho += w * np.outer(psi, np.conjugate(psi))

    elif states.shape[1] == 3:
        # states are Bloch vectors
        rho = np.zeros((2, 2), dtype=complex)
        for w, rj in zip(weights, states):
            x, y, z = rj
            rho_j = 0.5 * (I2 + x * sx + y * sy + z * sz)
            rho += w * rho_j
    else:
        raise ValueError("states must have shape (N,2) or (N,3).")

    # Bloch vector of rho
    rx = np.real(np.trace(rho @ sx))
    ry = np.real(np.trace(rho @ sy))
    rz = np.real(np.trace(rho @ sz))
    r = np.array([rx, ry, rz], dtype=float)

    # Create figure/axis if needed
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Bloch sphere surface
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs, color='lightgray', alpha=sphere_alpha, linewidth=0, shade=True)

    # Meridians / equator
    t = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), 0*t, color='gray', alpha=0.5, lw=1)
    ax.plot(np.cos(t), 0*t, np.sin(t), color='gray', alpha=0.5, lw=1)
    ax.plot(0*t, np.cos(t), np.sin(t), color='gray', alpha=0.5, lw=1)

    # Axes
    ax.quiver(0, 0, 0, 1.15, 0, 0, color='black', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, 1.15, 0, color='black', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, 1.15, color='black', arrow_length_ratio=0.08, linewidth=1)

    ax.text(1.22, 0, 0, r"$x$", fontsize=16)
    ax.text(0, 1.22, 0, r"$y$", fontsize=16)
    ax.text(0.2, 0, 1., r"$z$", fontsize=16)

    # Density-matrix Bloch vector
    ax.quiver(
        0, 0, 0,
        r[0], r[1], r[2],
        color=arrow_color,
        arrow_length_ratio=0.25,
        linewidth=4
    )

    # Formatting
    #ax.set_title(title, fontsize=18, pad=16)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    ax.text(
    0.2, -0.8, 0.8,              # slightly inside but visible
    r'$\mathcal{D}_2$',
    fontsize=18,
    ha='center',
    va='center',
    color='black'
    )


    ax.text(
   -0.1, 0, -1.7,              # slightly inside but visible
    r'Reduced Density Matrix',
    fontsize=10,
    ha='center',
    va='center',
    color='black'
    )


    if rhoS_label is not None:
        ax.text(
            rhoS_label_pos[0], rhoS_label_pos[1], rhoS_label_pos[2],
            rhoS_label,
            fontsize=16,
            ha='center',
            va='center',
            color='black'
        )

    ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=16)
    ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=16)


    if show_bottom_text:
        fig.text(0.5, 0.03, bottom_text, ha='center', va='center', fontsize=14)

    return rho, r, fig, ax

def GQS_Bloch_Sphere_chi(
    chi_S,
    qz,
    m_cmap=None,
    fname='none',
    ax=None,
    add_colorbar=True,
    show=True,
    title=None,
    inner_text=r'$CP^1$',
    inner_text_pos=(0.2, -0.8, 0.8),
    env_label=None,
    env_label_pos=(0.5, -0.1,0),
    elev=20,
    azim=-35,
    sphere_alpha=0.10,
    point_size=55,
    axis_arrow_length=1.15,
    axis_label_size=16,
    text_size=18,
    rasterized=True
):
    """
    Plot conditional single-qubit states |chi_a> on the Bloch sphere,
    colored by their associated probabilities qz = lambda_E[a],
    in a cleaner 'density-matrix-style' Bloch-sphere format.

    Parameters
    ----------
    chi_S : array-like, shape (N, 2)
        Conditional qubit states.
    qz : array-like, shape (N,)
        Associated probabilities.
    m_cmap : str or Colormap
        Colormap for probabilities.
    fname : str
        If not 'none', save figure as fname + '.png'.
    ax : matplotlib 3D axis, optional
        Existing axis to draw on.
    add_colorbar : bool
        Whether to add colorbar.
    show : bool
        Whether to show figure if axis was created here.
    title : str or None
        Optional title.
    inner_text : str or None
        Text to place inside the Bloch sphere.
    inner_text_pos : tuple
        3D position of the inner text.
    elev, azim : float
        Viewing angles.
    sphere_alpha : float
        Transparency of sphere.
    point_size : float
        Scatter marker size.
    axis_arrow_length : float
        Length of custom x,y,z arrows.
    rasterized : bool
        Rasterize scatter points for smaller saved file size.

    Returns
    -------
    ax : matplotlib 3D axis
    im : scatter artist
    """
    if m_cmap is None:
        colors_warm_red = ["#ffb3b3", "#ff4d4d", "#b30000", "#5a0000"]
        cmap_warm_red = LinearSegmentedColormap.from_list("warm_to_red", colors_warm_red)
        m_cmap = cmap_warm_red

    chi_S = np.asarray(chi_S)
    qz = np.asarray(qz)

    if chi_S.shape[0] != qz.size:
        raise ValueError("chi_S and qz must have the same length")

    # Apply your existing masking/renormalization
    chi_S, qz = _mask_chi_lambda(chi_S, qz, renormalize=True)

    # Bloch coordinates from chi
    sx, sy, sz = bloch_from_chi(chi_S)

    # Aggregate repeated points if desired
    sx, sy, sz, qz = aggregate_bloch(sx, sy, sz, qz)

    # Create figure/axis if needed
    created_ax = ax is None
    if created_ax:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # -------------------------
    # Bloch sphere surface
    # -------------------------
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        color='lightgray',
        alpha=sphere_alpha,
        linewidth=0,
        shade=True
    )

    # Great circles
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), 0*t, color='gray', alpha=0.45, lw=1)
    ax.plot(np.cos(t), 0*t, np.sin(t), color='gray', alpha=0.45, lw=1)
    ax.plot(0*t, np.cos(t), np.sin(t), color='gray', alpha=0.45, lw=1)

    # -------------------------
    # Custom axis arrows
    # -------------------------
    ax.quiver(0, 0, 0, axis_arrow_length, 0, 0,
              color='black', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, axis_arrow_length, 0,
              color='black', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, axis_arrow_length,
              color='black', arrow_length_ratio=0.08, linewidth=1)

    ax.text(1.22, 0, 0, r'$x$', fontsize=axis_label_size)
    ax.text(0, 1.22, 0, r'$y$', fontsize=axis_label_size)
    ax.text(0.2, 0, 1.0, r"$z$", fontsize=axis_label_size)

    # Optional |0>, |1> labels instead of or in addition to z-labels
    #ax.text(-0.08, 0,  1.10, r'$\left|0\right>$', fontsize=16)
    # ax.text(-0.08, 0, -1.15, r'$\left|1\right>$', fontsize=16)
    ax.text(-0.1, 0,  1.3, r'$\left|0\right>$', fontsize=16)
    ax.text(-0.1, 0, -1.3, r'$\left|1\right>$', fontsize=16)

    # -------------------------
    # GQS points
    # -------------------------
    im = ax.scatter(
        sx, sy, sz,
        s=point_size,
        c=qz,
        vmin=0,
        vmax=1,
        cmap=m_cmap,
        alpha=1.0,
        edgecolors='black',
        linewidths=0.6,
        rasterized=rasterized
    )

    # -------------------------
    # Inner text
    # -------------------------
    if inner_text is not None:
        ax.text(
            inner_text_pos[0],
            inner_text_pos[1],
            inner_text_pos[2],
            inner_text,
            fontsize=text_size,
            ha='center',
            va='center'
        )
    if env_label is not None:
        ax.text(
            env_label_pos[0],
            env_label_pos[1],
            env_label_pos[2],
            env_label,
            fontsize=14,
            ha='center',
            va='center'
        )
    ax.text(
   -0.1, 0, -1.7,              # slightly inside but visible
    r'Reduced State: GQS',
    fontsize=10,
    ha='center',
    va='center',
    color='black'
    )

    # -------------------------
    # Colorbar
    # -------------------------
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.ax.set_ylabel(r'$Q^S$', rotation=0, labelpad=18, fontsize=16)

    # -------------------------
    # Clean formatting
    # -------------------------
    if title is not None:
        ax.set_title(title, fontsize=20, pad=14)

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.view_init(elev=elev, azim=azim)

    # Remove default box/ticks/panes to match the density-matrix style
    ax.set_axis_off()

    if fname != 'none':
        fig.savefig(fname + '.png', format='png', dpi=300, bbox_inches='tight')

    if show and created_ax:
        plt.show()

    return ax, im

def plot_gqs_and_density_matrix(
    states,
    weights=None,
    m_cmap=None,
    figsize=(9, 4.5),
    fname=None,
    show=True,
    add_colorbar=True,
    gqs_title=None,
    rho_title=None,
    env_label=None,
    rhoS_label=None,
    elev=20,
    azim=-35,
    wspace=-0.05,
):
    """
    Make a 1 x 2 plot:
        left  = GQS on Bloch sphere
        right = reduced density matrix Bloch vector

    Parameters
    ----------
    states : array-like, shape (N, 2) or (N, 3)
        Conditional states. Usually shape (N, 2) for qubit kets |chi_a>.
        If shape (N, 3), they are interpreted as Bloch vectors for the density plot.

    weights : array-like, shape (N,), optional
        GQS weights qz = lambda_E[a]. If None, uniform weights are used.

    m_cmap : matplotlib colormap, optional
        Colormap for GQS weights.

    figsize : tuple
        Figure size.

    fname : str or None
        If not None, save figure to this filename.

    show : bool
        Whether to call plt.show().

    add_colorbar : bool
        Whether to add colorbar for the GQS plot.

    gqs_title, rho_title : str or None
        Optional titles.

    env_label, rhoS_label : str or None
        Optional text labels inside each sphere.

    elev, azim : float
        Shared viewing angles.

    wspace : float
        Horizontal spacing between subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib axes
    rho : ndarray, shape (2, 2)
        Reduced density matrix.

    r : ndarray, shape (3,)
        Bloch vector of rho.

    im : matplotlib scatter artist
        GQS scatter artist.
    """

    fig = plt.figure(figsize=figsize)

    ax_gqs = fig.add_subplot(1, 2, 1, projection="3d")
    ax_rho = fig.add_subplot(1, 2, 2, projection="3d")

    # -------------------------
    # Left: GQS
    # -------------------------
    ax_gqs, im = GQS_Bloch_Sphere_chi(
        chi_S=states,
        qz=weights,
        m_cmap=m_cmap,
        ax=ax_gqs,
        add_colorbar=add_colorbar,
        show=False,
        title=gqs_title,
        env_label=env_label,
        elev=elev,
        azim=azim,
    )

    # -------------------------
    # Right: Density matrix
    # -------------------------
    rho, r, _, ax_rho = plot_density_matrix_from_gqs(
        states=states,
        weights=weights,
        ax=ax_rho,
        title=rho_title,
        rhoS_label=rhoS_label,
        elev=elev,
        azim=azim,
    )

    # Layout
    fig.subplots_adjust(wspace=wspace)

    if fname is not None:
        fig.savefig(fname, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax_gqs, ax_rho), rho, r, im


