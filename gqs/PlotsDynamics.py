import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


from .distances import _mask_chi_lambda
from .gqs import bloch_from_chi, aggregate_bloch
from .states import Reduced_state_single_site, rho_single_spin
from .distances import Quantum_EMD, bures_distance


def Reduced_state_single_site_batch(d_hilbert, n_chain, system_site, Psi_all, eps=1e-12):
    """
    Batch equivalent of your Reduced_state_single_site, for Psi_all with columns = trajectories.

    Parameters
    ----------
    Psi_all : ndarray, shape (d_hilbert**n_chain, ntraj), complex

    Returns
    -------
    chi_S_all : ndarray, shape (ntraj, dE, d), complex
    lambda_E_all : ndarray, shape (ntraj, dE), float
    """
    d = int(d_hilbert)
    n = int(n_chain)
    if not (0 <= system_site < n):
        raise ValueError(f"system_site must be in [0, {n-1}]")

    Psi_all = np.asarray(Psi_all)
    dim, ntraj = Psi_all.shape
    expected = d**n
    if dim != expected:
        raise ValueError(f"Psi_all must have shape ({expected}, ntraj). Got ({dim}, {ntraj}).")

    # (1) Tensor view per trajectory: (ntraj, d, d, ..., d)
    Psi_tensor = Psi_all.T.reshape((ntraj,) + (d,) * n)

    # (2) Put chosen site first (after traj axis):
    # current axes: (traj, 0,1,2,...,n-1)
    # we want:      (traj, system_site, others...)
    axes = [0, 1 + system_site] + [1 + i for i in range(n) if i != system_site]
    Psi_perm = np.transpose(Psi_tensor, axes=axes)  # (ntraj, d, d, d, ...)

    # (3) Flatten env indices: A has shape (ntraj, d, dE)
    dE = d**(n - 1)
    A = Psi_perm.reshape(ntraj, d, dE)

    # (4) lambda_E[a] = sum_k |A_{k,a}|^2
    lambda_E_all = np.sum(np.abs(A)**2, axis=1)  # (ntraj, dE)

    # (5) chi[:,mask] = A[:,mask] / sqrt(lambda)
    chi_S_all = np.zeros((ntraj, dE, d), dtype=complex)  # (traj, env, system_dim)
    mask = lambda_E_all > eps

    # columns as (traj, env, d)
    cols = np.transpose(A, (0, 2, 1))  # (ntraj, dE, d)

    denom = np.sqrt(lambda_E_all[mask])  # 1D of all valid entries
    chi_S_all[mask] = cols[mask] / denom[:, None]

    return chi_S_all, lambda_E_all

def _format_bloch_angle_axes(ax, wrap_phi=True, fontsize=12):
    """Common axis formatting for Bloch-sphere angle plots."""
    if wrap_phi:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(
            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
            fontsize=fontsize
        )
    else:
        ax.set_xlim(0, 2*np.pi)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(
            [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
            fontsize=fontsize
        )

    ax.set_ylim(0, np.pi)
    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=fontsize)
    ax.grid(False)


def plot_two_gqs_trajectories_on_ax(
    ax,
    chi_steps,                  # shape (T, ntraj, dE, 2)
    lam_steps,                  # shape (T, ntraj, dE)
    traj_indices=(0, 1),
    wrap_phi=True,
    marker_size=18,
    marker_sizePast=10,
    alpha_past=0.2,
    alpha_final=1.0,
    cmaps=('Reds', 'Blues'),   # NEW: one colormap per trajectory
    aggregate=True,
    aggregate_decimals=10,
    show_title=True,
    title="QKT GQS",
    final_edgecolor="black",
    final_linewidth=0.4,
    final_size_scale=1.8,
    panel_label=None
):
    """
    Plot two GQS trajectories. Each trajectory is plotted using its own colormap.

    Parameters
    ----------
    chi_steps : ndarray
        Shape (T, ntraj, dE, 2)

    lam_steps : ndarray
        Shape (T, ntraj, dE)

    traj_indices : tuple of length 2
        Which two trajectories to plot.

    cmaps : tuple of length 2
        Colormaps for the two trajectories.
    """

    chi_steps = np.asarray(chi_steps)
    lam_steps = np.asarray(lam_steps)

    if chi_steps.ndim != 4 or chi_steps.shape[-1] != 2:
        raise ValueError("chi_steps must have shape (T, ntraj, dE, 2).")

    if lam_steps.ndim != 3:
        raise ValueError("lam_steps must have shape (T, ntraj, dE).")

    T, ntraj, dE, dh = chi_steps.shape

    if dh != 2:
        raise ValueError(
            "This plotting function is for qubit reduced states, so last axis must be 2."
        )

    if len(traj_indices) != 2:
        raise ValueError("traj_indices must contain exactly two trajectory indices.")

    if len(cmaps) != 2:
        raise ValueError("cmaps must contain exactly two colormaps.")

    i0, i1 = traj_indices
    if not (0 <= i0 < ntraj and 0 <= i1 < ntraj):
        raise ValueError(f"traj_indices must be between 0 and {ntraj - 1}.")

    _format_bloch_angle_axes(ax, wrap_phi=wrap_phi)

    
    # --------------------------------------------------
    # Define colormaps
    # --------------------------------------------------
    cmap_reference = LinearSegmentedColormap.from_list(
        "warm_to_red",
        [
            "#ffb3b3",
            "#ff4d4d",
            "#b30000",
            "#5a0000",
        ],
    )

    cmap_perturbed = LinearSegmentedColormap.from_list(
        "sky_to_blue",
        [
            "#aed8f2",
            "#4a90e2",
            "#2467AA",
            "#023360",
        ],
    )

    cmaps = (
        cmap_reference,
        cmap_perturbed,
    )


    markers = ["o", "o"]

    sc_last = None
    sc_last_list = []

    out = []

    for marker, tr, cmap_tr in zip(markers, traj_indices, cmaps):

        phi_t_list = []
        theta_t_list = []
        lam_t_list = []

        for t in range(T):
            chi = chi_steps[t, tr]      # shape (dE, 2)
            lam = lam_steps[t, tr]      # shape (dE,)

            # safe normalize
            norms = np.linalg.norm(chi, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            chi = chi / norms

            a = chi[:, 0]
            b = chi[:, 1]

            sx = 2 * np.real(np.conj(a) * b)
            sy = 2 * np.imag(np.conj(a) * b)
            sz = np.real(np.conj(a) * a - np.conj(b) * b)

            if aggregate:
                sx, sy, sz, lam = aggregate_bloch(
                    sx, sy, sz, lam, decimals=aggregate_decimals
                )

            theta = np.arccos(np.clip(sz, -1, 1))
            phi = np.arctan2(sy, sx)

            if not wrap_phi:
                phi = (phi + 2 * np.pi) % (2 * np.pi)

            phi_t_list.append(phi)
            theta_t_list.append(theta)
            lam_t_list.append(lam)

            if t < T - 1:
                ax.scatter(
                    phi,
                    theta,
                    c=lam,
                    cmap=cmap_tr,
                    vmin=0,
                    vmax=1,
                    s=marker_sizePast,
                    alpha=alpha_past,
                    marker=marker,
                )

            else:
                sc_last = ax.scatter(
                    phi,
                    theta,
                    c=lam,
                    cmap=cmap_tr,
                    vmin=0,
                    vmax=1,
                    s=marker_size * final_size_scale,
                    alpha=alpha_final,
                    marker=marker,
                    edgecolors=final_edgecolor,
                    linewidths=final_linewidth,
                    zorder=10,
                )

                sc_last_list.append(sc_last)

        out.append({
            "traj_index": tr,
            "phi": phi_t_list,
            "theta": theta_t_list,
            "lam": lam_t_list,
        })

    if show_title:
        ax.set_title(title, fontsize=14)

    if panel_label is not None:
        ax.text(
            -0.2, 1.05, panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top"
        )

    return sc_last_list, out


def plot_two_gqs_trajectory_row(
    U_F,
    psi_reference,
    psi_perturbed,
    selected_kicks=(0, 1, 2, 3, 4, 200),
    dhilbert=2,
    site=0,
    marker_size=12,
    marker_sizePast=3,
    alpha_past=0.20,
    alpha_final=1.0,
    aggregate=True,
    aggregate_decimals=10,
    show_w1=True,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=11,
    w1_fontsize=10,
    last_col_border_width=2.4,
    panel_width=1.55,
    panel_height=2.65,
    wspace=0.12,
    show=True,
):
    """
    Evolve two wave functions using a Floquet operator and plot their
    single-site GQS trajectories at selected kicks.

    The result is a 1 x N figure with two vertical colorbars.

    Parameters
    ----------
    U_F : ndarray, shape (D, D)
        Floquet operator.

    psi_reference : ndarray, shape (D,)
        Initial reference wave function.

    psi_perturbed : ndarray, shape (D,)
        Initial perturbed wave function.

    selected_kicks : sequence of int
        Kicks displayed in the columns.

    dhilbert : int, default=2
        Local Hilbert-space dimension. For qubits, use dhilbert=2.

    site : int, default=0
        Subsystem site retained when constructing the single-site GQS.

    marker_size : float
        Marker size for states at the current kick.

    marker_sizePast : float
        Marker size for states from earlier kicks.

    alpha_past : float
        Transparency of states from earlier kicks.

    alpha_final : float
        Transparency of states at the current kick.

    aggregate : bool
        Passed to plot_two_gqs_trajectories_on_ax.

    aggregate_decimals : int
        Decimal precision used when aggregating coincident GQS points.

    show_w1 : bool
        Whether to display W_1 inside each subplot.

    title_fontsize, label_fontsize, tick_fontsize, w1_fontsize : float
        Font sizes.

    last_col_border_width : float
        Border width of the final selected-kick panel.

    panel_width, panel_height : float
        Controls the overall figure size.

    wspace : float
        Horizontal spacing between panels and colorbars.

    show : bool
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.

    axes : ndarray
        Array containing the GQS subplot axes.

    w1_values : ndarray
        Wasserstein distance at each selected kick.

    chi_steps : ndarray
        Conditional subsystem states at all evolved kicks.

    lam_steps : ndarray
        Corresponding GQS weights at all evolved kicks.
    """

    # --------------------------------------------------
    # Convert and validate inputs
    # --------------------------------------------------
    U_F = np.asarray(U_F, dtype=complex)

    psi_reference = np.asarray(
        psi_reference,
        dtype=complex,
    ).reshape(-1)

    psi_perturbed = np.asarray(
        psi_perturbed,
        dtype=complex,
    ).reshape(-1)

    selected_kicks = np.asarray(
        selected_kicks,
        dtype=int,
    )

    if selected_kicks.size == 0:
        raise ValueError(
            "selected_kicks cannot be empty."
        )

    if np.any(selected_kicks < 0):
        raise ValueError(
            "selected_kicks must contain nonnegative integers."
        )

    if psi_reference.shape != psi_perturbed.shape:
        raise ValueError(
            "psi_reference and psi_perturbed must have the same shape."
        )

    dim = psi_reference.size

    if U_F.shape != (dim, dim):
        raise ValueError(
            f"U_F must have shape ({dim}, {dim}), "
            f"but received {U_F.shape}."
        )

    # --------------------------------------------------
    # Infer the number of qubits/sites
    # --------------------------------------------------
    nqubits_float = np.log(dim) / np.log(dhilbert)
    nqubits = int(round(nqubits_float))

    if dhilbert**nqubits != dim:
        raise ValueError(
            f"The wave-function dimension {dim} is not an exact "
            f"power of dhilbert={dhilbert}."
        )

    if site < 0 or site >= nqubits:
        raise ValueError(
            f"site must be between 0 and {nqubits - 1}."
        )

    ntraj = 2
    ncols = len(selected_kicks)
    max_kicks = int(np.max(selected_kicks))
    dE = dhilbert ** (nqubits - 1)

    # --------------------------------------------------
    # Define colormaps
    # --------------------------------------------------
    cmap_reference = LinearSegmentedColormap.from_list(
        "warm_to_red",
        [
            "#ffb3b3",
            "#ff4d4d",
            "#b30000",
            "#5a0000",
        ],
    )

    cmap_perturbed = LinearSegmentedColormap.from_list(
        "sky_to_blue",
        [
            "#aed8f2",
            "#4a90e2",
            "#2467AA",
            "#023360",
        ],
    )

    cmaps = (
        cmap_reference,
        cmap_perturbed,
    )

    # --------------------------------------------------
    # Store both wave functions as columns
    # --------------------------------------------------
    psi_all = np.column_stack(
        [
            psi_reference,
            psi_perturbed,
        ]
    )

    # Normalize both initial wave functions
    norms = np.linalg.norm(
        psi_all,
        axis=0,
        keepdims=True,
    )

    if np.any(np.isclose(norms, 0.0)):
        raise ValueError(
            "The initial wave functions must have nonzero norm."
        )

    psi_all = psi_all / norms

    # --------------------------------------------------
    # Allocate arrays for the GQS trajectories
    # --------------------------------------------------
    chi_steps = np.zeros(
        (
            max_kicks + 1,
            ntraj,
            dE,
            dhilbert,
        ),
        dtype=complex,
    )

    lam_steps = np.zeros(
        (
            max_kicks + 1,
            ntraj,
            dE,
        ),
        dtype=float,
    )

    # --------------------------------------------------
    # GQS at kick zero
    # --------------------------------------------------
    chi_steps[0], lam_steps[0] = (
        Reduced_state_single_site_batch(
            dhilbert,
            nqubits,
            site,
            psi_all,
        )
    )

    # --------------------------------------------------
    # Floquet evolution
    # --------------------------------------------------
    for nkick in range(1, max_kicks + 1):
        psi_all = U_F @ psi_all

        chi_steps[nkick], lam_steps[nkick] = (
            Reduced_state_single_site_batch(
                dhilbert,
                nqubits,
                site,
                psi_all,
            )
        )

    # --------------------------------------------------
    # Figure layout
    #
    # N plot columns + colorbar + spacer + colorbar
    # --------------------------------------------------
    width_ratios = (
        [1.0] * ncols
        + [0.035, 0.015, 0.035]
    )

    figure_width = panel_width * ncols + 1.2

    fig = plt.figure(
        figsize=(figure_width, panel_height)
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=ncols + 3,
        width_ratios=width_ratios,
        wspace=wspace,
    )

    axes = np.array(
        [
            fig.add_subplot(gs[0, col])
            for col in range(ncols)
        ],
        dtype=object,
    )

    w1_values = np.zeros(
        ncols,
        dtype=float,
    )

    # --------------------------------------------------
    # Plot each selected kick
    # --------------------------------------------------
    for col, nkicks in enumerate(selected_kicks):
        ax = axes[col]

        # Full trajectory history through the current kick
        chi_history = chi_steps[:nkicks + 1]
        lam_history = lam_steps[:nkicks + 1]

        # Reference GQS at the current kick
        chi_ref = chi_steps[nkicks, 0]
        lam_ref = lam_steps[nkicks, 0]

        # Perturbed GQS at the current kick
        chi_pert = chi_steps[nkicks, 1]
        lam_pert = lam_steps[nkicks, 1]

        # Wasserstein distance at the current kick
        w1 = Quantum_EMD(
            chi_ref,
            lam_ref,
            chi_pert,
            lam_pert,
        )

        w1_values[col] = w1

        # Plot both trajectories
        plot_two_gqs_trajectories_on_ax(
            ax,
            chi_history,
            lam_history,
            traj_indices=(0, 1),
            cmaps=cmaps,
            marker_size=marker_size,
            marker_sizePast=marker_sizePast,
            alpha_past=alpha_past,
            alpha_final=alpha_final,
            aggregate=aggregate,
            aggregate_decimals=aggregate_decimals,
            show_title=False,
        )

        # Panel title
        ax.set_title(
            rf"$t_n  = {nkicks}$",
            fontsize=title_fontsize,
            pad=5,
        )

        # Wasserstein distance inside panel
        if show_w1:
            ax.text(
                0.04,
                0.94,
                rf"$W_1={w1:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=w1_fontsize,
            )

        # x-axis
        ax.set_xlabel(
            r"$\phi$",
            fontsize=label_fontsize,
            labelpad=-4,
        )

        ax.set_xticks(
            [-np.pi, np.pi]
        )

        ax.set_xticklabels(
            [r"$-\pi$", r"$\pi$"],
            fontsize=tick_fontsize,
        )

        # y-axis only on the first panel
        if col == 0:
            ax.set_ylabel(
                r"$\theta$",
                fontsize=label_fontsize,
                labelpad=-4,
            )

            ax.set_yticks(
                [0, np.pi]
            )

            ax.set_yticklabels(
                [r"$0$", r"$\pi$"],
                fontsize=tick_fontsize,
            )

        else:
            ax.set_ylabel("")
            ax.set_yticks([])

        # Emphasize the final selected-kick panel
        if col == ncols - 1:
            for spine in ax.spines.values():
                spine.set_linewidth(
                    last_col_border_width
                )
                spine.set_color("black")

    # --------------------------------------------------
    # Shared colorbars
    # --------------------------------------------------
    norm = mcolors.Normalize(
        vmin=0,
        vmax=1,
    )

    cax_reference = fig.add_subplot(
        gs[0, ncols]
    )

    cax_perturbed = fig.add_subplot(
        gs[0, ncols + 2]
    )

    reference_mappable = cm.ScalarMappable(
        norm=norm,
        cmap=cmap_reference,
    )
    reference_mappable.set_array([])

    perturbed_mappable = cm.ScalarMappable(
        norm=norm,
        cmap=cmap_perturbed,
    )
    perturbed_mappable.set_array([])

    # Reference colorbar
    cbar_reference = fig.colorbar(
        reference_mappable,
        cax=cax_reference,
        orientation="vertical",
    )

    cbar_reference.set_ticks([0, 1])

    cbar_reference.ax.tick_params(
        labelsize=tick_fontsize,
        length=2,
        width=0.6,
    )

    cbar_reference.set_label(
        r"$Q^S$ Reference",
        fontsize=label_fontsize,
        labelpad=-8,
    )

    cbar_reference.outline.set_linewidth(0.6)

    # Perturbed colorbar
    cbar_perturbed = fig.colorbar(
        perturbed_mappable,
        cax=cax_perturbed,
        orientation="vertical",
    )

    cbar_perturbed.set_ticks([0, 1])

    cbar_perturbed.ax.tick_params(
        labelsize=tick_fontsize,
        length=2,
        width=0.6,
    )

    cbar_perturbed.set_label(
        r"$Q^S$ Perturbed",
        fontsize=label_fontsize,
        labelpad=-4,
    )

    cbar_perturbed.outline.set_linewidth(0.6)

    if show:
        plt.show()

    return fig, axes, w1_values, chi_steps, lam_steps
