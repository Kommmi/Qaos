import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ot
from .states import Initial_state
from .dynamics import Hamiltonian_QK, floquet_operator_from_H

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


def build_initial_state_batch(nqubit, thetaArr, phiArr, dhilbert=2):
    """
    Build initial pure states as columns of shape (dim, ntraj).

    Parameters
    ----------
    nqubit : int
        Number of qubits.
    thetaArr, phiArr : array-like
        Lists/arrays of Bloch angles for each trajectory.
    dhilbert : int, default=2
        Local Hilbert-space dimension.

    Returns
    -------
    Psi_all : ndarray, shape (dim, ntraj), complex
        Initial states stored column-wise.
    """
    thetaArr = np.asarray(thetaArr, dtype=float)
    phiArr = np.asarray(phiArr, dtype=float)

    if len(thetaArr) != len(phiArr):
        raise ValueError("thetaArr and phiArr must have the same length.")

    ntraj = len(thetaArr)
    dim = dhilbert ** nqubit

    Psi_all = np.zeros((dim, ntraj), dtype=complex)
    for j, (theta0, phi0) in enumerate(zip(thetaArr, phiArr)):
        Psi_all[:, j] = Initial_state(nqubit, theta0, phi0)

    return Psi_all


def evolve_reduced_single_site(
    tau,
    kappa,
    nqubit,
    site,
    nkicks,
    thetaArr,
    phiArr,
    dhilbert=2,
):
    """
    Evolve the kicked system and record single-site reduced-state decomposition
    at each time step.

    Parameters
    ----------
    tau, kappa : float
        Floquet parameters.
    nqubit : int
        Number of qubits.
    site : int
        Site index for reduced state.
    nkicks : int
        Number of kicks.
    thetaArr, phiArr : array-like
        Initial-state angles for each trajectory.
    dhilbert : int, default=2
        Local Hilbert-space dimension.

    Returns
    -------
    chi_steps : ndarray, shape (nkicks+1, ntraj, dE, dhilbert)
    lam_steps : ndarray, shape (nkicks+1, ntraj, dE)
    """
    ntraj = len(thetaArr)
    dE = dhilbert ** (nqubit - 1)

    H1, H2 = Hamiltonian_QK(tau, kappa, nqubit)
    U_F = floquet_operator_from_H(H1, H2, tau)

    Psi_all = build_initial_state_batch(
        nqubit=nqubit,
        thetaArr=thetaArr,
        phiArr=phiArr,
        dhilbert=dhilbert,
    )

    chi_steps = np.zeros((nkicks + 1, ntraj, dE, dhilbert), dtype=complex)
    lam_steps = np.zeros((nkicks + 1, ntraj, dE), dtype=float)

    # t = 0
    chi_steps[0], lam_steps[0] = Reduced_state_single_site_batch(
        dhilbert, nqubit, site, Psi_all
    )

    # t = 1, ..., nkicks
    for t in range(1, nkicks + 1):
        Psi_all = U_F @ Psi_all
        chi_steps[t], lam_steps[t] = Reduced_state_single_site_batch(
            dhilbert, nqubit, site, Psi_all
        )

    return chi_steps, lam_steps


def time_aggregated_gqs_histogram_geometric(
    chi_steps,
    lam_steps,
    nkicks,
    ntraj,
    dhilbert,
    nqubit,
    n_theta=50,
    n_phi=100,
    include_last=True,
    method="equal_area",   # "equal_area" (recommended) or "sin_weighted"
    eps=1e-15,
):
    """
    Geometrically-correct (area-aware) time-aggregated GQS histogram on the Bloch sphere.

    You get a discrete probability measure on S^2 represented on a (theta, phi) grid:
        nu_T(B_ij) = (1/T) sum_t Q(B_ij, t)

    Two geometry-correct options:

    1) method="equal_area" (recommended):
       Use bins uniform in u = cos(theta), so each (u,phi) bin corresponds to equal area on S^2.
       This avoids having to divide by sin(theta).

    2) method="sin_weighted":
       Use bins uniform in theta, but interpret the histogram as "mass per unit area" by dividing
       each theta-row by its bin's surface area: area_ij = Δphi * (cos(theta_i) - cos(theta_{i+1})).

    Returns
    -------
    mass : (ntraj, n_theta, n_phi) array
        Probability mass in each spherical bin (sums to 1 per trajectory).
        This is the discretized measure ν_T itself.

    density : (ntraj, n_theta, n_phi) array
        Estimated density w.r.t. surface area on S^2 (approximately integrates to 1).
        For OT on a grid, you typically use `mass`. For plotting "density per area", use `density`.

    theta_edges, phi_edges : arrays
        Bin edges in theta and phi.

    Notes
    -----
    - `mass` is the correct discrete probability measure on the partition.
    - `density` differs only by dividing by bin area; useful for visualization as a density.
    """
    dE = int(dhilbert ** (nqubit - 1))
    T = nkicks + 1 if include_last else nkicks

    # Reshape to (T, ntraj, dE, dhilbert) and (T, ntraj, dE)
    chi = np.asarray(chi_steps).reshape(nkicks + 1, ntraj, dE, dhilbert)[:T]
    qz  = np.asarray(lam_steps).reshape(nkicks + 1, ntraj, dE)[:T]

    # --- Bloch coordinates from chi (pure states) ---
    # chi[...,0]=a, chi[...,1]=b for qubit
    a = chi[..., 0]
    b = chi[..., 1]
    sx = 2.0 * np.real(np.conj(a) * b)
    sy = 2.0 * np.imag(np.conj(a) * b)
    sz = np.real(np.conj(a) * a - np.conj(b) * b)

    # Spherical coords
    theta = np.arccos(np.clip(sz, -1.0, 1.0))          # [0, pi]
    phi   = np.mod(np.arctan2(sy, sx), 2.0*np.pi)      # [0, 2pi)

    # --- Bin edges ---
    phi_edges = np.linspace(0.0, 2.0*np.pi, n_phi + 1)

    # Equal-area bins: uniform in u = cos(theta) with INCREASING edges

    if method == "equal_area":
        # Uniform in u = cos(theta) ∈ [-1, 1] gives equal-area theta bands.
        u_edges = np.linspace(1.0, -1.0, n_theta + 1)  # from north pole to south pole
        theta_edges = np.arccos(np.clip(u_edges, -1.0, 1.0))

        # We'll histogram in (u, phi) but store output as (theta-bin-index, phi-bin-index)
        # Equal-area bins: uniform in u = cos(theta) with INCREASING edges
        u_edges = np.linspace(-1.0, 1.0, n_theta + 1)          # ✅ increasing
        phi_edges = np.linspace(0.0, 2.0*np.pi, n_phi + 1)     # ✅ increasing

        u = np.cos(theta)  # or u = sz (they're the same for pure states up to numerical error)

        mass = np.zeros((ntraj, n_theta, n_phi), dtype=float)

        for t in range(T):
            for j in range(ntraj):
                H, _, _ = np.histogram2d(
                    u[t, j].ravel(),
                    phi[t, j].ravel(),
                    bins=[u_edges, phi_edges],
                    weights=qz[t, j].ravel()
                )
                mass[j] += H

        mass /= float(T)
        mass /= mass.sum(axis=(1, 2), keepdims=True)  # normalize per trajectory
        # Numerical safety renorm
        s = mass.sum(axis=(1, 2), keepdims=True)
        mass = np.divide(mass, s, out=np.zeros_like(mass), where=(s > 0))

        # Density per unit area: divide by bin areas (constant across theta for equal-area bands? not constant in phi)
        # Area of bin (u_i..u_{i+1}, phi_j..phi_{j+1}) = Δphi * (u_i - u_{i+1})
        du = np.diff(u_edges)          # positive
        dphi = np.diff(phi_edges)
        area = du[:, None] * dphi[None, :]  # (n_theta, n_phi)
        density = mass / np.maximum(area[None, :, :], 1e-15)

        return mass, density, theta_edges, phi_edges

    elif method == "sin_weighted":
        # Uniform in theta; mass is still correct probability mass per bin,
        # but density should divide by true spherical surface area of each bin.
        theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
        mass = np.zeros((ntraj, n_theta, n_phi), dtype=float)

        for t in range(T):
            for j in range(ntraj):
                H, _, _ = np.histogram2d(
                    theta[t, j].ravel(), phi[t, j].ravel(),
                    bins=[theta_edges, phi_edges],
                    weights=qz[t, j].ravel()
                )
                mass[j] += H

        mass /= float(T)
        s = mass.sum(axis=(1, 2), keepdims=True)
        mass = np.divide(mass, s, out=np.zeros_like(mass), where=(s > 0))

        # True area of bin: ∫_{phi_j}^{phi_{j+1}} ∫_{theta_i}^{theta_{i+1}} sinθ dθ dφ
        # = Δphi * (cos(theta_i) - cos(theta_{i+1}))
        dphi = np.diff(phi_edges)  # (n_phi,)
        band = (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))  # (n_theta,), positive
        area = band[:, None] * dphi[None, :]  # (n_theta, n_phi)
        density = mass / np.maximum(area[None, :, :], eps)

        return mass, density, theta_edges, phi_edges

    else:
        raise ValueError("method must be 'equal_area' or 'sin_weighted'")


def compute_time_aggregated_gqs_mass(
    chi_steps,
    lam_steps,
    dhilbert,
    nqubit,
    n_theta=30,
    n_phi=30,
    include_last=True,
    method="equal_area",
):
    """
    Compute time-aggregated GQS histogram from recorded chi/lambda trajectories.

    Parameters
    ----------
    chi_steps : ndarray, shape (nkicks+1, ntraj, dE, dhilbert)
    lam_steps : ndarray, shape (nkicks+1, ntraj, dE)
    dhilbert, nqubit : int
    n_theta, n_phi : int
        Histogram resolution.
    include_last : bool
    method : str
        Histogram binning method.

    Returns
    -------
    mass : ndarray
    density : ndarray
    theta_edges : ndarray
    phi_edges : ndarray
    """
    nkicks = chi_steps.shape[0] - 1
    ntraj = chi_steps.shape[1]
    dE = chi_steps.shape[2]

    chiB = chi_steps.reshape(nkicks + 1, ntraj * dE, dhilbert)
    qzB = lam_steps.reshape(nkicks + 1, ntraj * dE)

    mass, density, theta_edges, phi_edges = time_aggregated_gqs_histogram_geometric(
        chi_steps=chiB,
        lam_steps=qzB,
        nkicks=nkicks,
        ntraj=ntraj,
        dhilbert=dhilbert,
        nqubit=nqubit,
        n_theta=n_theta,
        n_phi=n_phi,
        include_last=include_last,
        method=method,
    )

    return mass, density, theta_edges, phi_edges


def _sphere_bin_centers(theta_edges, phi_edges):
    """Return bin centers in (theta,phi) and Cartesian (x,y,z)."""
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])  # (n_theta,)
    phi_c   = 0.5 * (phi_edges[:-1]   + phi_edges[1:])    # (n_phi,)
    TH, PH = np.meshgrid(theta_c, phi_c, indexing="ij")   # (n_theta,n_phi)

    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    xyz = np.stack([x, y, z], axis=-1)                    # (n_theta,n_phi,3)

    return TH, PH, xyz

def _sphere_bin_areas(theta_edges, phi_edges):
    """
    Surface area of each (theta,phi) bin on the unit sphere:
        area_ij = ∫_{phi_j}^{phi_{j+1}} ∫_{theta_i}^{theta_{i+1}} sinθ dθ dφ
                = Δphi * (cosθ_i - cosθ_{i+1})
    """
    dphi = np.diff(phi_edges)  # (n_phi,)
    band = (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))  # (n_theta,)
    area = band[:, None] * dphi[None, :]  # (n_theta,n_phi)
    return area

def _geodesic_cost_matrix_from_xyz(xyz, p=2, eps=1e-12):
    """
    xyz: (N,3) unit vectors. Returns C_ij = d(i,j)^p where
    d(i,j) = arccos( <xi, xj> ) (geodesic distance on unit sphere).
    """
    X = xyz.reshape(-1, 3)
    G = np.clip(X @ X.T, -1.0, 1.0)
    D = np.arccos(G)  # in [0, pi]
    if p == 1:
        return D
    return np.power(np.maximum(D, eps), p)

def SSCI_calculate(
    nu_mass,                 # (ntraj, n_theta, n_phi) probability masses, sums to 1
    theta_edges,
    phi_edges,
    p=2,
    Z0_xyz=(0.0, 0.0, 1.0),  # Dirac location (default north pole)
    sigma_mode="area",       # "area" (recommended) or "uniform_bins"
    ot_method="emd",         # "emd" (exact) or "sinkhorn"
    sinkhorn_reg=1e-2,
    check_normalization=True,
):
    """
    Compute State-Space Coverage Index for each trajectory:
        S_p(T) = 1 - W_p(nu, sigma)/W_p(delta_Z0, sigma)

    Parameters
    ----------
    nu_mass : ndarray
        Shape (ntraj, n_theta, n_phi). Each trajectory is a probability mass
        on the spherical bins (should sum to 1).
    theta_edges, phi_edges : ndarray
        Bin edges defining the partition of the sphere.
    p : int or float
        Wasserstein order.
    Z0_xyz : tuple
        Dirac location on sphere, unit vector.
    sigma_mode : str
        - "area": sigma proportional to spherical bin areas (true uniform on S^2).
        - "uniform_bins": sigma uniform over bins (appropriate if you used equal-area bins).
    ot_method : str
        - "emd": exact OT via POT ot.emd2 (can be slow for many bins).
        - "sinkhorn": entropic OT via POT ot.sinkhorn2 (faster, approximate).
    sinkhorn_reg : float
        Entropic regularization epsilon for Sinkhorn.
    check_normalization : bool
        If True, renormalize nu_mass rows and sigma defensively.

    Returns
    -------
    S : ndarray, shape (ntraj,)
        Coverage index per trajectory (typically in [0,1], may slightly exceed due to discretization).
    W_nu_sigma : ndarray, shape (ntraj,)
        Wasserstein distances W_p(nu, sigma).
    W_delta_sigma : float
        Denominator Wasserstein distance W_p(delta_Z0, sigma).
    meta : dict
        Contains sigma vector, cost matrix shape, etc.
    """

    nu_mass = np.asarray(nu_mass, dtype=float)
    ntraj, n_theta, n_phi = nu_mass.shape
    N = n_theta * n_phi

    # --- Build sigma on bins ---
    if sigma_mode == "area":
        area = _sphere_bin_areas(theta_edges, phi_edges)  # (n_theta,n_phi)
        sigma = area.reshape(-1)
    elif sigma_mode == "uniform_bins":
        sigma = np.ones(N, dtype=float)
    else:
        raise ValueError("sigma_mode must be 'area' or 'uniform_bins'")

    # Normalize sigma
    sigma = sigma / sigma.sum()

    # Normalize nu (defensive)
    nu = nu_mass.reshape(ntraj, -1)
    if check_normalization:
        s = nu.sum(axis=1, keepdims=True)
        nu = np.divide(nu, s, out=np.zeros_like(nu), where=(s > 0))

    # --- Cost matrix on bin centers (geodesic distance^p) ---
    _, _, xyz = _sphere_bin_centers(theta_edges, phi_edges)     # (n_theta,n_phi,3)
    xyz_flat = xyz.reshape(-1, 3)
    C = _geodesic_cost_matrix_from_xyz(xyz_flat, p=p)           # (N,N)

    # --- Denominator: W_p(delta_{Z0}, sigma) ---
    Z0 = np.asarray(Z0_xyz, dtype=float)
    Z0 = Z0 / np.linalg.norm(Z0)
    # find closest bin center to Z0 (max dot)
    dots = xyz_flat @ Z0
    i0 = int(np.argmax(dots))

    # For a Dirac delta at i0, OT cost to sigma is fixed:
    # W_p^p = sum_j sigma_j * d(i0,j)^p = sum_j sigma_j * C[i0,j]
    W_delta_p = float(np.sum(sigma * C[i0, :]))
    W_delta = W_delta_p ** (1.0 / p)

    # --- Numerator: W_p(nu, sigma) for each trajectory ---
    W_nu_sigma = np.zeros(ntraj, dtype=float)

    for k in range(ntraj):
        a = nu[k]
        if ot_method == "emd":
            # emd2 returns optimal transport cost = sum_ij T_ij * C_ij = W_p^p
            cost_p = ot.emd2(a, sigma, C)
        elif ot_method == "sinkhorn":
            # sinkhorn2 returns regularized cost; use it as an approximation to W_p^p
            cost_p = ot.sinkhorn2(a, sigma, C, reg=sinkhorn_reg)
            cost_p = float(cost_p)
        else:
            raise ValueError("ot_method must be 'emd' or 'sinkhorn'")
        W_nu_sigma[k] = cost_p ** (1.0 / p)

    # --- Coverage index ---
    # S = 1 - W(nu,sigma)/W(delta,sigma)
    # Guard against divide-by-zero (shouldn't happen unless sigma degenerate)
    if W_delta <= 0:
        S = np.full(ntraj, np.nan)
    else:
        S = 1.0 - (W_nu_sigma / W_delta)

    meta = {
        "sigma": sigma,
        "i0": i0,
        "Z0_xyz": Z0,
        "N_bins": N,
        "cost_shape": C.shape,
        "sigma_mode": sigma_mode,
        "ot_method": ot_method,
        "p": p,
    }
    return S, W_nu_sigma, W_delta, meta

def run_gqs_mass(
    kappa,
    tau=1.0,
    nqubit=3,
    site=0,
    nkicks=50000,
    thetaArr=(np.pi/2,),
    phiArr=(np.pi/2,),
    dhilbert=2,
    n_theta=30,
    n_phi=30,
    include_last=True,
    method="equal_area",
):
    """
    Full pipeline: evolve system for one kappa and compute time-aggregated GQS mass.

    Returns
    -------
    result : dict
        Dictionary containing chi_steps, lam_steps, mass, density, theta_edges,
        phi_edges, and metadata.
    """
    chi_steps, lam_steps = evolve_reduced_single_site(
        tau=tau,
        kappa=kappa,
        nqubit=nqubit,
        site=site,
        nkicks=nkicks,
        thetaArr=thetaArr,
        phiArr=phiArr,
        dhilbert=dhilbert,
    )

    mass, density, theta_edges, phi_edges = compute_time_aggregated_gqs_mass(
        chi_steps=chi_steps,
        lam_steps=lam_steps,
        dhilbert=dhilbert,
        nqubit=nqubit,
        n_theta=n_theta,
        n_phi=n_phi,
        include_last=include_last,
        method=method,
    )

    return {
        "kappa": kappa,
        "tau": tau,
        "nqubit": nqubit,
        "site": site,
        "nkicks": nkicks,
        "thetaArr": np.asarray(thetaArr),
        "phiArr": np.asarray(phiArr),
        "chi_steps": chi_steps,
        "lam_steps": lam_steps,
        "mass": mass,
        "density": density,
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
    }


def plot_gqs_mass_mollweide(
    mass,
    theta_edges,
    phi_edges,
    traj_index=0,
    cmap="viridis",
    title=None,
):
    """
    Plot time-aggregated GQS mass on Bloch sphere using
    Mollweide equal-area projection.

    Parameters
    ----------
    mass : ndarray
        Shape (ntraj, n_theta, n_phi)
    theta_edges, phi_edges : arrays
        Bin edges used to construct histogram
    traj_index : int
        Which trajectory to plot
    """
    # Select trajectory
    # --------------------------------------------------
    M = np.asarray(mass[traj_index]).copy()

    # --------------------------------------------------
    # Bin centers
    # --------------------------------------------------
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Mollweide latitude:
    lat_c = np.pi / 2 - theta_c

    # --------------------------------------------------
    # Convert longitude to [-pi, pi)
    # --------------------------------------------------
    if np.min(phi_edges) >= -1e-12 and np.max(phi_edges) > np.pi:
        lon_c = (phi_c + np.pi) % (2 * np.pi) - np.pi
        phi_order = np.argsort(lon_c)
        lon_c = lon_c[phi_order]
        M = M[:, phi_order]

    else:
        # phi is already stored in [-pi, pi].
        lon_c = phi_c

    # Grid shape: (n_theta, n_phi)
    LON, LAT = np.meshgrid(lon_c, lat_c, indexing="xy")

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection="mollweide")

    M_masked = np.ma.masked_where(M < 1e-12, M)  # Mask near-zero bins

    im = ax.pcolormesh(
        LON,
        LAT,
        M_masked,
        shading="auto",
        cmap=cmap,
    )

    ax.grid(True)

    if title is not None:
        ax.set_title(title)

    ax.set_xticks([-5*np.pi/6, -2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/6,0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6])
    ax.set_xticklabels([" ", " ", "-90°", " ", " ", "0°", " ", " ", "90°", " ", " "],fontsize=10)

    ax.set_yticks([-5*np.pi/12,-np.pi/3,-np.pi/4, -np.pi/6,-np.pi/12, 0,np.pi/12, np.pi/6,np.pi/4, np.pi/3, 5*np.pi/12])
    ax.set_yticklabels([" ", " ", "-45°", " ", " ", "0°", " ", " ", "45°", " ", " "],fontsize=10)

    cbar = plt.colorbar(im, ax=ax, pad=0.08, orientation="horizontal", shrink=0.8)
    vmin, vmax = im.get_clim()
    cbar.set_ticks(np.linspace(vmin, vmax, 3))
    cbar.set_label("Aggregated GQS Mass")

    plt.tight_layout()
    plt.show()

def state_space_coverage_index(kappa=0.5,
    tau=1.0,
    nqubit=3,
    site=0,
    nkicks=5000,
    theta=np.pi/2,
    phi=np.pi/2,
    dhilbert=2,
    n_theta=30,
    n_phi=30,
    include_last=True,
    method="equal_area",
    show_plot=False,
    ):

    """
    Compute State-Space Coverage Index (SSCI) for a given set of parameters.

    Returns
    -------
    S : ndarray, shape (ntraj,) 
        Coverage index per trajectory.
    W_nu_sigma : ndarray, shape (ntraj,)
        Wasserstein distances W_p(nu, sigma).
    W_delta_sigma : float
        Denominator Wasserstein distance W_p(delta_Z0, sigma).
    meta : dict
        Metadata including sigma vector, cost matrix shape, etc.
    """

    thetaArr = (theta,)
    phiArr = (phi,)

    # Run the full GQS mass computation pipeline
    result = run_gqs_mass(
        kappa=kappa,
        tau=tau,
        nqubit=nqubit,
        site=site,
        nkicks=nkicks,
        thetaArr=thetaArr,
        phiArr=phiArr,
        dhilbert=dhilbert,
        n_theta=n_theta,
        n_phi=n_phi,
        include_last=include_last,
        method=method,
    )

    mass = result["mass"]
    theta_edges = result["theta_edges"]
    phi_edges = result["phi_edges"]

    # Compute SSCI
    S,_,_,_ = SSCI_calculate(
        nu_mass=mass,   
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        p=1,
        Z0_xyz=(0.0, 0.0, 1.0),  # North pole
        sigma_mode="uniform_bins",  # Use uniform bins for sigma
        ot_method="emd",
    )

    colors_purples = ["#dbcbfa", "#a060dc", "#6924AA", "#410479"]
    cmap_purples = LinearSegmentedColormap.from_list("purple_to_dark", colors_purples)
    m_cmap = cmap_purples
        # Plot the time-aggregated GQS mass on the Bloch sphere (optional)
    if show_plot:
        plot_gqs_mass_mollweide(
            mass=mass,
            theta_edges=theta_edges,
            phi_edges=phi_edges,
            traj_index=0,
            cmap=m_cmap,
            title=f"State space coverage Index: {S[0]:.2f}",) 
    
    return S
