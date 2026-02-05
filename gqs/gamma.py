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
from tqdm import tqdm
import matplotlib.pyplot as plt

from .states import Initial_state, Reduced_state_single_site, rho_single_spin
from .dynamics import Hamiltonian_QK, floquet_operator_from_H
from .distances import Quantum_EMD, Psi_Dist, bures_distance
from .perturbations import perturb_theta_phi_isotropic

def Compare_distances_QKT(U,Psi_0,Psi_p,d_hilbert,n_chain,system_site,N_kicks,renormalize=False,show_plt=False):
    """returns the Earth Mover's Distance between perturbed and unperturbed distribution
         as a function of time for a single iteration

    Parameters  
    ----------
    Evolution_rule : function
        map function
    params : array
        parameters of the map function
    x0s : array
        samples from the original distribution
    xNs : array
        samples from the perturbed distribution
    nbins : int
        number of bins for the histogram
    traj_len : int
        number of iterations
    shw_plt : bool
        show the plot of EMD vs time
    

    Returns
    -------
    D_t : array
        Earth Mover's Distance between the two distributions as a function of time
    x0s : array
        samples from the original distribution after some iterations
    xNs : array
        samples from the perturbed distribution
    """
    D_global = np.zeros(N_kicks)
    D_local = np.zeros(N_kicks)
    D_local_rho = np.zeros(N_kicks)
    # 1. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
    Dg0 = Psi_Dist(Psi_0, Psi_p)

    # 2. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
    chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
    chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)

    # 3.Compute the density matrices of the two GQS
    rho_s0 = rho_single_spin(d_hilbert, n_chain, system_site, Psi_0)
    rho_sp = rho_single_spin(d_hilbert, n_chain, system_site, Psi_p)

    # 4. Compute the Quantum EMD between the two GQS
    Dl0 = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep)
    Dr0 = bures_distance(rho_s0, rho_sp)

    # 5. Evolve the two states for N_kicks
    for i in range(N_kicks):
        # 6. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
        D_global[i] = Psi_Dist(Psi_0, Psi_p) / Dg0
        # 7. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
        chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
        chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)
        # 8. Compute the Quantum EMD between the two GQS
        D_local[i] = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep) / Dl0
        # 9. Compute the density matrices of the two GQS
        rho_s0 = rho_single_spin(d_hilbert, n_chain, system_site, Psi_0)
        rho_sp = rho_single_spin(d_hilbert, n_chain, system_site, Psi_p)
        # 10. Compute the Bures distance between the two density matrices
        D_local_rho[i] = bures_distance(rho_s0, rho_sp) / Dr0
        # 11. Evolve the two states
        Psi_0 = U @ Psi_0
        Psi_p = U @ Psi_p
        if renormalize:
            Psi_0 = Psi_0 / np.linalg.norm(Psi_0)
            Psi_p = Psi_p / np.linalg.norm(Psi_p)
    
    if show_plt:
        plt.figure(figsize=(10,3))
        ml1, _, _ = plt.stem(range(N_kicks), D_global-1,linefmt='C0-',markerfmt='C0o',basefmt=' ',label='Global State Distance')
        ml1.set_markersize(2)
        ml2, _, _ = plt.stem(range(N_kicks), D_local-1,linefmt='C1-',markerfmt='C1^',basefmt=' ',label='Quantum EMD')
        ml2.set_markersize(2)
        ml3, _, _ = plt.stem(range(N_kicks), D_local_rho-1,linefmt='C2-',markerfmt='C2s',basefmt=' ',label='Bures Distance')
        ml3.set_markersize(2)
        plt.xlabel('n - Number of Floquet Kicks',fontsize=14)
        plt.ylabel(r'$\frac{d(n)}{d(0)} - 1$',fontsize=14)
        plt.title('Comparison of Global and Local Distances over Time',fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return D_global,D_local,D_local_rho

def LLE_single_its_Quantum_Kicked_Top(U,Psi_0,Psi_p,d_hilbert,n_chain,system_site,N_kicks,renormalize=False):
    """returns the Earth Mover's Distance between perturbed and unperturbed distribution
         as a function of time for a single iteration

    Parameters  
    ----------
    Evolution_rule : function
        map function
    params : array
        parameters of the map function
    x0s : array
        samples from the original distribution
    xNs : array
        samples from the perturbed distribution
    nbins : int
        number of bins for the histogram
    traj_len : int
        number of iterations
    shw_plt : bool
        show the plot of EMD vs time
    

    Returns
    -------
    D_t : array
        Earth Mover's Distance between the two distributions as a function of time
    x0s : array
        samples from the original distribution after some iterations
    xNs : array
        samples from the perturbed distribution
    """
    D_global = np.zeros(N_kicks)
    D_local = np.zeros(N_kicks)
    # 1. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
    Dg0 = Psi_Dist(Psi_0, Psi_p)

    # 2. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
    chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
    chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)

    # 3. Compute the Quantum EMD between the two GQS
    Dl0 = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep)

    #4.Evolve the two states for N_kicks
    for i in range(N_kicks):
        # 5. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
        D_global[i] = Psi_Dist(Psi_0, Psi_p) / Dg0
        # 6. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
        chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
        chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)
        # 7. Compute the Quantum EMD between the two GQS
        D_local[i] = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep) / Dl0
        # 8. Evolve the two states
        Psi_0 = U @ Psi_0
        Psi_p = U @ Psi_p
        if renormalize:
            Psi_0 = Psi_0 / np.linalg.norm(Psi_0)
            Psi_p = Psi_p / np.linalg.norm(Psi_p)
    return D_global,D_local,Psi_0

def LLE_ln_avg_distance_separation(D_t_arr,traj_len):
    """returns the average log of the distance of separation after 
    removing zero entries of the distance array

    Parameters
    ----------
    D_t_arr : array
        Earth Mover's Distance between the two distributions as a function of time
    traj_len : int
        number of iterations

    Returns
    -------
    ln_avg_dist : array
        average log of the distance of separation
    """
    ln_avg_dist = np.zeros(traj_len)
    for k in range(traj_len):
        div_traj_k = D_t_arr[:,k]
        # filter entries where distance is zero (would lead to -inf after log)
        nonzero = np.where(div_traj_k != 0)
        if len(nonzero[0]) == 0:
          # if all entries where zero, we have to use -inf
          ln_avg_dist[k] = -np.inf
        else:
            ln_avg_dist[k] = np.mean(np.log(div_traj_k[nonzero]))
    return ln_avg_dist

def Avg_separation_rate_local(dhilbert,nqubit,system_site,U_F,theta0,phi0,eps,N_traj,N_kicks):
    """
    Compute the average separation rate of local Quantum EMD over N_traj perturbations.

    Parameters
    ----------
    dhilbert : int
        Local Hilbert space dimension (2 for qubits).
    nqubit : int
        Number of qubits in the system.
    system_site : int
        Which qubit is the "system" (0-based index).
    U_F : ndarray
        Floquet operator (dim, dim).
    theta0, phi0 : float
        Initial angles for the reference state.
    eps : float
        Perturbation strength (radians).
    N_traj : int
        Number of random perturbation trajectories to average over.
    N_kicks : int
        Number of kicks (time steps) to evolve.

    Returns
    -------
    avg_rates : ndarray, shape (N_kicks,)
        Average separation rates at each kick.
    """
    # Reference initial state
    Psi_0 = Initial_state(nqubit, theta0, phi0)
    # --- Initialize the distance arrays ---
    D_Global= np.zeros((N_traj,N_kicks))
    D_Local= np.zeros((N_traj,N_kicks))
    for avg in range(N_traj):
        # Perturbed initial state
        Psi_pert, _, _ = perturb_theta_phi_isotropic(nqubit, theta0, phi0, angle_sigma=eps)
        # Evolve and compute rates
        dg,dl,_=LLE_single_its_Quantum_Kicked_Top(U_F,Psi_0,Psi_pert,dhilbert,nqubit,system_site=system_site,N_kicks=N_kicks)
        D_Global[avg,:] = dg
        D_Local[avg,:] = dl
    ln_avg_dist_G = LLE_ln_avg_distance_separation(D_Global,N_kicks)
    ln_avg_dist_L = LLE_ln_avg_distance_separation(D_Local,N_kicks)
    Gamma = np.sum(ln_avg_dist_L) / N_kicks
    avg_D_L = np.mean(D_Local,axis=0)
    avg_D_G = np.mean(D_Global,axis=0)
    return Gamma,ln_avg_dist_L, avg_D_L, ln_avg_dist_G, avg_D_G

def compute_Gamma_for_theta_phi(
    dhilbert, nqubit, system_site, U_F,
    th, ph, eps, N_traj, N_kicks
):
    """
    Worker: compute Gamma + ln_avg_dist_L for a single (theta, phi).
    Returns (key, result_dict) so it can be used like your LLE example.
    """
    Gamma, ln_avg_dist_L, _, _, _ = Avg_separation_rate_local(
        dhilbert, nqubit, system_site, U_F,
        th, ph, eps, N_traj, N_kicks
    )
    return (float(th), float(ph)), {
        "Gamma": float(Gamma),
        "t": np.arange(N_kicks, dtype=int),
        "log_dist": np.asarray(ln_avg_dist_L)
    }

def parallel_Gamma_over_theta_phi_grid(
    dhilbert, nqubit, system_site, U_F,
    theta_array, phi_array, eps, N_traj, N_kicks,
    n_jobs=-1, backend="loky"
):
    """
    Parallel over all (theta, phi) pairs.
    Returns a dict: data[(theta,phi)] = {"Gamma":..., "t":..., "log_dist":...}
    """
    theta_array = np.asarray(theta_array, dtype=float)
    phi_array = np.asarray(phi_array, dtype=float)

    grid = [(th, ph) for th in theta_array for ph in phi_array]

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(compute_Gamma_for_theta_phi)(
            dhilbert, nqubit, system_site, U_F,
            th, ph, eps, N_traj, N_kicks
        )
        for th, ph in tqdm(grid, desc="Parallel over (theta, phi)")
    )

    data = {key: result for key, result in results}
    return data

def Gamma_phase_space_map(
    dhilbert, nqubit, system_site, U_F,
    theta_num, phi_num, eps, N_traj, N_kicks,
    fname=None, n_jobs=-1, backend="loky",
    return_arrays=True
):
    """
    Driver in the style of your LLE_vs_noise_vs_parameters_* function:
      - runs the parallel grid evaluation
      - optionally converts dict -> arrays on the (theta,phi) grid
      - optionally saves to file

    Parameters
    ----------
    fname : str or None
        If provided:
          - if endswith '.npz': saves compressed arrays + metadata
          - else: uses np.save(fname, data_dict) (pickle object array)
    return_arrays : bool
        If True, returns (Gamma_map, ln_avg_dist_L_map, data_dict).
        If False, returns only data_dict.

    Returns
    -------
    If return_arrays:
        Gamma_map : (len(theta), len(phi))
        ln_avg_dist_L_map : (len(theta), len(phi), N_kicks)
        data : dict keyed by (theta,phi)
    Else:
        data : dict
    """
    theta_array = np.linspace(0.0, np.pi, theta_num)
    phi_array   = np.linspace(-np.pi, np.pi, phi_num)

    t0 = time.time()
    data = parallel_Gamma_over_theta_phi_grid(
        dhilbert, nqubit, system_site, U_F,
        theta_array, phi_array, eps, N_traj, N_kicks,
        n_jobs=n_jobs, backend=backend
    )
    runtime = time.time() - t0

    # Optionally build grid arrays (useful for heatmaps later)
    Gamma_map = None
    ln_avg_dist_L_map = None
    if return_arrays:
        theta_num = len(theta_array)
        phi_num = len(phi_array)
        Gamma_map = np.zeros((theta_num, phi_num), dtype=float)
        ln_avg_dist_L_map = np.zeros((theta_num, phi_num, N_kicks), dtype=float)

        # Fill in a consistent order
        for i, th in enumerate(theta_array):
            for j, ph in enumerate(phi_array):
                entry = data[(float(th), float(ph))]
                Gamma_map[i, j] = entry["Gamma"]
                ln_avg_dist_L_map[i, j, :] = entry["log_dist"]

    # Save results
    if fname is not None:
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)

        meta = dict(
            dhilbert=dhilbert,
            nqubit=nqubit,
            system_site=system_site,
            eps=eps,
            N_traj=N_traj,
            N_kicks=N_kicks,
            runtime_seconds=runtime,
            backend=backend,
            n_jobs=n_jobs,
            saved_unix_time=time.time(),
        )

        if fname.endswith(".npz"):
            # Save arrays + coordinate grids + meta. (Does NOT save U_F by default.)
            np.savez_compressed(
                fname,
                Gamma_map=Gamma_map if Gamma_map is not None else np.array([]),
                ln_avg_dist_L_map=ln_avg_dist_L_map if ln_avg_dist_L_map is not None else np.array([]),
                theta_array=theta_array,
                phi_array=phi_array,
                meta=meta,
            )
        else:
            # Save dict (like your code). Note: object pickle under the hood.
            np.save(fname, data, allow_pickle=True)

    if return_arrays:
        return Gamma_map, ln_avg_dist_L_map, data
    return data

def Single_its_QKT(U,Psi_0,Psi_p,d_hilbert,n_chain,system_site,N_kicks,renormalize=False):
    """returns the Earth Mover's Distance between perturbed and unperturbed distribution
         as a function of time for a single iteration

    Parameters  
    ----------
    Evolution_rule : function
        map function
    params : array
        parameters of the map function
    x0s : array
        samples from the original distribution
    xNs : array
        samples from the perturbed distribution
    nbins : int
        number of bins for the histogram
    traj_len : int
        number of iterations
    shw_plt : bool
        show the plot of EMD vs time
    

    Returns
    -------
    D_t : array
        Earth Mover's Distance between the two distributions as a function of time
    """
    D_global = np.zeros(N_kicks)
    D_local_GQS = np.zeros(N_kicks)
    D_local_rho = np.zeros(N_kicks)

    # 1. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
    Dg0 = Psi_Dist(Psi_0, Psi_p)

    # 2a. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
    chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
    chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)
    # 2b. Compute the density matrices of the two GQS
    rho_s0 = rho_single_spin(d_hilbert, n_chain, system_site, Psi_0)
    rho_sp = rho_single_spin(d_hilbert, n_chain, system_site, Psi_p)

    # 3. Compute the Quantum EMD between the two GQS
    Dl0 = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep)
    Dr0 = bures_distance(rho_s0, rho_sp)

    #4.Evolve the two states for N_kicks
    for i in range(N_kicks):
        # 5. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
        D_global[i] = Psi_Dist(Psi_0, Psi_p) / Dg0
        # 6a. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
        chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
        chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)
        # 6b. Compute the Reduced density matrices of the two global states Psi_0 and Psi_p
        rho_s0 = rho_single_spin(d_hilbert, n_chain, system_site, Psi_0)
        rho_sp = rho_single_spin(d_hilbert, n_chain, system_site, Psi_p)
        # 7a. Compute the Quantum EMD between the two GQS
        D_local_GQS[i] = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep) / Dl0
        # 7b. Compute the Bures distance between the two reduced density matrices
        D_local_rho[i] = bures_distance(rho_s0, rho_sp) / Dr0
        # 8. Evolve the two states
        Psi_0 = U @ Psi_0
        Psi_p = U @ Psi_p
        if renormalize:
            Psi_0 = Psi_0 / np.linalg.norm(Psi_0)
            Psi_p = Psi_p / np.linalg.norm(Psi_p)
    return D_global,D_local_GQS,D_local_rho

def Avg_separation_QKT(dhilbert,nqubit,system_site,U_F,theta0,phi0,eps,N_traj,N_kicks):
    """
    Compute the average separation rate of local Quantum EMD over N_traj perturbations.

    Parameters
    ----------
    dhilbert : int
        Local Hilbert space dimension (2 for qubits).
    nqubit : int
        Number of qubits in the system.
    system_site : int
        Which qubit is the "system" (0-based index).
    U_F : ndarray
        Floquet operator (dim, dim).
    theta0, phi0 : float
        Initial angles for the reference state.
    eps : float
        Perturbation strength (radians).
    N_traj : int
        Number of random perturbation trajectories to average over.
    N_kicks : int
        Number of kicks (time steps) to evolve.

    Returns
    -------
    avg_rates : ndarray, shape (N_kicks,)
        Average separation rates at each kick.
    """
    # Reference initial state
    Psi_0 = Initial_state(nqubit, theta0, phi0)
    # --- Initialize the distance arrays ---
    D_Global= np.zeros((N_traj,N_kicks))
    D_local_GQS = np.zeros((N_traj,N_kicks))
    D_local_rho = np.zeros((N_traj,N_kicks))
    for avg in range(N_traj):
        # Perturbed initial state
        Psi_pert, _, _ = perturb_theta_phi_isotropic(nqubit, theta0, phi0, angle_sigma=eps)
        # Evolve and compute rates
        dg,dl,dr = Single_its_QKT(U_F,Psi_0,Psi_pert,dhilbert,nqubit,system_site=system_site,N_kicks=N_kicks)
        D_Global[avg,:] = dg
        D_local_GQS[avg,:] = dl
        D_local_rho[avg,:] = dr
    # Compute average rates
    ln_avg_dist_G = LLE_ln_avg_distance_separation(D_Global,N_kicks)
    ln_avg_dist_L = LLE_ln_avg_distance_separation(D_local_GQS,N_kicks)
    ln_avg_dist_R = LLE_ln_avg_distance_separation(D_local_rho,N_kicks)
    avg_D_G = np.mean(D_Global,axis=0)
    avg_D_L = np.mean(D_local_GQS,axis=0)
    avg_D_R = np.mean(D_local_rho,axis=0)
    datal = {}
    datal['ln_avg_dist_G'] = ln_avg_dist_G
    datal['ln_avg_dist_L'] = ln_avg_dist_L
    datal['ln  avg_dist_R'] = ln_avg_dist_R
    datal['avg_D_G'] = avg_D_G
    datal['avg_D_L'] = avg_D_L
    datal['avg_D_R'] = avg_D_R
    return datal

