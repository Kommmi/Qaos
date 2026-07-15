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
from .stats import preprocess_series, dominant_period_cepstrum,split_into_periods,average_per_period
from .stats import period_averaged_variance, directional_increment_rates_per_period, directional_asymmetry
from .stats import period_statistics_summary

def Compare_distances_All(U,Psi_0,Psi_p,d_hilbert=2,n_chain=3,system_site=0,N_kicks=200,renormalize=False,show_plt=False):
    """returns the Earth Mover's Distance between perturbed and unperturbed distribution
         as a function of time for a single iteration

    Parameters  
    ----------
    U : ndarray
        Floquet operator (dim, dim).
    Psi_0 : ndarray
        Initial unperturbed state vector.
    Psi_p : ndarray
        Initial perturbed state vector.
    d_hilbert : int
        Local Hilbert space dimension (2 for qubits).
    n_chain : int
        Number of qubits in the chain.
    system_site : int
        Which qubit is the "system" (0-based index).
    N_kicks : int
        Number of kicks (time steps) to evolve.
    renormalize : bool
        Whether to renormalize the states after each kick.
    show_plt : bool
        Show the plot of distances vs time.
    

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
        ml1, sl1, _ = plt.stem(range(N_kicks), D_global - 1, linefmt='-', markerfmt='o', basefmt=' ', label='Global State Distance')
        plt.setp(sl1, color="#FF9747")
        plt.setp(ml1, color='#FF9747', markerfacecolor='#FF9747', markeredgecolor='#FF9747')
        ml1.set_markersize(2)

        ml2, sl2, _ = plt.stem(range(N_kicks), D_local - 1, linefmt='-', markerfmt='^', basefmt=' ', label='Quantum EMD')
        plt.setp(sl2, color='#4682B4')
        plt.setp(ml2, color='#4682B4', markerfacecolor='#4682B4', markeredgecolor='#4682B4')
        ml2.set_markersize(2)

        ml3, sl3, _ = plt.stem(range(N_kicks), D_local_rho - 1, linefmt='-', markerfmt='s', basefmt=' ', label='Bures Distance')
        plt.setp(sl3, color='#143c6c')
        plt.setp(ml3, color='#143c6c', markerfacecolor='#143c6c', markeredgecolor='#143c6c')
        ml3.set_markersize(2)
        plt.xlabel(r'$t_n$ - Number of Floquet Kicks',fontsize=12)
        plt.ylabel(r'$\frac{W_1(t_n)}{W_1(0)} - 1$',fontsize=12)
        plt.title('Comparison of Global and Local Distances over Time',fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.ylim([-1, 1])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return D_global,D_local,D_local_rho

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


def Single_its_GQS(U,Psi_0,Psi_p,d_hilbert,n_chain,system_site,N_kicks,renormalize=False):
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

    # 1. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
    Dg0 = Psi_Dist(Psi_0, Psi_p)

    # 2a. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
    chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
    chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)

    # 3. Compute the Quantum EMD between the two GQS
    Dl0 = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep)

    #4.Evolve the two states for N_kicks
    for i in range(N_kicks):
        # 5. Compute the Fubini-Study distance between the two global states Psi_0 and Psi_p
        D_global[i] = Psi_Dist(Psi_0, Psi_p) / Dg0
        # 6a. Compute the Geometric Quantum States of the two global states Psi_0 and Psi_p
        chi_s0,lambda_e0 = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_0)
        chi_sp,lambda_ep = Reduced_state_single_site(d_hilbert,n_chain,system_site,Psi_SE=Psi_p)
        # 7a. Compute the Quantum EMD between the two GQS
        D_local_GQS[i] = Quantum_EMD(chi_s0,lambda_e0,chi_sp,lambda_ep) / Dl0
        # 8. Evolve the two states
        Psi_0 = U @ Psi_0
        Psi_p = U @ Psi_p
        if renormalize:
            Psi_0 = Psi_0 / np.linalg.norm(Psi_0)
            Psi_p = Psi_p / np.linalg.norm(Psi_p)

    return D_global, D_local_GQS

def Gamma_calculator(U_F,dhilbert=2,nqubit=3,system_site=0,theta0=np.pi/2,phi0=np.pi/2,eps=0.2,N_traj=100,N_kicks=200,min_period=10,max_period=100,show_plot=False):
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
    min_period : int
        Minimum period for averaging.
    max_period : int
        Maximum period for averaging.
    show_plot : bool
        Whether to show the plot of average separation rates.

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
    for avg in range(N_traj):
        # Perturbed initial state
        Psi_pert, _, _ = perturb_theta_phi_isotropic(nqubit, theta0, phi0, angle_sigma=eps)
        DG0 = Psi_Dist(Psi_0, Psi_pert)
        if DG0 == 0:
            #print("Warning: Perturbation did not change the state. Skipping this trajectory.")
            Psi_pert, _, _ = perturb_theta_phi_isotropic(nqubit, theta0, phi0, angle_sigma=eps)
        # Evolve and compute rates
        dg,dl = Single_its_GQS(U_F,Psi_0,Psi_pert,dhilbert,nqubit,system_site=system_site,N_kicks=N_kicks)
        D_Global[avg,:] = dg
        D_local_GQS[avg,:] = dl
    # Compute average rates
    ln_avg_dist_G = LLE_ln_avg_distance_separation(D_Global,N_kicks)
    ln_avg_dist_L = LLE_ln_avg_distance_separation(D_local_GQS,N_kicks)
    avg_D_G = np.mean(D_Global,axis=0)
    avg_D_L = np.mean(D_local_GQS,axis=0)
    datal = {}
    datal['ln_avg_dist_G'] = ln_avg_dist_G
    datal['ln_avg_dist_L'] = ln_avg_dist_L
    datal['avg_D_G'] = avg_D_G
    datal['avg_D_L'] = avg_D_L
    #Compute the separation rate - Distinguishability growth rate
    periodl,_,_ = dominant_period_cepstrum(ln_avg_dist_L,min_period=min_period,max_period=max_period)
    avg_T, per_period_avgs = average_per_period(ln_avg_dist_L, periodl)
    datal['periodl'] = periodl
    datal['Gamma'] = avg_T
    datal['per_period_avgs'] = per_period_avgs
    if show_plot:
        plt.figure(figsize=(10,3))
        ml2, sl2, _ = plt.stem(
            range(N_kicks),
            ln_avg_dist_L,
            linefmt='-',
            markerfmt='^',
            basefmt=' ',
            label=rf'$\Gamma$ = {avg_T:.2f}'
        )
        plt.setp(sl2, color='#4682B4')
        plt.setp(ml2, color='#4682B4', markerfacecolor='#4682B4', markeredgecolor='#4682B4')
        ml2.set_markersize(2)
        plt.xlabel(r'$t_n$ - Number of Floquet Kicks',fontsize=12)
        plt.ylabel(r'$\langle \log \frac{W_1(t_n)}{W_1(0)} \rangle$',fontsize=12)
        plt.title('Local Earth Mover\'s Distances b/w GQSs over Time',fontsize=12)
        plt.legend(fontsize=14, loc='upper left')
        plt.ylim([-1, 1])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return avg_T, datal
