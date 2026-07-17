# State-Based Diagnostics of Quantum Dynamical Complexity

This repository implements a **state-based geometric framework** for quantifying dynamical complexity in interacting quantum systems.

![Classical and quantum kicked-top dynamics](imagesREADME/CKT_QKT.jpg)

The framework extends the classical ideas of **sensitivity to initial conditions** and **phase-space exploration** to quantum systems. Subsystem dynamics are represented as probability measures on projective Hilbert space, referred to as **geometric quantum states (GQSs)**.

---

## 🧠 Why Geometric Quantum States? — Notebook 1

Consider a quantum system $S$ coupled to an environment $E$. The global state can be written as

$$|\Psi_{SE}(t)\rangle =\sum_{k=1}^{d_S}\sum_{j=1}^{d_E}\psi_{kj}(t) |s_k\rangle \otimes |e_j\rangle$$

Conditioning on the environment basis $\{|e_j\rangle\}$ gives the decomposition

$$|\Psi_{SE}(t)\rangle=\sum_{j=1}^{d_E}
\sqrt{\lambda_j^E(t)}
\,|\chi_j^S(t)\rangle |e_j\rangle ,
$$

where

$$ \lambda_j^E(t) =
\sum_{k=1}^{d_S}
|\psi_{kj}(t)|^2
$$

and, for $\lambda_j^E(t)>0$,

$$|\chi_j^S(t)\rangle=
\frac{1}{\sqrt{\lambda_j^E(t)}}
\sum_{k=1}^{d_S}
\psi_{kj}(t)|s_k\rangle .
$$

### Reduced density matrix

The subsystem state is conventionally represented by the reduced density matrix

$$ \rho_S(t)=
\sum_{j=1}^{d_E}
\lambda_j^E(t)
|\chi_j^S(t)\rangle
\langle\chi_j^S(t)| .
$$

The reduced density matrix:

- reproduces all observable statistics of the subsystem;
- encodes the underlying ensemble structure only implicitly;
- does not distinguish between different pure-state ensembles that produce the same density matrix.

### Geometric quantum state

The corresponding geometric quantum state is the probability measure

$$ Q^S(Z,t)=
\sum_{j=1}^{d_E}
\lambda_j^E(t)
\delta\left(Z-\mathbf{Z}_j^S(t)\right)
\in
\mathcal{P}\left(\mathbb{C}P^{d_S-1}\right),
$$

where $\mathbf{Z}_j^S(t)$ is the point in projective Hilbert space associated with the conditional pure state $|\chi_j^S(t)\rangle$.

A GQS:

- retains the full distribution of conditional pure states on projective Hilbert space;
- distinguishes geometrically different ensembles that correspond to the same density matrix;
- provides a natural setting for comparing subsystem states using optimal transport.

> **Basis dependence:** The environment-conditioned GQS depends on the chosen environment basis. The calculations in this repository use conditioning in the computational basis.

---

## 📐 Distance Measures — Notebook 2

### Fubini–Study distance

The **Fubini–Study distance** measures the geometric separation between pure states in projective Hilbert space.

### Wasserstein distance

The **Wasserstein distance** is the primary distance used in this framework. It quantifies the minimum transport cost required to transform one GQS probability measure into another, using the Fubini–Study distance as the underlying ground metric.

<img src="imagesREADME/bloch_ot_tables.jpg" alt="Optimal transport between geometric quantum states" width="70%">

Operator-based distances between reduced density matrices can also be computed for comparison.

---

## 🔬 Diagnostics of Quantum Complexity - Notebook-3,4

We study how interactions among qubits affect the local dynamics of a selected subsystem.

Wasserstein geometry directly captures how environment-conditioned probability measures **separate, deform, and spread** across the quantum state manifold. This provides a state-based perspective on quantum dynamical complexity that complements operator-based diagnostics such as out-of-time-order correlators and Loschmidt echoes.

![GQS dynamics for different interaction strengths](imagesREADME/three_kappa_gqs.gif)

<img src="imagesREADME/three_kappa_gqs_distance.gif" alt="GQS distinguishability for different interaction strengths" width="90%">

The framework introduces two complementary diagnostics:

- **Distinguishability measure $\Gamma$:** quantifies the growth or decay of the Wasserstein distance between initially nearby GQS measures. It plays a role analogous to a finite-time Lyapunov-type sensitivity measure for probability distributions.

- **State-Space Coverage Index (SSCI):** quantifies how broadly the subsystem explores projective Hilbert space over time.

These diagnostics provide a two-dimensional characterization of subsystem dynamics by separating:

1. **sensitivity to initial conditions**, measured by $\Gamma$; and
2. **long-time state-space exploration**, measured by the SSCI.

---

## ⚙️ Computational Workflow

1. Initialize a global spin-coherent product state.
2. Apply a small perturbation to generate a nearby initial state.
3. Evolve both states under a global interacting model, such as the quantum kicked top.
4. Construct the environment-conditioned subsystem states.
5. Represent the subsystem dynamics as GQS probability measures.
6. Compute Wasserstein distances between the reference and perturbed GQSs.
7. Estimate:
   - $\Gamma$, describing sensitivity to perturbations;
   - the SSCI, describing state-space exploration.

---

## 📊 What This Repository Enables

This repository provides tools for:

- quantifying interaction-induced quantum dynamical complexity;
- visualizing local subsystem dynamics on projective Hilbert space;
- comparing GQS-based and density-matrix-based descriptions;
- studying the dependence of subsystem dynamics on:
  - interaction strength $\kappa$;
  - initial-state coordinates $(\theta,\phi)$;
  - environment size $L_E$;
  - integer- and half-integer-spin system sizes.

![GQS dynamics for increasing odd system sizes](imagesREADME/three_LE_gqs_odd.gif)

![GQS dynamics for increasing even system sizes](imagesREADME/three_LE_gqs_even.gif)

---

## Install

### Option A: editable install (recommended for development)
```bash
pip install -e .
```

### Option B: install pinned dependencies only
```bash
pip install -r requirements.txt
```

## Package layout
- `gqs/operators.py`: spin operators
- `gqs/states.py`: initial states + reduced/conditional states
- `gqs/dynamics.py`: kicked-top Hamiltonian + Floquet operator
- `gqs/gqs.py`: GQS / Bloch utilities + visualizations
- `gqs/distances.py`: Fubini–Study + Wasserstein (OT) distances
- `gqs/entropy.py`: entropy/purity utilities
- `gqs/perturbations.py`: (theta,phi) perturbation helpers
- `gqs/gamma.py`: Gamma / separation-rate computations
- `gqs/plotting.py`: plotting helpers

