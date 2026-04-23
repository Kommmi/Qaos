# State-Based Diagnostics of Quantum Dynamical Complexity

This repository implements a **state-based, geometric framework** for quantifying dynamical complexity in interacting quantum systems.

 ![Quantum-Dynamical-Systems](imagesREADME/CKT_QKT.jpg)

The approach extends classical notions of **trajectory sensitivity** and **phase-space exploration** to quantum systems by representing subsystem dynamics as **probability measures over projective Hilbert space**, referred to as **geometric quantum states (GQS)**.

---

## ✨ Core Idea

We study how **small differences in global quantum preparations** manifest at the level of **local subsystem dynamics** under interactions.

Rather than representing reduced states solely as density matrices, we construct **environment-conditioned ensembles of pure states**, yielding a geometric representation:
$$Q^S(Z,t) = \sum_j \lambda_j(t)\,\delta(Z - Z_j^S(t)).$$

Distances between these ensembles are computed using **Wasserstein (optimal transport) geometry**, enabling a direct characterization of how probability mass **deforms and spreads** on the quantum state manifold.

---

## 🔬 Diagnostics of Quantum Complexity

This framework introduces two complementary diagnostics:

### • Distinguishability Growth Rate (Γ)
Measures the average exponential growth of distances between nearby GQS ensembles:
- quantifies **local sensitivity**
- serves as a quantum analogue of a Lyapunov exponent

### • State-Space Coverage Index (SSCI)
Measures how widely the subsystem explores its state space over time:
- quantifies **global spreading**
- captures long-time geometric exploration

Together, these define a **two-dimensional characterization of complexity**:
> sensitivity (Γ) vs spread (SSCI)

---

## 🧠 Why Geometric Quantum States?

Standard reduced-state measures (entropy, trace distance) compress dynamics into a single operator.

In contrast, GQS:
- retain the **distribution of pure states** on projective Hilbert space,
- distinguish geometrically distinct ensembles with identical density matrices,
- provide a natural connection to **optimal transport geometry**.

This enables a **state-based perspective on quantum complexity**, complementing operator-based diagnostics such as OTOCs and Loschmidt echoes.

---

## 📐 Distance Measures

- **Fubini–Study distance**  
  Geometry of pure states on projective Hilbert space

- **Wasserstein distance (primary tool)**  
  Quantifies transport of probability mass between GQS ensembles

- *(Optional)* Operator-based distances for comparison

---

## ⚙️ Computational Workflow

1. Initialize a global spin-coherent product state  
2. Apply a small perturbation to generate nearby initial conditions  
3. Evolve under a global interacting Hamiltonian (e.g. quantum kicked top)  
4. Construct environment-conditioned subsystem states  
5. Build geometric quantum state ensembles (GQS)  
6. Compute Wasserstein distances between ensembles  
7. Estimate:
   - **Γ (sensitivity)**
   - **SSCI (state-space exploration)**

---

## 📊 What This Repository Enables

- Quantification of **interaction-induced complexity**
- Visualization of **deformation of subsystem state space**
- Analysis of:
  - interaction strength (κ)
  - initial state dependence (θ, φ)
  - environment size (Lₑ)
  - even–odd system-size effects

---

## 📌 Key Perspective

Complexity in quantum systems emerges not only through operator growth, but through the **geometric redistribution of probability mass** over the space of pure states.

This framework makes that redistribution **directly measurable and interpretable**.

---

%## 🔗 Links

%- 📄 Paper: *[add arXiv link here]*  
%- 💻 Repository: *[this repository]*  


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

