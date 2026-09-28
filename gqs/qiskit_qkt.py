"""Qiskit kicked-top circuits and finite-shot conditional tomography.

The system is Qiskit qubit 0; environment labels follow qubits 1,...,L-1.
Each kick implements the Qaos Floquet map exactly up to global phase:
all ZZ pair terms commute. Simulation is ideal Aer sampling, not hardware.
Import this optional module explicitly; Qiskit is not a core dependency.
"""
import warnings
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def initial_state_circuit(nqubits, theta, phi):
    """
    Prepare on every qubit

        |psi> =
        cos(theta/2)|0>
        + exp(i phi) sin(theta/2)|1>

    up to a global phase.
    """
    circuit = QuantumCircuit(nqubits)
    for q in range(nqubits):
        circuit.ry(theta, q)
        circuit.rz(phi, q)
    return circuit

def qkt_kick_circuit(nqubits, kappa):
    """
    One Floquet step

        U_F =
        exp[-i kappa/(2j) Jz^2]
        exp[-i pi/2 Jy]

    where j = L/2 and L = nqubits.
    """
    if nqubits < 1:
        raise ValueError('nqubits must be positive')
    L = nqubits
    circuit = QuantumCircuit(L)
    for q in range(L):
        circuit.ry(np.pi / 2, q)
    rzz_angle = kappa / L
    for i in range(L):
        for j in range(i + 1, L):
            circuit.rzz(rzz_angle, i, j)
    return circuit

def tomography_circuit(base_circuit, system, environment, basis):
    """
    Add system tomography rotation and then measure
    system + environment.

    IMPORTANT:
    All measurements are terminal.
    """
    nqubits = base_circuit.num_qubits
    circuit = QuantumCircuit(nqubits, nqubits)
    circuit.compose(base_circuit, inplace=True)
    if basis == 'X':
        circuit.h(system)
    elif basis == 'Y':
        circuit.sdg(system)
        circuit.h(system)
    elif basis == 'Z':
        pass
    else:
        raise ValueError("basis must be 'X', 'Y', or 'Z'")
    circuit.measure(system, system)
    for q in environment:
        circuit.measure(q, q)
    return circuit

def build_GQS_circuits(theta0, phi0, kappa, max_kicks, nqubits):
    """
    Build

        X,Y,Z tomography circuits

    for

        kick = 1,...,max_kicks.

    Circuit ordering is

        kick1-X
        kick1-Y
        kick1-Z

        kick2-X
        kick2-Y
        kick2-Z

        ...
    """
    system = 0
    environment = list(range(1, nqubits))
    base = initial_state_circuit(nqubits, theta0, phi0)
    kick = qkt_kick_circuit(nqubits, kappa)
    circuits = []
    for n in range(1, max_kicks + 1):
        base.compose(kick, inplace=True)
        cx = tomography_circuit(base, system, environment, 'X')
        cx.name = f'kick_{n}_X'
        cy = tomography_circuit(base, system, environment, 'Y')
        cy.name = f'kick_{n}_Y'
        cz = tomography_circuit(base, system, environment, 'Z')
        cz.name = f'kick_{n}_Z'
        circuits.extend([cx, cy, cz])
    return circuits

def counts_to_joint(counts, system, environment):
    """
    Convert Qiskit counts into

        joint[e, s]

    where

        e = environment outcome
        s = system outcome (0 or 1)

    Example for 2 environment qubits:

        joint[0] -> environment 00
        joint[1] -> environment 01
        joint[2] -> environment 10
        joint[3] -> environment 11

    and

        joint[e,0] = number of system-0 outcomes
        joint[e,1] = number of system-1 outcomes.
    """
    dE = 2 ** len(environment)
    joint = np.zeros((dE, 2), dtype=np.int64)
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')[::-1]
        s = int(bits[system])
        env_bits = ''.join((bits[q] for q in environment))
        e = int(env_bits or '0', 2)
        joint[e, s] += count
    return joint

def conditional_expectation_from_counts(joint, env_state):
    """
    Compute

        <sigma>_e
          =
        P(0|e) - P(1|e)

    directly from counts.
    """
    n0 = joint[env_state, 0]
    n1 = joint[env_state, 1]
    total = n0 + n1
    if total == 0:
        return np.nan
    return (n0 - n1) / total

def reconstruct_GQS_from_counts(counts_x, counts_y, counts_z, system, environment, repetitions):
    """
    Reconstruct a pure-state ensemble estimate from X/Y/Z counts.

    Assumes ideal joint pure states and rank-one environment measurements.
    x/y/z retain the raw finite-shot Bloch vector. theta/phi describe its
    normalized direction, imposing purity; do not use this projection to
    characterize mixed conditional states from hardware noise.
    Missing basis samples or zero vectors are omitted with a warning;
    returned weights then need not sum to one. repetitions is retained for
    compatibility, but weights use actual totals from the counts.
    Environment labels follow the supplied environment order (first is MSB).
    """
    joint_x = counts_to_joint(counts_x, system, environment)
    joint_y = counts_to_joint(counts_y, system, environment)
    joint_z = counts_to_joint(counts_z, system, environment)
    dE = 2 ** len(environment)
    Q_S = []
    dropped_mass = 0.0
    env_counts_x = joint_x.sum(axis=1)
    env_counts_y = joint_y.sum(axis=1)
    env_counts_z = joint_z.sum(axis=1)
    env_counts = env_counts_x + env_counts_y + env_counts_z
    total_env_shots = env_counts.sum()
    if total_env_shots == 0:
        raise ValueError("Counts contain no shots")
    for e in range(dE):
        lam = env_counts[e] / total_env_shots
        if lam == 0:
            continue
        if env_counts_x[e] == 0 or env_counts_y[e] == 0 or env_counts_z[e] == 0:
            dropped_mass += lam
            continue
        x = conditional_expectation_from_counts(joint_x, e)
        y = conditional_expectation_from_counts(joint_y, e)
        z = conditional_expectation_from_counts(joint_z, e)
        bloch = np.array([x, y, z], dtype=float)
        r = np.linalg.norm(bloch)
        if r > 0:
            bloch_pure = bloch / r
        else:
            dropped_mass += lam
            continue
        x_p, y_p, z_p = bloch_pure
        z_p = np.clip(z_p, -1.0, 1.0)
        theta = np.arccos(z_p)
        phi = np.mod(np.arctan2(y_p, x_p), 2 * np.pi)
        env_bits = format(e, f'0{len(environment)}b')
        Q_S.append({'environment': env_bits, 'lambda': lam, 'x': x, 'y': y, 'z': z, 'theta': theta, 'phi': phi})
    if dropped_mass > 0:
        warnings.warn(f'Dropped probability mass {dropped_mass:.6g}: insufficient tomography data or undefined direction. Increase shots before using ensemble distances.', RuntimeWarning)
    return Q_S

def measure_GQS_trajectory(theta0, phi0, kappa, max_kicks, nqubits=3, repetitions=10000, seed=None):
    """
    Fast Qiskit version.

    Returns

        Q_S_all[0] = Q^S(1)
        Q_S_all[1] = Q^S(2)
        ...
        Q_S_all[N-1] = Q^S(N)

    Improvements over previous implementation:

    1. No memory=True
    2. No individual-shot Python loop
    3. Terminal measurements only
    4. All tomography circuits built together
    5. All circuits transpiled together
    6. All circuits submitted in ONE Aer job
    """
    system = 0
    environment = list(range(1, nqubits))
    if max_kicks < 1 or repetitions < 1:
        raise ValueError('max_kicks and repetitions must be positive')
    simulator = AerSimulator(method='statevector', max_parallel_experiments=0)
    circuits = build_GQS_circuits(theta0=theta0, phi0=phi0, kappa=kappa, max_kicks=max_kicks, nqubits=nqubits)
    compiled = transpile(circuits, simulator, optimization_level=0)
    job = simulator.run(compiled, shots=repetitions, seed_simulator=seed)
    result = job.result()
    Q_S_all = []
    for n in range(max_kicks):
        counts_x = result.get_counts(3 * n)
        counts_y = result.get_counts(3 * n + 1)
        counts_z = result.get_counts(3 * n + 2)
        Q_n = reconstruct_GQS_from_counts(counts_x=counts_x, counts_y=counts_y, counts_z=counts_z, system=system, environment=environment, repetitions=repetitions)
        Q_S_all.append(Q_n)
    return Q_S_all

def show_single_experimental_step(nqubits=3, kappa=0.5, theta0=np.pi / 2 + 0.5, phi0=np.pi / 2, basis='X', output='text'):
    """
    Build one complete experimental GQS circuit:

        initial state
            ->
        one QKT kick
            ->
        system tomography rotation
            ->
        measure system + environment

    Parameters
    ----------
    nqubits : int
        Total number of qubits.

    kappa : float
        QKT interaction strength.

    theta0, phi0 : float
        Initial spin-coherent state angles.

    basis : {"X", "Y", "Z"}
        Tomography basis for the system qubit.

    output : {"text", "mpl"}
        Circuit drawing format.

    Returns
    -------
    QuantumCircuit or matplotlib.figure.Figure
        Circuit for text output; drawing figure for mpl output.
    """
    system = 0
    environment = list(range(1, nqubits))
    circuit = initial_state_circuit(nqubits=nqubits, theta=theta0, phi=phi0)
    circuit.barrier()
    kick = qkt_kick_circuit(nqubits=nqubits, kappa=kappa)
    circuit.compose(kick, inplace=True)
    circuit.barrier()
    from qiskit import ClassicalRegister
    c = ClassicalRegister(nqubits, 'c')
    circuit.add_register(c)
    if basis == 'X':
        circuit.h(system)
    elif basis == 'Y':
        circuit.sdg(system)
        circuit.h(system)
    elif basis == 'Z':
        pass
    else:
        raise ValueError("basis must be 'X', 'Y', or 'Z'")
    circuit.barrier()
    circuit.measure(system, c[system])
    for q in environment:
        circuit.measure(q, c[q])
    if output == 'mpl':
        return circuit.draw(output='mpl', fold=-1)
    else:
        print(circuit.draw(output='text', fold=-1))
        return circuit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
colors_pink_red = ['#ffb3b3', '#ff4d4d', '#b30000', '#6e0707']
cmap_pink_red = LinearSegmentedColormap.from_list('pink_to_red', colors_pink_red)

def plot_QS_history(Q_S, kick, past_marker_size=30, current_marker_size=120, past_alpha=0.2, vmin=0.0, vmax=1.0):
    """
    Plot all GQS points up to a given kick.

    Past kicks:
        smaller markers
        reduced alpha

    Current kick:
        larger markers
        alpha = 1

    Q_S[0] = Q^S(1)
    Q_S[1] = Q^S(2)
    ...
    """
    if kick < 1 or kick > len(Q_S):
        raise ValueError(f'kick must be between 1 and {len(Q_S)}')
    fig, ax = plt.subplots(figsize=(7, 5))
    norm = Normalize(vmin=vmin, vmax=vmax)
    for n in range(kick - 1):
        Q_n = Q_S[n]
        phi = np.array([state['phi'] for state in Q_n])
        theta = np.array([state['theta'] for state in Q_n])
        lam = np.array([state['lambda'] for state in Q_n])
        ax.scatter(phi, theta, c=lam, cmap=cmap_pink_red, norm=norm, s=past_marker_size, alpha=past_alpha, edgecolors='none')
    Q_current = Q_S[kick - 1]
    phi = np.array([state['phi'] for state in Q_current])
    theta = np.array([state['theta'] for state in Q_current])
    lam = np.array([state['lambda'] for state in Q_current])
    scatter = ax.scatter(phi, theta, c=lam, cmap=cmap_pink_red, norm=norm, s=current_marker_size, alpha=1.0, edgecolors='black', linewidths=0.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
    ax.set_yticks([0, np.pi / 2, np.pi])
    ax.set_yticklabels(['$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_xlabel('$\\phi$')
    ax.set_ylabel('$\\theta$')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('$\\lambda_j$')
    plt.tight_layout()
    return (fig, ax)

def ensemble_to_arrays(ensemble, *, renormalize=False):
    """Convert pure-direction estimates to Qaos (chi_S, lambda_E) arrays.

    Reject missing mass unless renormalize=True explicitly conditions on
    retained outcomes. This adapter inherits the pure-state assumption.
    It does not restore omitted outcomes or their probability mass.
    """
    if not ensemble:
        raise ValueError("No reconstructed states; increase repetitions")
    theta = np.array([s["theta"] for s in ensemble])
    phi = np.array([s["phi"] for s in ensemble])
    lam = np.array([s["lambda"] for s in ensemble])
    if not np.isclose(lam.sum(), 1.0):
        if not renormalize:
            raise ValueError("Incomplete ensemble mass; increase repetitions or explicitly set renormalize=True")
        lam = lam / lam.sum()
    chi = np.column_stack((np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)))
    return chi, lam
