#!/usr/bin/env python3
"""
Computational View of Qubits - Comprehensive Python Demonstrations

This script demonstrates quantum gates, circuits, and computational operations on qubits.
It covers single-qubit gates, gate properties, circuit construction, and quantum algorithms.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

np.random.seed(42)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)


def plot_quantum_gates():
    """
    Visualize quantum gates and their effects on the Bloch sphere.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Quantum Gates and Their Effects', fontsize=16, fontweight='bold')
    
    gates = [
        ('X Gate', X, 'red'),
        ('Y Gate', Y, 'green'),
        ('Z Gate', Z, 'blue'),
        ('H Gate', H, 'purple'),
        ('S Gate', S, 'orange'),
        ('T Gate', T, 'cyan')
    ]
    
    for idx, (gate_name, gate, color) in enumerate(gates):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
        
        ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'k-', alpha=0.3, linewidth=1)
        
        state_0 = gate @ ket0
        state_plus = gate @ (H @ ket0)
        
        def state_to_bloch(state):
            alpha = state[0, 0]
            beta = state[1, 0]
            theta = 2 * np.arccos(np.abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return x, y, z
        
        x0, y0, z0 = state_to_bloch(ket0)
        x1, y1, z1 = state_to_bloch(state_0)
        
        ax.quiver(0, 0, 0, x0, y0, z0, color='blue', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
        ax.quiver(0, 0, 0, x1, y1, z1, color=color, arrow_length_ratio=0.15, linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(gate_name)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('quantum_gates.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_gates.png")
    plt.show()


def plot_gate_matrices():
    """
    Visualize quantum gate matrices and their properties.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum Gate Matrices', fontsize=16, fontweight='bold')
    
    gates = [
        ('Pauli-X (NOT)', X),
        ('Pauli-Y', Y),
        ('Pauli-Z', Z),
        ('Hadamard', H),
        ('Phase (S)', S),
        ('T Gate', T)
    ]
    
    for idx, (gate_name, gate) in enumerate(gates):
        ax = axes[idx // 3, idx % 3]
        
        real_part = np.real(gate)
        imag_part = np.imag(gate)
        
        combined = np.zeros((2, 4))
        combined[:, :2] = real_part
        combined[:, 2:] = imag_part
        
        im = ax.imshow(combined, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        
        for i in range(2):
            for j in range(2):
                val_real = real_part[i, j]
                text_color = 'white' if abs(val_real) > 0.5 else 'black'
                ax.text(j, i, f'{val_real:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
                
                val_imag = imag_part[i, j]
                text_color = 'white' if abs(val_imag) > 0.5 else 'black'
                if abs(val_imag) > 0.01:
                    ax.text(j+2, i, f'{val_imag:.2f}i', ha='center', va='center', 
                           color=text_color, fontsize=10, fontweight='bold')
                else:
                    ax.text(j+2, i, '0', ha='center', va='center', 
                           color='black', fontsize=10)
        
        ax.set_xticks([0.5, 2.5])
        ax.set_xticklabels(['Real', 'Imaginary'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['|0⟩', '|1⟩'])
        ax.set_title(gate_name)
        
        ax.axvline(x=1.5, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('gate_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: gate_matrices.png")
    plt.show()


def plot_gate_properties():
    """
    Visualize properties of quantum gates (unitarity, hermiticity, etc.).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Gate Properties', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    gates_list = [('X', X), ('Y', Y), ('Z', Z), ('H', H), ('S', S), ('T', T)]
    gate_names = [name for name, _ in gates_list]
    
    unitarity_errors = []
    for name, gate in gates_list:
        product = gate.conj().T @ gate
        error = np.linalg.norm(product - I)
        unitarity_errors.append(error)
    
    bars = ax.bar(gate_names, unitarity_errors, color='lightblue', edgecolor='black')
    ax.set_ylabel('||U†U - I||')
    ax.set_title('Unitarity Check (should be ~0)')
    ax.set_ylim(0, max(unitarity_errors) * 1.2 if max(unitarity_errors) > 0 else 0.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1e-10, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.legend()
    
    for bar, val in zip(bars, unitarity_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    
    ax = axes[0, 1]
    
    hermiticity_errors = []
    for name, gate in gates_list:
        error = np.linalg.norm(gate.conj().T - gate)
        hermiticity_errors.append(error)
    
    bars = ax.bar(gate_names, hermiticity_errors, color='lightgreen', edgecolor='black')
    ax.set_ylabel('||U† - U||')
    ax.set_title('Hermiticity Check (Pauli gates should be ~0)')
    ax.set_ylim(0, max(hermiticity_errors) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, hermiticity_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax = axes[1, 0]
    
    for name, gate in gates_list:
        eigenvalues = np.linalg.eigvals(gate)
        ax.plot(np.real(eigenvalues), np.imag(eigenvalues), 'o', 
               markersize=10, label=name, alpha=0.7)
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, alpha=0.3, label='Unit circle')
    
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('Eigenvalues (should lie on unit circle)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    ax = axes[1, 1]
    
    pauli_gates = [('X', X), ('Y', Y), ('Z', Z)]
    n = len(pauli_gates)
    commutator_matrix = np.zeros((n, n))
    
    for i, (name1, gate1) in enumerate(pauli_gates):
        for j, (name2, gate2) in enumerate(pauli_gates):
            commutator = gate1 @ gate2 - gate2 @ gate1
            commutator_matrix[i, j] = np.linalg.norm(commutator)
    
    im = ax.imshow(commutator_matrix, cmap='YlOrRd', vmin=0, vmax=np.max(commutator_matrix))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([name for name, _ in pauli_gates])
    ax.set_yticklabels([name for name, _ in pauli_gates])
    ax.set_title('Commutator Norms ||[A,B]||')
    
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{commutator_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=12)
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('gate_properties.png', dpi=300, bbox_inches='tight')
    print("Saved: gate_properties.png")
    plt.show()


def plot_quantum_circuits():
    """
    Visualize quantum circuit construction and execution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Circuits', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title('Circuit: H → X → H')
    
    ax.plot([0, 10], [1, 1], 'k-', linewidth=2)
    
    gates_x = [2, 5, 8]
    gate_labels = ['H', 'X', 'H']
    for x, label in zip(gates_x, gate_labels):
        rect = Rectangle((x-0.3, 0.7), 0.6, 0.6, facecolor='lightblue', 
                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 1, label, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(0.5, 1.5, '|0⟩', fontsize=12)
    ax.text(3.5, 1.5, '|+⟩', fontsize=12)
    ax.text(6.5, 1.5, '|-⟩', fontsize=12)
    ax.text(9.5, 1.5, '|1⟩', fontsize=12)
    
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Bell State: (|00⟩ + |11⟩)/√2')
    
    ax.plot([0, 10], [2, 2], 'k-', linewidth=2)
    ax.plot([0, 10], [1, 1], 'k-', linewidth=2)
    
    rect = Rectangle((2-0.3, 1.7), 0.6, 0.6, facecolor='lightblue', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 2, 'H', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.plot([5, 5], [1, 2], 'k-', linewidth=2)
    circle = Circle((5, 2), 0.15, facecolor='black')
    ax.add_patch(circle)
    circle2 = Circle((5, 1), 0.3, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle2)
    ax.plot([4.8, 5.2], [1, 1], 'k-', linewidth=2)
    ax.plot([5, 5], [0.8, 1.2], 'k-', linewidth=2)
    
    ax.text(0.5, 2, 'q₀:', fontsize=12)
    ax.text(0.5, 1, 'q₁:', fontsize=12)
    ax.text(9, 2, '(|00⟩+|11⟩)/√2', fontsize=10)
    
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title('Decomposition: Y = iXZ')
    
    ax.plot([0, 10], [1, 1], 'k-', linewidth=2)
    
    rect = Rectangle((2-0.3, 0.7), 0.6, 0.6, facecolor='lightgreen', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 1, 'Y', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(4, 1, '=', fontsize=20, ha='center', va='center')
    
    rect = Rectangle((6-0.3, 0.7), 0.6, 0.6, facecolor='lightcoral', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 1, 'X', ha='center', va='center', fontsize=14, fontweight='bold')
    
    rect = Rectangle((8-0.3, 0.7), 0.6, 0.6, facecolor='lightblue', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(8, 1, 'Z', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(5, 1.5, 'i ×', fontsize=12)
    
    ax = axes[1, 1]
    
    depths = np.arange(1, 21)
    fidelity_ideal = np.ones_like(depths, dtype=float)
    fidelity_noisy = np.exp(-depths * 0.01)  # 1% error per gate
    fidelity_very_noisy = np.exp(-depths * 0.05)  # 5% error per gate
    
    ax.plot(depths, fidelity_ideal, 'g-', linewidth=2, label='Ideal', marker='o')
    ax.plot(depths, fidelity_noisy, 'b-', linewidth=2, label='1% error/gate', marker='s')
    ax.plot(depths, fidelity_very_noisy, 'r-', linewidth=2, label='5% error/gate', marker='^')
    
    ax.set_xlabel('Circuit Depth')
    ax.set_ylabel('Fidelity')
    ax.set_title('Fidelity vs Circuit Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(19, 0.52, 'Threshold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('quantum_circuits.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_circuits.png")
    plt.show()


def plot_gate_sequences():
    """
    Visualize the effect of gate sequences on qubit states.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gate Sequences and State Evolution', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    n_steps = 50
    theta = np.linspace(0, 2*np.pi, n_steps)
    
    prob_0 = []
    prob_1 = []
    state = ket0.copy()
    
    for i in range(n_steps):
        angle = 2 * np.pi / n_steps
        Rx = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                      [-1j*np.sin(angle/2), np.cos(angle/2)]], dtype=complex)
        state = Rx @ state
        
        prob_0.append(np.abs(state[0, 0])**2)
        prob_1.append(np.abs(state[1, 0])**2)
    
    ax.plot(theta, prob_0, 'b-', linewidth=2, label='P(|0⟩)')
    ax.plot(theta, prob_1, 'r-', linewidth=2, label='P(|1⟩)')
    ax.set_xlabel('Rotation Angle (radians)')
    ax.set_ylabel('Probability')
    ax.set_title('Rabi Oscillations: Continuous X Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    ax.axvline(x=np.pi, color='green', linestyle='--', alpha=0.7)
    ax.text(np.pi, 1.05, 'π', fontsize=12, ha='center')
    ax.axvline(x=2*np.pi, color='green', linestyle='--', alpha=0.7)
    ax.text(2*np.pi, 1.05, '2π', fontsize=12, ha='center')
    
    ax = axes[0, 1]
    
    n_iterations = 10
    states = [ket0]
    state = ket0.copy()
    
    for i in range(n_iterations):
        state = H @ state
        states.append(state.copy())
    
    iterations = np.arange(n_iterations + 1)
    probs_0 = [np.abs(s[0, 0])**2 for s in states]
    probs_1 = [np.abs(s[1, 0])**2 for s in states]
    
    ax.plot(iterations, probs_0, 'bo-', linewidth=2, markersize=8, label='P(|0⟩)')
    ax.plot(iterations, probs_1, 'ro-', linewidth=2, markersize=8, label='P(|1⟩)')
    ax.set_xlabel('Number of H Gates')
    ax.set_ylabel('Probability')
    ax.set_title('Repeated Hadamard Gates (H² = I)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    ax = axes[1, 0]
    
    n_steps = 20
    phases = []
    state = ket1.copy()
    
    for i in range(n_steps):
        state = T @ state
        phase = np.angle(state[1, 0])
        phases.append(phase)
    
    steps = np.arange(1, n_steps + 1)
    expected_phases = steps * np.pi / 4
    
    ax.plot(steps, phases, 'bo-', linewidth=2, markersize=6, label='Actual phase')
    ax.plot(steps, expected_phases, 'r--', linewidth=2, label='Expected (nπ/4)')
    ax.set_xlabel('Number of T Gates')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Phase Accumulation with T Gates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i in range(1, 9):
        ax.axhline(y=i*np.pi/4, color='gray', linestyle=':', alpha=0.3)
    
    ax = axes[1, 1]
    
    n_gates = 50
    gate_pool = [X, Y, Z, H, S, T]
    gate_names = ['X', 'Y', 'Z', 'H', 'S', 'T']
    
    state = ket0.copy()
    fidelities = [1.0]  # Fidelity with initial state
    
    for i in range(n_gates):
        gate = np.random.choice(gate_pool)
        state = gate @ state
        fidelity = np.abs(np.vdot(ket0, state))**2
        fidelities.append(fidelity)
    
    ax.plot(range(n_gates + 1), fidelities, 'b-', linewidth=2)
    ax.set_xlabel('Number of Random Gates')
    ax.set_ylabel('Fidelity with |0⟩')
    ax.set_title('Random Circuit: Fidelity Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('gate_sequences.png', dpi=300, bbox_inches='tight')
    print("Saved: gate_sequences.png")
    plt.show()


def plot_universal_gates():
    """
    Visualize universal gate sets and gate decomposition.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Universal Gate Sets', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    gate_sets = ['Clifford+T\n{H,S,T,CNOT}', 'Rotation\n{Rx,Ry,Rz,CNOT}', 
                 'U3+CNOT\n{U3,CNOT}', 'Toffoli\n{H,T,Toffoli}']
    completeness = [1.0, 1.0, 1.0, 1.0]
    efficiency = [0.8, 0.9, 0.95, 0.7]
    hardware_support = [0.9, 0.8, 0.85, 0.6]
    
    x = np.arange(len(gate_sets))
    width = 0.25
    
    bars1 = ax.bar(x - width, completeness, width, label='Completeness', 
                   color='lightblue', alpha=0.8)
    bars2 = ax.bar(x, efficiency, width, label='Efficiency', 
                   color='lightgreen', alpha=0.8)
    bars3 = ax.bar(x + width, hardware_support, width, label='Hardware Support', 
                   color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Universal Gate Set Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(gate_sets)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    
    epsilon = np.logspace(-6, -1, 50)
    c = 2  # Solovay-Kitaev constant
    
    gate_count_SK = np.log(1/epsilon)**c
    gate_count_naive = 1/epsilon
    
    ax.loglog(epsilon, gate_count_SK, 'b-', linewidth=2, 
             label='Solovay-Kitaev: O(log^c(1/ε))')
    ax.loglog(epsilon, gate_count_naive, 'r--', linewidth=2, 
             label='Naive: O(1/ε)')
    
    ax.set_xlabel('Approximation Error (ε)')
    ax.set_ylabel('Number of Gates')
    ax.set_title('Solovay-Kitaev Theorem')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()
    
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Gate Decomposition Example')
    
    rect = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1", 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 8.5, 'U', ha='center', va='center', fontsize=14, fontweight='bold')
    
    for i, (x, label) in enumerate([(2, 'Rz(φ)'), (5, 'Ry(θ)'), (8, 'Rz(λ)')]):
        rect = FancyBboxPatch((x-0.8, 6), 1.6, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 6.4, label, ha='center', va='center', fontsize=10)
        ax.plot([5, x], [8, 6.8], 'k-', linewidth=1.5)
    
    for i, (x, label) in enumerate([(1, 'H'), (1.5, 'T'), (2, 'H'), 
                                     (4.5, 'H'), (5, 'T'), (5.5, 'H'),
                                     (7.5, 'H'), (8, 'T'), (8.5, 'H')]):
        rect = Rectangle((x-0.2, 4), 0.4, 0.6, facecolor='lightyellow', 
                        edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 4.3, label, ha='center', va='center', fontsize=8)
    
    for x_parent in [2, 5, 8]:
        for x_child in [x_parent-0.5, x_parent, x_parent+0.5]:
            ax.plot([x_parent, x_child], [6, 4.6], 'k-', linewidth=1, alpha=0.5)
    
    ax.text(5, 2, 'Universal gate set: {H, T}', ha='center', fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = axes[1, 1]
    
    precisions = np.array([1, 2, 3, 4, 5, 6])  # Decimal places
    epsilon_vals = 10.0**(-precisions)
    
    gate_counts_clifford_t = np.array([5, 15, 45, 135, 405, 1215])
    gate_counts_rotation = np.array([3, 5, 7, 9, 11, 13])
    
    ax.semilogy(precisions, gate_counts_clifford_t, 'bo-', linewidth=2, 
               markersize=8, label='Clifford+T')
    ax.semilogy(precisions, gate_counts_rotation, 'rs-', linewidth=2, 
               markersize=8, label='Continuous Rotations')
    
    ax.set_xlabel('Precision (decimal places)')
    ax.set_ylabel('Gate Count (log scale)')
    ax.set_title('Gate Count vs Precision')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('universal_gates.png', dpi=300, bbox_inches='tight')
    print("Saved: universal_gates.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about computational view of qubits.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: Computational View of Qubits")
    print("="*80)
    
    print("\n1. QUANTUM GATES:")
    print("   • Unitary operations that transform qubit states")
    print("   • Reversible (unlike classical gates)")
    print("   • Preserve probability (norm)")
    print("   • Can create superposition and entanglement")
    
    print("\n2. SINGLE-QUBIT GATES:")
    print("   • Pauli-X: Bit flip (|0⟩ ↔ |1⟩)")
    print("   • Pauli-Y: Bit flip with phase")
    print("   • Pauli-Z: Phase flip")
    print("   • Hadamard: Creates superposition")
    print("   • S: π/2 phase gate")
    print("   • T: π/4 phase gate")
    
    print("\n3. GATE PROPERTIES:")
    print("   • Unitarity: U†U = I (reversibility)")
    print("   • Hermiticity: U† = U (Pauli gates)")
    print("   • Eigenvalues on unit circle")
    print("   • Non-commutativity: [X,Y] ≠ 0")
    
    print("\n4. UNIVERSAL GATE SETS:")
    print("   • {H, T, CNOT}: Clifford + T (most common)")
    print("   • {Rx, Ry, Rz, CNOT}: Continuous rotations")
    print("   • {U3, CNOT}: Arbitrary single-qubit + CNOT")
    print("   • Any quantum computation can be approximated")
    
    print("\n5. QUANTUM CIRCUITS:")
    print("   • Sequences of gates applied to qubits")
    print("   • Time flows left to right")
    print("   • Circuit depth: Number of sequential layers")
    print("   • Gate count: Total number of gates")
    
    print("\n6. GATE DECOMPOSITION:")
    print("   • Any single-qubit gate: U = e^(iα) Rz(β) Ry(γ) Rz(δ)")
    print("   • Solovay-Kitaev: Approximate any gate with O(log^c(1/ε)) gates")
    print("   • Trade-off: Precision vs gate count")
    
    print("\n7. COMPUTATIONAL COMPLEXITY:")
    print("   • Circuit depth: Critical for decoherence")
    print("   • Gate count: Affects error accumulation")
    print("   • Qubit count: Limits current implementations")
    print("   • Two-qubit gates more expensive than single-qubit")
    
    print("\n8. QUANTUM ADVANTAGE:")
    print("   • Superposition: Process multiple inputs simultaneously")
    print("   • Interference: Amplify correct, cancel wrong answers")
    print("   • Entanglement: Create impossible classical correlations")
    print("   • Exponential state space: n qubits → 2^n amplitudes")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("COMPUTATIONAL VIEW OF QUBITS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates quantum gates, circuits, and computational")
    print("operations on qubits.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating quantum gates visualization...")
    plot_quantum_gates()
    
    print("\n2. Creating gate matrices visualization...")
    plot_gate_matrices()
    
    print("\n3. Creating gate properties analysis...")
    plot_gate_properties()
    
    print("\n4. Creating quantum circuits visualization...")
    plot_quantum_circuits()
    
    print("\n5. Creating gate sequences visualization...")
    plot_gate_sequences()
    
    print("\n6. Creating universal gates analysis...")
    plot_universal_gates()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • quantum_gates.png")
    print("  • gate_matrices.png")
    print("  • gate_properties.png")
    print("  • quantum_circuits.png")
    print("  • gate_sequences.png")
    print("  • universal_gates.png")
    print("\nThese visualizations demonstrate the computational model of quantum")
    print("computing using gates and circuits.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
