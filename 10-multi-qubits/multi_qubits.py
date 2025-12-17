#!/usr/bin/env python3
"""
Multi-Qubit Systems - Comprehensive Python Demonstrations

This script demonstrates multi-qubit systems, tensor products, two-qubit gates,
quantum parallelism, and scaling properties.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from itertools import product

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

ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=complex)

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)


def tensor_product(*matrices):
    """Compute tensor product of multiple matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def plot_state_space_growth():
    """
    Visualize exponential growth of state space with number of qubits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Qubit State Space Growth', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    n_qubits = np.arange(1, 21)
    dimensions = 2**n_qubits
    
    ax.semilogy(n_qubits, dimensions, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Hilbert Space Dimension')
    ax.set_title('State Space Dimension: 2ⁿ')
    ax.grid(True, alpha=0.3, which='both')
    
    special_points = [10, 20, 30, 40, 50]
    for n in special_points:
        if n <= 20:
            dim = 2**n
            ax.plot(n, dim, 'ro', markersize=10)
            ax.text(n, dim*1.5, f'{n} qubits\n{dim:,}', ha='center', fontsize=8)
    
    ax = axes[0, 1]
    
    n_qubits_mem = np.arange(1, 51)
    bytes_per_amplitude = 16  # Complex number (8 bytes real + 8 bytes imag)
    memory_bytes = 2**n_qubits_mem * bytes_per_amplitude
    memory_gb = memory_bytes / (1024**3)
    
    ax.semilogy(n_qubits_mem, memory_gb, 'g-', linewidth=2)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Memory (GB, log scale)')
    ax.set_title('Classical Simulation Memory Requirements')
    ax.grid(True, alpha=0.3, which='both')
    
    milestones = [(30, 'Laptop\n~17 GB'), (40, 'Server\n~17 TB'), (50, 'Supercomputer\n~18 PB')]
    for n, label in milestones:
        mem = 2**n * bytes_per_amplitude / (1024**3)
        ax.plot(n, mem, 'ro', markersize=10)
        ax.text(n, mem*2, label, ha='center', fontsize=8)
    
    ax = axes[1, 0]
    
    n_range = np.arange(1, 11)
    
    for n in n_range:
        basis_states = 2**n
        ax.bar(n, basis_states, color=plt.cm.viridis(n/10), alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Number of Basis States')
    ax.set_title('Computational Basis States: 2ⁿ')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    for n in n_range:
        basis_states = 2**n
        ax.text(n, basis_states, f'{basis_states}', ha='center', va='bottom', fontsize=8)
    
    ax = axes[1, 1]
    
    n_qubits_comp = np.arange(1, 16)
    classical_states = n_qubits_comp  # n bits can represent n states at once
    quantum_states = 2**n_qubits_comp  # n qubits can be in superposition of 2^n states
    
    ax.semilogy(n_qubits_comp, classical_states, 'r-', linewidth=2, marker='s', 
               markersize=6, label='Classical (n bits)')
    ax.semilogy(n_qubits_comp, quantum_states, 'b-', linewidth=2, marker='o', 
               markersize=6, label='Quantum (n qubits)')
    
    ax.set_xlabel('Number of Bits/Qubits')
    ax.set_ylabel('Simultaneous States (log scale)')
    ax.set_title('Classical vs Quantum State Capacity')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('state_space_growth.png', dpi=300, bbox_inches='tight')
    print("Saved: state_space_growth.png")
    plt.show()


def plot_two_qubit_gates():
    """
    Visualize two-qubit gates and their effects.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Two-Qubit Gates', fontsize=16, fontweight='bold')
    
    ket00 = tensor_product(ket0, ket0)
    ket01 = tensor_product(ket0, ket1)
    ket10 = tensor_product(ket1, ket0)
    ket11 = tensor_product(ket1, ket1)
    
    basis_states = [ket00, ket01, ket10, ket11]
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    gates = [
        ('CNOT', CNOT),
        ('CZ', CZ),
        ('SWAP', SWAP)
    ]
    
    for idx, (gate_name, gate) in enumerate(gates):
        ax = axes[0, idx]
        
        real_part = np.real(gate)
        im = ax.imshow(real_part, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        
        for i in range(4):
            for j in range(4):
                val = real_part[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(basis_labels)
        ax.set_yticklabels(basis_labels)
        ax.set_title(f'{gate_name} Gate Matrix')
        
        ax = axes[1, idx]
        
        x_pos = np.arange(4)
        width = 0.35
        
        input_probs = [1, 1, 1, 1]  # Each basis state has probability 1
        bars1 = ax.bar(x_pos - width/2, input_probs, width, label='Input', 
                      color='lightblue', alpha=0.7, edgecolor='black')
        
        output_mapping = []
        for basis_state in basis_states:
            output_state = gate @ basis_state
            output_idx = np.argmax(np.abs(output_state))
            output_mapping.append(output_idx)
        
        output_probs = [0] * 4
        for i, out_idx in enumerate(output_mapping):
            output_probs[out_idx] += 0.25
        
        bars2 = ax.bar(x_pos + width/2, [1]*4, width, label='Output', 
                      color='lightcoral', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{gate_name} Action on Basis States')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(basis_labels)
        ax.legend()
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, out_idx in enumerate(output_mapping):
            if i != out_idx:
                ax.annotate('', xy=(out_idx + width/2, 1.3), xytext=(i - width/2, 1.3),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig('two_qubit_gates.png', dpi=300, bbox_inches='tight')
    print("Saved: two_qubit_gates.png")
    plt.show()


def plot_entanglement_creation():
    """
    Visualize creation of entangled states from separable states.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Entanglement Creation', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Bell State Preparation: H ⊗ I → CNOT')
    
    ax.plot([0, 10], [2, 2], 'k-', linewidth=2)
    ax.plot([0, 10], [1, 1], 'k-', linewidth=2)
    
    ax.text(0.5, 2, '|0⟩', fontsize=12, ha='center')
    ax.text(0.5, 1, '|0⟩', fontsize=12, ha='center')
    
    rect = Rectangle((2-0.3, 1.7), 0.6, 0.6, facecolor='lightblue', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 2, 'H', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(3.5, 2, '|+⟩', fontsize=10, ha='center')
    ax.text(3.5, 1, '|0⟩', fontsize=10, ha='center')
    
    ax.plot([5, 5], [1, 2], 'k-', linewidth=2)
    circle = Circle((5, 2), 0.15, facecolor='black')
    ax.add_patch(circle)
    circle2 = Circle((5, 1), 0.3, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle2)
    ax.plot([4.8, 5.2], [1, 1], 'k-', linewidth=2)
    ax.plot([5, 5], [0.8, 1.2], 'k-', linewidth=2)
    
    ax.text(7, 1.5, '(|00⟩+|11⟩)/√2', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax = axes[0, 1]
    
    initial_state = tensor_product(ket0, ket0)
    
    H_I = tensor_product(H, I)
    after_h = H_I @ initial_state
    
    bell_state = CNOT @ after_h
    
    states = [initial_state, after_h, bell_state]
    state_labels = ['|00⟩', 'H⊗I|00⟩', 'Bell State']
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    x = np.arange(len(basis_labels))
    width = 0.25
    
    for i, (state, label) in enumerate(zip(states, state_labels)):
        probs = np.abs(state.flatten())**2
        ax.bar(x + i*width, probs, width, label=label, alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('State Evolution During Bell State Preparation')
    ax.set_xticks(x + width)
    ax.set_xticklabels(basis_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.6)
    
    ax = axes[1, 0]
    
    separable = tensor_product((ket0 + ket1)/np.sqrt(2), ket0)
    sep_probs = np.abs(separable.flatten())**2
    
    entangled = (tensor_product(ket0, ket0) + tensor_product(ket1, ket1)) / np.sqrt(2)
    ent_probs = np.abs(entangled.flatten())**2
    
    x = np.arange(4)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sep_probs, width, label='Separable: |+⟩|0⟩', 
                   color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, ent_probs, width, label='Entangled: |Φ⁺⟩', 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Probability')
    ax.set_title('Separable vs Entangled States')
    ax.set_xticks(x)
    ax.set_xticklabels(basis_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.6)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax = axes[1, 1]
    
    bell_states = {
        '|Φ⁺⟩': (tensor_product(ket0, ket0) + tensor_product(ket1, ket1)) / np.sqrt(2),
        '|Φ⁻⟩': (tensor_product(ket0, ket0) - tensor_product(ket1, ket1)) / np.sqrt(2),
        '|Ψ⁺⟩': (tensor_product(ket0, ket1) + tensor_product(ket1, ket0)) / np.sqrt(2),
        '|Ψ⁻⟩': (tensor_product(ket0, ket1) - tensor_product(ket1, ket0)) / np.sqrt(2)
    }
    
    x = np.arange(4)
    width = 0.2
    
    for i, (name, state) in enumerate(bell_states.items()):
        probs = np.abs(state.flatten())**2
        ax.bar(x + i*width, probs, width, label=name, alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('Four Bell States')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(basis_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig('entanglement_creation.png', dpi=300, bbox_inches='tight')
    print("Saved: entanglement_creation.png")
    plt.show()


def plot_quantum_parallelism():
    """
    Visualize quantum parallelism with multi-qubit systems.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Parallelism', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    n_qubits_range = np.arange(1, 11)
    superposition_states = 2**n_qubits_range
    
    ax.semilogy(n_qubits_range, superposition_states, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('States in Superposition')
    ax.set_title('Hadamard Transform: H⊗ⁿ|0⟩⊗ⁿ')
    ax.grid(True, alpha=0.3, which='both')
    
    for n in [1, 2, 3, 4, 5]:
        states = 2**n
        ax.plot(n, states, 'ro', markersize=10)
        ax.text(n, states*1.5, f'{states}', ha='center', fontsize=10)
    
    ax = axes[0, 1]
    
    n = 2
    basis_states = [f'{i:0{n}b}' for i in range(2**n)]
    
    classical_evals = list(range(1, 2**n + 1))
    
    quantum_evals = [1] * (2**n)
    
    x = np.arange(2**n)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, classical_evals, width, label='Classical (sequential)', 
                   color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, quantum_evals, width, label='Quantum (parallel)', 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Evaluation Step')
    ax.set_title('Function Evaluation: Classical vs Quantum')
    ax.set_xticks(x)
    ax.set_xticklabels(basis_states)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    
    n_inputs = 2**n_qubits_range
    classical_time = n_inputs  # Linear in number of inputs
    quantum_time = np.ones_like(n_inputs, dtype=float)  # Constant time
    
    ax.semilogy(n_qubits_range, classical_time, 'r-', linewidth=2, marker='s', 
               markersize=6, label='Classical')
    ax.semilogy(n_qubits_range, quantum_time, 'b-', linewidth=2, marker='o', 
               markersize=6, label='Quantum')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Evaluation Time (arbitrary units)')
    ax.set_title('Quantum Parallelism Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[1, 1]
    
    n = 3
    H_3 = tensor_product(H, H, H)
    initial = tensor_product(ket0, ket0, ket0)
    superposition = H_3 @ initial
    
    amplitudes = np.abs(superposition.flatten())
    phases = np.angle(superposition.flatten())
    
    basis_states_3 = [f'|{i:03b}⟩' for i in range(2**n)]
    x = np.arange(2**n)
    
    bars = ax.bar(x, amplitudes, color=plt.cm.viridis(phases / (2*np.pi)), 
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Amplitude')
    ax.set_title('Uniform Superposition: H⊗³|000⟩')
    ax.set_xticks(x)
    ax.set_xticklabels(basis_states_3, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.5)
    
    for bar, amp in zip(bars, amplitudes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{amp:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('quantum_parallelism.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_parallelism.png")
    plt.show()


def plot_measurement_correlations():
    """
    Visualize measurement correlations in multi-qubit systems.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Qubit Measurement Correlations', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    separable = tensor_product((ket0 + ket1)/np.sqrt(2), ket0)
    
    n_shots = 1000
    probs = np.abs(separable.flatten())**2
    measurements = np.random.choice(4, size=n_shots, p=probs)
    
    outcomes = ['00', '01', '10', '11']
    counts = [np.sum(measurements == i) for i in range(4)]
    
    bars = ax.bar(outcomes, counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('Separable State: |+⟩|0⟩ (Independent)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[0, 1]
    
    bell = (tensor_product(ket0, ket0) + tensor_product(ket1, ket1)) / np.sqrt(2)
    
    probs_bell = np.abs(bell.flatten())**2
    measurements_bell = np.random.choice(4, size=n_shots, p=probs_bell)
    
    counts_bell = [np.sum(measurements_bell == i) for i in range(4)]
    
    bars = ax.bar(outcomes, counts_bell, color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('Bell State: (|00⟩+|11⟩)/√2 (Correlated)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_bell):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 0]
    
    
    first_qubit_outcomes = ['0', '1']
    second_qubit_given_first = {
        '0': [1.0, 0.0],  # P(second=0|first=0), P(second=1|first=0)
        '1': [0.0, 1.0]   # P(second=0|first=1), P(second=1|first=1)
    }
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, second_qubit_given_first['0'], width, 
                   label='Given first=0', color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, second_qubit_given_first['1'], width, 
                   label='Given first=1', color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Probability')
    ax.set_title('Partial Measurement: Second Qubit Given First')
    ax.set_xticks(x)
    ax.set_xticklabels(['Second=0', 'Second=1'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)
    
    ax = axes[1, 1]
    
    states = {
        'Separable\n|+⟩|0⟩': separable,
        'Bell\n|Φ⁺⟩': bell,
        'Product\n|0⟩|0⟩': tensor_product(ket0, ket0),
        'GHZ-like\n(|00⟩+|11⟩)/√2': bell
    }
    
    correlations = []
    for name, state in states.items():
        probs = np.abs(state.flatten())**2
        correlation = probs[0] + probs[3] - probs[1] - probs[2]  # P(00)+P(11)-P(01)-P(10)
        correlations.append(correlation)
    
    bars = ax.bar(range(len(states)), correlations, color='lightgreen', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Measurement Correlation Strength')
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(list(states.keys()), rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('measurement_correlations.png', dpi=300, bbox_inches='tight')
    print("Saved: measurement_correlations.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about multi-qubit systems.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: Multi-Qubit Systems")
    print("="*80)
    
    print("\n1. STATE SPACE GROWTH:")
    print("   • n qubits → 2ⁿ dimensional Hilbert space")
    print("   • Exponential growth enables quantum advantage")
    print("   • 50 qubits → 10¹⁵ dimensions (intractable classically)")
    print("   • Memory: 2ⁿ⁺⁴ bytes for classical simulation")
    
    print("\n2. TENSOR PRODUCT STRUCTURE:")
    print("   • Multi-qubit states: |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩")
    print("   • Separable: Can be factored into single-qubit states")
    print("   • Entangled: Cannot be factored (most states are entangled)")
    print("   • Basis states: |i⟩ for i = 0 to 2ⁿ-1")
    
    print("\n3. TWO-QUBIT GATES:")
    print("   • CNOT: Controlled-NOT (creates entanglement)")
    print("   • CZ: Controlled-Z (phase gate)")
    print("   • SWAP: Exchanges qubit states")
    print("   • Toffoli: Three-qubit controlled-controlled-NOT")
    
    print("\n4. QUANTUM PARALLELISM:")
    print("   • H⊗ⁿ|0⟩⊗ⁿ = (1/√2ⁿ) Σₓ |x⟩")
    print("   • All 2ⁿ inputs in superposition")
    print("   • Function evaluated on all inputs simultaneously")
    print("   • Challenge: Extracting useful information")
    
    print("\n5. MEASUREMENT:")
    print("   • Full measurement: All qubits collapse")
    print("   • Partial measurement: Some qubits collapse")
    print("   • Correlations: Entangled states show perfect correlations")
    print("   • Outcome probabilities: |cᵢ|² for basis state |i⟩")
    
    print("\n6. SCALING CHALLENGES:")
    print("   • Decoherence increases with qubit count")
    print("   • Gate errors accumulate")
    print("   • Connectivity constraints require SWAP gates")
    print("   • Classical simulation becomes intractable")
    
    print("\n7. APPLICATIONS:")
    print("   • Shor's algorithm: Integer factorization")
    print("   • Grover's algorithm: Database search")
    print("   • Quantum simulation: Chemistry, materials")
    print("   • Quantum machine learning: High-dimensional data")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("MULTI-QUBIT SYSTEMS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates multi-qubit systems, tensor products,")
    print("two-qubit gates, quantum parallelism, and scaling properties.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating state space growth visualization...")
    plot_state_space_growth()
    
    print("\n2. Creating two-qubit gates visualization...")
    plot_two_qubit_gates()
    
    print("\n3. Creating entanglement creation visualization...")
    plot_entanglement_creation()
    
    print("\n4. Creating quantum parallelism visualization...")
    plot_quantum_parallelism()
    
    print("\n5. Creating measurement correlations visualization...")
    plot_measurement_correlations()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • state_space_growth.png")
    print("  • two_qubit_gates.png")
    print("  • entanglement_creation.png")
    print("  • quantum_parallelism.png")
    print("  • measurement_correlations.png")
    print("\nThese visualizations demonstrate multi-qubit systems and their")
    print("exponential scaling properties in quantum computing.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
