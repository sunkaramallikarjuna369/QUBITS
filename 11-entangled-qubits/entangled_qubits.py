#!/usr/bin/env python3
"""
Entangled Qubits - Comprehensive Python Demonstrations

This script demonstrates quantum entanglement, Bell states, EPR paradox,
Bell's theorem, and applications of entanglement.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch
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

ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


def tensor_product(*matrices):
    """Compute tensor product of multiple matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def concurrence(state):
    """Calculate concurrence (entanglement measure) for a two-qubit pure state."""
    rho = state @ state.conj().T
    
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y_2 = tensor_product(sigma_y, sigma_y)
    
    rho_tilde = sigma_y_2 @ rho.conj() @ sigma_y_2
    eigenvalues = np.linalg.eigvalsh(rho @ rho_tilde)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    return C


def plot_bell_states():
    """
    Visualize the four Bell states and their properties.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Four Bell States', fontsize=16, fontweight='bold')
    
    ket00 = tensor_product(ket0, ket0)
    ket01 = tensor_product(ket0, ket1)
    ket10 = tensor_product(ket1, ket0)
    ket11 = tensor_product(ket1, ket1)
    
    bell_states = {
        '|Φ⁺⟩ = (|00⟩+|11⟩)/√2': (ket00 + ket11) / np.sqrt(2),
        '|Φ⁻⟩ = (|00⟩-|11⟩)/√2': (ket00 - ket11) / np.sqrt(2),
        '|Ψ⁺⟩ = (|01⟩+|10⟩)/√2': (ket01 + ket10) / np.sqrt(2),
        '|Ψ⁻⟩ = (|01⟩-|10⟩)/√2': (ket01 - ket10) / np.sqrt(2)
    }
    
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    for idx, (name, state) in enumerate(bell_states.items()):
        ax = axes[idx // 2, idx % 2]
        
        amplitudes = state.flatten()
        real_parts = np.real(amplitudes)
        imag_parts = np.imag(amplitudes)
        magnitudes = np.abs(amplitudes)
        
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_parts, width, label='Real', 
                      color='lightblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, imag_parts, width, label='Imaginary', 
                      color='lightcoral', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Amplitude')
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.8, 0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, val in zip(bars1, real_parts):
            if abs(val) > 0.01:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=8)
        
        C = concurrence(state)
        ax.text(0.98, 0.98, f'Concurrence: {C:.3f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('bell_states.png', dpi=300, bbox_inches='tight')
    print("Saved: bell_states.png")
    plt.show()


def plot_entanglement_creation():
    """
    Visualize the process of creating entanglement.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Creating Entanglement: Bell State Preparation', fontsize=16, fontweight='bold')
    
    initial = tensor_product(ket0, ket0)
    
    H_I = tensor_product(H, I)
    after_h = H_I @ initial
    
    bell_state = CNOT @ after_h
    
    states = [initial, after_h, bell_state]
    state_names = ['Initial: |00⟩', 'After H⊗I', 'After CNOT: |Φ⁺⟩']
    basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    for idx, (state, name) in enumerate(zip(states[:3], state_names)):
        ax = axes[idx // 2, idx % 2]
        
        probs = np.abs(state.flatten())**2
        bars = ax.bar(basis_labels, probs, color=plt.cm.viridis(probs), 
                     alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Probability')
        ax.set_title(name)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.6)
        
        for bar, val in zip(bars, probs):
            if val > 0.01:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        C = concurrence(state)
        ax.text(0.98, 0.98, f'Concurrence: {C:.3f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Quantum Circuit')
    
    ax.plot([0, 10], [2, 2], 'k-', linewidth=2)
    ax.plot([0, 10], [1, 1], 'k-', linewidth=2)
    
    ax.text(0.5, 2, 'q₀:', fontsize=12, ha='right')
    ax.text(0.5, 1, 'q₁:', fontsize=12, ha='right')
    
    ax.text(1, 2.3, '|0⟩', fontsize=10, ha='center')
    ax.text(1, 0.7, '|0⟩', fontsize=10, ha='center')
    
    rect = Rectangle((2.5, 1.7), 0.6, 0.6, facecolor='lightblue', 
                    edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2.8, 2, 'H', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.plot([5, 5], [1, 2], 'k-', linewidth=2)
    circle = Circle((5, 2), 0.15, facecolor='black')
    ax.add_patch(circle)
    circle2 = Circle((5, 1), 0.3, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle2)
    ax.plot([4.8, 5.2], [1, 1], 'k-', linewidth=2)
    ax.plot([5, 5], [0.8, 1.2], 'k-', linewidth=2)
    
    ax.text(7.5, 1.5, '|Φ⁺⟩', fontsize=14, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('entanglement_creation.png', dpi=300, bbox_inches='tight')
    print("Saved: entanglement_creation.png")
    plt.show()


def plot_measurement_correlations():
    """
    Visualize measurement correlations in entangled states.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Measurement Correlations in Entangled States', fontsize=16, fontweight='bold')
    
    ket00 = tensor_product(ket0, ket0)
    ket11 = tensor_product(ket1, ket1)
    bell_phi_plus = (ket00 + ket11) / np.sqrt(2)
    
    ket01 = tensor_product(ket0, ket1)
    ket10 = tensor_product(ket1, ket0)
    bell_psi_plus = (ket01 + ket10) / np.sqrt(2)
    
    ax = axes[0, 0]
    
    n_shots = 1000
    probs = np.abs(bell_phi_plus.flatten())**2
    measurements = np.random.choice(4, size=n_shots, p=probs)
    
    outcomes = ['00', '01', '10', '11']
    counts = [np.sum(measurements == i) for i in range(4)]
    
    bars = ax.bar(outcomes, counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('|Φ⁺⟩: Perfect Matching Correlation')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[0, 1]
    
    probs_psi = np.abs(bell_psi_plus.flatten())**2
    measurements_psi = np.random.choice(4, size=n_shots, p=probs_psi)
    
    counts_psi = [np.sum(measurements_psi == i) for i in range(4)]
    
    bars = ax.bar(outcomes, counts_psi, color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('|Ψ⁺⟩: Perfect Anti-Correlation')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_psi):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 0]
    
    first_outcomes = ['First=0', 'First=1']
    second_given_first = {
        'Second=0': [1.0, 0.0],
        'Second=1': [0.0, 1.0]
    }
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, second_given_first['Second=0'], width, 
                   label='Second=0', color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, second_given_first['Second=1'], width, 
                   label='Second=1', color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Probability')
    ax.set_title('|Φ⁺⟩: Conditional Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(first_outcomes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)
    
    ax = axes[1, 1]
    
    separable = tensor_product((ket0 + ket1)/np.sqrt(2), ket0)
    
    states = {
        'Separable\n|+⟩|0⟩': separable,
        'Bell |Φ⁺⟩\n(|00⟩+|11⟩)/√2': bell_phi_plus,
        'Bell |Ψ⁺⟩\n(|01⟩+|10⟩)/√2': bell_psi_plus,
        'Product\n|0⟩|0⟩': ket00
    }
    
    correlations = []
    for name, state in states.items():
        probs = np.abs(state.flatten())**2
        correlation = probs[0] + probs[3] - probs[1] - probs[2]
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


def plot_bell_inequality():
    """
    Visualize Bell's inequality and quantum violation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Bell's Theorem and Inequality Violation", fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    
    angles = np.linspace(0, np.pi/2, 50)
    classical_bound = 2 * np.ones_like(angles)
    quantum_values = 2 * np.sqrt(2) * np.abs(np.cos(angles) + np.sin(angles)) / np.sqrt(2)
    
    ax.plot(angles * 180/np.pi, classical_bound, 'r--', linewidth=2, label='Classical bound')
    ax.plot(angles * 180/np.pi, quantum_values, 'b-', linewidth=2, label='Quantum prediction')
    ax.axhline(y=2*np.sqrt(2), color='g', linestyle=':', linewidth=2, label='Quantum maximum')
    
    ax.set_xlabel('Measurement Angle (degrees)')
    ax.set_ylabel('CHSH Parameter S')
    ax.set_title("CHSH Inequality: Classical vs Quantum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    ax.fill_between(angles * 180/np.pi, classical_bound, quantum_values, 
                    where=(quantum_values > classical_bound), alpha=0.3, color='blue',
                    label='Quantum violation')
    
    ax = axes[0, 1]
    
    experiments = ['Aspect\n1982', 'Weihs\n1998', 'Giustina\n2013', 'Hensen\n2015', 'Shalm\n2015']
    classical_limit = [2.0] * len(experiments)
    quantum_prediction = [2.828] * len(experiments)
    experimental_values = [2.70, 2.73, 2.76, 2.42, 2.77]  # Approximate values
    errors = [0.05, 0.04, 0.03, 0.08, 0.04]
    
    x = np.arange(len(experiments))
    
    ax.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Classical limit')
    ax.axhline(y=2.828, color='g', linestyle=':', linewidth=2, label='Quantum maximum')
    ax.errorbar(x, experimental_values, yerr=errors, fmt='bo', markersize=8, 
               capsize=5, linewidth=2, label='Experimental')
    
    ax.set_ylabel('CHSH Parameter S')
    ax.set_title('Experimental Bell Tests')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(1.5, 3.0)
    
    ax = axes[1, 0]
    
    angles = np.linspace(0, np.pi, 100)
    
    classical_corr = 1 - 2 * angles / np.pi
    
    quantum_corr = -np.cos(2 * angles)
    
    ax.plot(angles * 180/np.pi, classical_corr, 'r--', linewidth=2, label='Local hidden variable')
    ax.plot(angles * 180/np.pi, quantum_corr, 'b-', linewidth=2, label='Quantum mechanics')
    
    ax.set_xlabel('Relative Angle (degrees)')
    ax.set_ylabel('Correlation E(a,b)')
    ax.set_title('Correlation Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax = axes[1, 1]
    
    entanglement_params = np.linspace(0, 1, 50)
    
    chsh_values = 2 * np.sqrt(2) * entanglement_params
    
    ax.plot(entanglement_params, chsh_values, 'b-', linewidth=2)
    ax.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Classical limit')
    ax.axhline(y=2.828, color='g', linestyle=':', linewidth=2, label='Maximum violation')
    
    ax.set_xlabel('Entanglement (Concurrence)')
    ax.set_ylabel('CHSH Parameter S')
    ax.set_title('Bell Violation vs Entanglement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    ax.fill_between(entanglement_params, 0, 2, alpha=0.2, color='red', label='Classical')
    ax.fill_between(entanglement_params, 2, chsh_values, 
                    where=(chsh_values > 2), alpha=0.2, color='blue', label='Quantum')
    
    plt.tight_layout()
    plt.savefig('bell_inequality.png', dpi=300, bbox_inches='tight')
    print("Saved: bell_inequality.png")
    plt.show()


def plot_entanglement_applications():
    """
    Visualize applications of quantum entanglement.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Applications of Quantum Entanglement', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Quantum Teleportation')
    
    ax.text(1, 4, 'Alice', fontsize=12, fontweight='bold', ha='center')
    ax.add_patch(Circle((1, 3), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(1, 3, '|ψ⟩', ha='center', va='center', fontsize=10)
    ax.add_patch(Circle((1, 2), 0.3, facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(1, 2, 'A', ha='center', va='center', fontsize=10)
    
    ax.text(9, 4, 'Bob', fontsize=12, fontweight='bold', ha='center')
    ax.add_patch(Circle((9, 2), 0.3, facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(9, 2, 'B', ha='center', va='center', fontsize=10)
    
    ax.plot([1, 9], [2, 2], 'r--', linewidth=2, label='Entanglement')
    ax.text(5, 2.3, 'Bell pair', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.annotate('', xy=(9, 3.5), xytext=(1, 3.5),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(5, 3.7, 'Classical bits', ha='center', fontsize=10, color='blue')
    
    ax.text(9, 1, '|ψ⟩', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Superdense Coding')
    
    ax.text(1, 4, 'Alice', fontsize=12, fontweight='bold', ha='center')
    ax.add_patch(Circle((1, 2.5), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(1, 2.5, 'A', ha='center', va='center', fontsize=10)
    ax.text(1, 1.5, '2 classical bits', ha='center', fontsize=9)
    
    ax.text(9, 4, 'Bob', fontsize=12, fontweight='bold', ha='center')
    ax.add_patch(Circle((9, 2.5), 0.3, facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(9, 2.5, 'B', ha='center', va='center', fontsize=10)
    
    ax.plot([1, 5], [2.5, 2.5], 'r--', linewidth=2)
    ax.text(3, 2.8, 'Bell pair', ha='center', fontsize=9)
    
    ax.annotate('', xy=(9, 2.5), xytext=(5, 2.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(7, 2.8, '1 qubit', ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax.text(9, 1.5, '2 bits received!', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax = axes[1, 0]
    
    n_bits = 100
    alice_bases = np.random.choice([0, 1], n_bits)
    bob_bases = np.random.choice([0, 1], n_bits)
    
    matching = alice_bases == bob_bases
    key_length = np.sum(matching)
    
    distances = np.linspace(0, 100, 50)
    key_rates = 1000 * np.exp(-distances / 20)  # Exponential decay
    
    ax.plot(distances, key_rates, 'b-', linewidth=2)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Key Rate (bits/s)')
    ax.set_title('QKD: Key Rate vs Distance')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1200)
    
    ax.axvline(x=50, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(50, 1000, 'Practical\nlimit', ha='center', fontsize=9)
    
    ax = axes[1, 1]
    
    distances = np.linspace(0, 1000, 100)
    
    fidelity_direct = np.exp(-distances / 20)
    
    repeater_spacing = 100
    fidelity_repeater = np.exp(-repeater_spacing / 20) ** (distances / repeater_spacing)
    
    ax.plot(distances, fidelity_direct, 'r-', linewidth=2, label='Direct transmission')
    ax.plot(distances, fidelity_repeater, 'b-', linewidth=2, label='Quantum repeaters')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Entanglement Fidelity')
    ax.set_title('Long-Distance Entanglement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(900, 0.52, 'Threshold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('entanglement_applications.png', dpi=300, bbox_inches='tight')
    print("Saved: entanglement_applications.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about entangled qubits.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: Entangled Qubits")
    print("="*80)
    
    print("\n1. QUANTUM ENTANGLEMENT:")
    print("   • Correlation between qubits that cannot be described independently")
    print("   • State cannot be factored: |ψ⟩ ≠ |ψ₁⟩ ⊗ |ψ₂⟩")
    print("   • Measuring one qubit affects the other instantly")
    print("   • Stronger than any classical correlation")
    
    print("\n2. BELL STATES:")
    print("   • Four maximally entangled two-qubit states")
    print("   • |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print("   • |Φ⁻⟩ = (|00⟩ - |11⟩)/√2")
    print("   • |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2")
    print("   • |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2")
    
    print("\n3. CREATING ENTANGLEMENT:")
    print("   • Use two-qubit gates (CNOT, CZ)")
    print("   • Bell state: H ⊗ I → CNOT")
    print("   • Start from separable state |00⟩")
    print("   • Result: Maximally entangled state")
    
    print("\n4. MEASUREMENT CORRELATIONS:")
    print("   • |Φ⁺⟩: Perfect matching (both 0 or both 1)")
    print("   • |Ψ⁺⟩: Perfect anti-correlation (opposite results)")
    print("   • Correlations persist regardless of distance")
    print("   • No faster-than-light communication possible")
    
    print("\n5. EPR PARADOX & BELL'S THEOREM:")
    print("   • EPR (1935): Entanglement seems non-local")
    print("   • Bell (1964): No local hidden variables")
    print("   • Bell inequality: Classical ≤ 2, Quantum ≤ 2√2")
    print("   • Experiments confirm quantum mechanics")
    
    print("\n6. ENTANGLEMENT MEASURES:")
    print("   • Concurrence: C ∈ [0,1]")
    print("   • Entanglement entropy: S(ρₐ)")
    print("   • Schmidt rank: Number of non-zero coefficients")
    print("   • All zero for separable, non-zero for entangled")
    
    print("\n7. APPLICATIONS:")
    print("   • Quantum teleportation: Transfer state using entanglement")
    print("   • Superdense coding: Send 2 bits with 1 qubit")
    print("   • QKD: Secure key distribution (E91 protocol)")
    print("   • Quantum error correction: Protect quantum information")
    
    print("\n8. MULTI-QUBIT ENTANGLEMENT:")
    print("   • GHZ state: (|000⟩ + |111⟩)/√2")
    print("   • W state: (|001⟩ + |010⟩ + |100⟩)/√3")
    print("   • Cluster states: For measurement-based QC")
    print("   • Different entanglement structures and properties")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("ENTANGLED QUBITS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates quantum entanglement, Bell states,")
    print("EPR paradox, Bell's theorem, and applications of entanglement.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating Bell states visualization...")
    plot_bell_states()
    
    print("\n2. Creating entanglement creation visualization...")
    plot_entanglement_creation()
    
    print("\n3. Creating measurement correlations visualization...")
    plot_measurement_correlations()
    
    print("\n4. Creating Bell inequality visualization...")
    plot_bell_inequality()
    
    print("\n5. Creating entanglement applications visualization...")
    plot_entanglement_applications()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • bell_states.png")
    print("  • entanglement_creation.png")
    print("  • measurement_correlations.png")
    print("  • bell_inequality.png")
    print("  • entanglement_applications.png")
    print("\nThese visualizations demonstrate quantum entanglement and its")
    print("fundamental role in quantum information science.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
