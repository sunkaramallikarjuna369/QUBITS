#!/usr/bin/env python3
"""
Single Qubit Operations - Comprehensive Python Demonstrations

This script demonstrates single qubit operations, Bloch sphere representation,
rotations, measurements, and error models.

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


def state_to_bloch(state):
    """Convert quantum state to Bloch sphere coordinates."""
    alpha = state[0, 0]
    beta = state[1, 0]
    
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return x, y, z


def bloch_to_state(theta, phi):
    """Convert Bloch sphere coordinates to quantum state."""
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([[alpha], [beta]], dtype=complex)


def rotation_matrix(axis, angle):
    """Generate rotation matrix around given axis."""
    if axis == 'X':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * X
    elif axis == 'Y':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y
    elif axis == 'Z':
        return np.cos(angle/2) * I - 1j * np.sin(angle/2) * Z
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'")


def plot_bloch_sphere_states():
    """
    Visualize important quantum states on the Bloch sphere.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Important States on the Bloch Sphere', fontsize=16, fontweight='bold')
    
    states = {
        '|0⟩': ket0,
        '|1⟩': ket1,
        '|+⟩': (ket0 + ket1) / np.sqrt(2),
        '|−⟩': (ket0 - ket1) / np.sqrt(2),
        '|+i⟩': (ket0 + 1j*ket1) / np.sqrt(2),
        '|−i⟩': (ket0 - 1j*ket1) / np.sqrt(2)
    }
    
    for idx, (name, state) in enumerate(states.items()):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'k-', alpha=0.3, linewidth=1)
        
        x, y, z = state_to_bloch(state)
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.15, linewidth=3)
        
        ax.text(0, 0, 1.3, '|0⟩', fontsize=10, color='blue')
        ax.text(0, 0, -1.3, '|1⟩', fontsize=10, color='blue')
        ax.text(1.3, 0, 0, '|+⟩', fontsize=10, color='green')
        ax.text(0, 1.3, 0, '|+i⟩', fontsize=10, color='purple')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'State: {name}')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_states.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_states.png")
    plt.show()


def plot_single_qubit_rotations():
    """
    Visualize single qubit rotations around different axes.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Single Qubit Rotations', fontsize=16, fontweight='bold')
    
    axes_list = ['X', 'Y', 'Z']
    n_steps = 20
    
    for idx, axis in enumerate(axes_list):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        state = ket0.copy()
        trajectory_x, trajectory_y, trajectory_z = [], [], []
        
        for i in range(n_steps + 1):
            angle = (i / n_steps) * 2 * np.pi
            R = rotation_matrix(axis, angle)
            rotated_state = R @ ket0
            x, y, z = state_to_bloch(rotated_state)
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_z.append(z)
        
        ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2, label='Rotation path')
        
        ax.quiver(0, 0, 0, trajectory_x[0], trajectory_y[0], trajectory_z[0], 
                 color='blue', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
        ax.quiver(0, 0, 0, trajectory_x[n_steps//2], trajectory_y[n_steps//2], trajectory_z[n_steps//2], 
                 color='green', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
        
        if axis == 'X':
            ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'r-', linewidth=3, alpha=0.7)
        elif axis == 'Y':
            ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'g-', linewidth=3, alpha=0.7)
        elif axis == 'Z':
            ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'b-', linewidth=3, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Rotation around {axis}-axis')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.legend()
        
        ax2 = fig.add_subplot(2, 3, idx + 4)
        
        angles = np.linspace(0, 2*np.pi, n_steps)
        prob_0 = []
        prob_1 = []
        
        for angle in angles:
            R = rotation_matrix(axis, angle)
            rotated_state = R @ ket0
            prob_0.append(np.abs(rotated_state[0, 0])**2)
            prob_1.append(np.abs(rotated_state[1, 0])**2)
        
        ax2.plot(angles, prob_0, 'b-', linewidth=2, label='P(|0⟩)')
        ax2.plot(angles, prob_1, 'r-', linewidth=2, label='P(|1⟩)')
        ax2.set_xlabel('Rotation Angle (radians)')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Measurement Probabilities: R{axis}(θ)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        ax2.axvline(x=np.pi/2, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=np.pi, color='orange', linestyle='--', alpha=0.5)
        ax2.axvline(x=3*np.pi/2, color='purple', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('single_qubit_rotations.png', dpi=300, bbox_inches='tight')
    print("Saved: single_qubit_rotations.png")
    plt.show()


def plot_measurement_bases():
    """
    Visualize measurement in different bases.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Measurement in Different Bases', fontsize=16, fontweight='bold')
    
    test_state = (ket0 + ket1) / np.sqrt(2)
    
    ax = axes[0, 0]
    
    n_measurements = 1000
    prob_0 = np.abs(test_state[0, 0])**2
    prob_1 = np.abs(test_state[1, 0])**2
    
    measurements = np.random.choice([0, 1], size=n_measurements, p=[prob_0, prob_1])
    counts = [np.sum(measurements == 0), np.sum(measurements == 1)]
    
    bars = ax.bar(['|0⟩', '|1⟩'], counts, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax.axhline(y=n_measurements/2, color='green', linestyle='--', linewidth=2, label='Expected')
    ax.set_ylabel('Counts')
    ax.set_title('Computational Basis: {|0⟩, |1⟩}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[0, 1]
    
    state_x_basis = H @ test_state
    prob_plus = np.abs(state_x_basis[0, 0])**2
    prob_minus = np.abs(state_x_basis[1, 0])**2
    
    measurements_x = np.random.choice([0, 1], size=n_measurements, p=[prob_plus, prob_minus])
    counts_x = [np.sum(measurements_x == 0), np.sum(measurements_x == 1)]
    
    bars = ax.bar(['|+⟩', '|−⟩'], counts_x, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('X-Basis: {|+⟩, |−⟩}')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_x):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 0]
    
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
    state_y_basis = H @ S_dag @ test_state
    prob_plus_i = np.abs(state_y_basis[0, 0])**2
    prob_minus_i = np.abs(state_y_basis[1, 0])**2
    
    measurements_y = np.random.choice([0, 1], size=n_measurements, p=[prob_plus_i, prob_minus_i])
    counts_y = [np.sum(measurements_y == 0), np.sum(measurements_y == 1)]
    
    bars = ax.bar(['|+i⟩', '|−i⟩'], counts_y, color=['purple', 'cyan'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('Y-Basis: {|+i⟩, |−i⟩}')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 1]
    
    test_states = {
        '|0⟩': ket0,
        '|+⟩': (ket0 + ket1) / np.sqrt(2),
        '|+i⟩': (ket0 + 1j*ket1) / np.sqrt(2)
    }
    
    bases = ['Z-basis', 'X-basis', 'Y-basis']
    x = np.arange(len(bases))
    width = 0.25
    
    for i, (state_name, state) in enumerate(test_states.items()):
        probs = []
        
        probs.append(np.abs(state[0, 0])**2)
        
        state_x = H @ state
        probs.append(np.abs(state_x[0, 0])**2)
        
        state_y = H @ S_dag @ state
        probs.append(np.abs(state_y[0, 0])**2)
        
        ax.bar(x + i*width, probs, width, label=state_name, alpha=0.7)
    
    ax.set_ylabel('P(first outcome)')
    ax.set_title('Measurement Outcome Probabilities')
    ax.set_xticks(x + width)
    ax.set_xticklabels(bases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('measurement_bases.png', dpi=300, bbox_inches='tight')
    print("Saved: measurement_bases.png")
    plt.show()


def plot_quantum_errors():
    """
    Visualize different types of quantum errors.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Error Models', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    error_probs = np.linspace(0, 0.5, 50)
    initial_state = ket0
    
    fidelities = []
    for p in error_probs:
        rho = (1-p) * (initial_state @ initial_state.conj().T) + p * (X @ initial_state @ initial_state.conj().T @ X.conj().T)
        fidelity = np.real(np.trace(rho @ initial_state @ initial_state.conj().T))
        fidelities.append(fidelity)
    
    ax.plot(error_probs, fidelities, 'b-', linewidth=2)
    ax.set_xlabel('Error Probability (p)')
    ax.set_ylabel('Fidelity with |0⟩')
    ax.set_title('Bit Flip Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax = axes[0, 1]
    
    initial_state = (ket0 + ket1) / np.sqrt(2)  # |+⟩ state
    
    fidelities = []
    for p in error_probs:
        rho = (1-p) * (initial_state @ initial_state.conj().T) + p * (Z @ initial_state @ initial_state.conj().T @ Z.conj().T)
        fidelity = np.real(np.trace(rho @ initial_state @ initial_state.conj().T))
        fidelities.append(fidelity)
    
    ax.plot(error_probs, fidelities, 'r-', linewidth=2)
    ax.set_xlabel('Error Probability (p)')
    ax.set_ylabel('Fidelity with |+⟩')
    ax.set_title('Phase Flip Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax = axes[1, 0]
    
    initial_state = ket0
    
    fidelities = []
    for p in error_probs:
        rho_pure = initial_state @ initial_state.conj().T
        rho_mixed = I / 2  # Maximally mixed state
        rho = (1-p) * rho_pure + p * rho_mixed
        fidelity = np.real(np.trace(rho @ initial_state @ initial_state.conj().T))
        fidelities.append(fidelity)
    
    ax.plot(error_probs, fidelities, 'g-', linewidth=2)
    ax.set_xlabel('Error Probability (p)')
    ax.set_ylabel('Fidelity with |0⟩')
    ax.set_title('Depolarizing Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax = axes[1, 1]
    
    times = np.linspace(0, 5, 100)  # In units of T1
    initial_state = ket1  # Start in excited state
    
    populations_1 = []
    for t in times:
        gamma = 1 - np.exp(-t)  # Damping parameter
        pop_1 = (1 - gamma) * np.abs(initial_state[1, 0])**2
        populations_1.append(pop_1)
    
    ax.plot(times, populations_1, 'purple', linewidth=2, label='P(|1⟩)')
    ax.plot(times, 1 - np.array(populations_1), 'orange', linewidth=2, label='P(|0⟩)')
    ax.set_xlabel('Time (T₁ units)')
    ax.set_ylabel('Population')
    ax.set_title('Amplitude Damping (Energy Relaxation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax.text(1.1, 0.9, 'T₁', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('quantum_errors.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_errors.png")
    plt.show()


def plot_state_tomography():
    """
    Visualize quantum state tomography process.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum State Tomography', fontsize=16, fontweight='bold')
    
    theta_true = np.pi / 3
    phi_true = np.pi / 4
    true_state = bloch_to_state(theta_true, phi_true)
    
    ax = axes[0, 0]
    
    n_shots = 1000
    prob_0 = np.abs(true_state[0, 0])**2
    prob_1 = np.abs(true_state[1, 0])**2
    
    measurements_z = np.random.choice([0, 1], size=n_shots, p=[prob_0, prob_1])
    counts_z = [np.sum(measurements_z == 0), np.sum(measurements_z == 1)]
    
    bars = ax.bar(['|0⟩', '|1⟩'], counts_z, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('Z-Basis Measurement')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_z):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[0, 1]
    
    state_x = H @ true_state
    prob_plus = np.abs(state_x[0, 0])**2
    prob_minus = np.abs(state_x[1, 0])**2
    
    measurements_x = np.random.choice([0, 1], size=n_shots, p=[prob_plus, prob_minus])
    counts_x = [np.sum(measurements_x == 0), np.sum(measurements_x == 1)]
    
    bars = ax.bar(['|+⟩', '|−⟩'], counts_x, color=['green', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('X-Basis Measurement')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_x):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 0]
    
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
    state_y = H @ S_dag @ true_state
    prob_plus_i = np.abs(state_y[0, 0])**2
    prob_minus_i = np.abs(state_y[1, 0])**2
    
    measurements_y = np.random.choice([0, 1], size=n_shots, p=[prob_plus_i, prob_minus_i])
    counts_y = [np.sum(measurements_y == 0), np.sum(measurements_y == 1)]
    
    bars = ax.bar(['|+i⟩', '|−i⟩'], counts_y, color=['purple', 'cyan'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Counts')
    ax.set_title('Y-Basis Measurement')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, counts_y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom')
    
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    x_bloch = (counts_x[0] - counts_x[1]) / n_shots
    y_bloch = (counts_y[0] - counts_y[1]) / n_shots
    z_bloch = (counts_z[0] - counts_z[1]) / n_shots
    
    x_true, y_true, z_true = state_to_bloch(true_state)
    ax.quiver(0, 0, 0, x_true, y_true, z_true, color='red', 
             arrow_length_ratio=0.15, linewidth=3, label='True state')
    
    ax.quiver(0, 0, 0, x_bloch, y_bloch, z_bloch, color='blue', 
             arrow_length_ratio=0.15, linewidth=3, alpha=0.7, label='Reconstructed')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('State Reconstruction')
    ax.legend()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('state_tomography.png', dpi=300, bbox_inches='tight')
    print("Saved: state_tomography.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about single qubit operations.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: Single Qubit Operations")
    print("="*80)
    
    print("\n1. SINGLE QUBIT STATE SPACE:")
    print("   • 2D complex Hilbert space")
    print("   • State: |ψ⟩ = α|0⟩ + β|1⟩")
    print("   • Normalization: |α|² + |β|² = 1")
    print("   • Infinite possible states (continuous parameters)")
    print("   • Only 1 classical bit extractable by measurement")
    
    print("\n2. BLOCH SPHERE:")
    print("   • Geometric representation of qubit states")
    print("   • |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print("   • North pole: |0⟩, South pole: |1⟩")
    print("   • Equator: Equal superposition states")
    print("   • Any point on surface is a pure state")
    
    print("\n3. SINGLE QUBIT ROTATIONS:")
    print("   • Rx(θ): Rotation around X-axis")
    print("   • Ry(θ): Rotation around Y-axis")
    print("   • Rz(θ): Rotation around Z-axis")
    print("   • Pauli gates: 180° rotations (θ = π)")
    print("   • Any single-qubit gate is a rotation")
    
    print("\n4. MEASUREMENT:")
    print("   • Probabilistic collapse to basis states")
    print("   • P(|0⟩) = |α|², P(|1⟩) = |β|²")
    print("   • Can measure in any orthonormal basis")
    print("   • Measurement destroys superposition")
    print("   • Outcome is random but probabilities fixed")
    
    print("\n5. QUANTUM ERRORS:")
    print("   • Bit flip: |0⟩ ↔ |1⟩")
    print("   • Phase flip: Relative phase changes")
    print("   • Depolarizing: State becomes mixed")
    print("   • Amplitude damping: Energy relaxation (T₁)")
    print("   • Phase damping: Coherence loss (T₂)")
    
    print("\n6. STATE TOMOGRAPHY:")
    print("   • Reconstruct unknown state from measurements")
    print("   • Measure in Z, X, and Y bases")
    print("   • Need many copies of the state")
    print("   • Extract Bloch coordinates from statistics")
    print("   • Cannot determine global phase")
    
    print("\n7. BENCHMARKING:")
    print("   • Randomized benchmarking: Average gate fidelity")
    print("   • Gate fidelity: F = |⟨ψ_ideal|ψ_actual⟩|²")
    print("   • Typical values: F > 99% for good qubits")
    print("   • Process tomography: Full gate characterization")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("SINGLE QUBIT OPERATIONS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates single qubit operations, Bloch sphere")
    print("representation, rotations, measurements, and error models.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating Bloch sphere states visualization...")
    plot_bloch_sphere_states()
    
    print("\n2. Creating single qubit rotations visualization...")
    plot_single_qubit_rotations()
    
    print("\n3. Creating measurement bases visualization...")
    plot_measurement_bases()
    
    print("\n4. Creating quantum errors visualization...")
    plot_quantum_errors()
    
    print("\n5. Creating state tomography visualization...")
    plot_state_tomography()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • bloch_sphere_states.png")
    print("  • single_qubit_rotations.png")
    print("  • measurement_bases.png")
    print("  • quantum_errors.png")
    print("  • state_tomography.png")
    print("\nThese visualizations demonstrate single qubit operations and their")
    print("properties in quantum computing.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
