#!/usr/bin/env python3
"""
Bloch Sphere - Comprehensive Python Demonstrations

This script demonstrates the Bloch sphere representation of single qubit states,
including visualization, rotations, measurements, and properties.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


def plot_bloch_sphere_basics():
    """
    Visualize the basic Bloch sphere with special states.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Bloch Sphere: Basic Visualization', fontsize=16, fontweight='bold')
    
    special_states = {
        '|0⟩': (0, 0),
        '|1⟩': (np.pi, 0),
        '|+⟩': (np.pi/2, 0),
        '|−⟩': (np.pi/2, np.pi),
        '|+i⟩': (np.pi/2, np.pi/2),
        '|−i⟩': (np.pi/2, 3*np.pi/2)
    }
    
    for idx, (name, (theta, phi)) in enumerate(special_states.items()):
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
        
        equator_x = np.cos(u)
        equator_y = np.sin(u)
        equator_z = np.zeros_like(u)
        ax.plot(equator_x, equator_y, equator_z, 'b--', alpha=0.5, linewidth=1)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.15, linewidth=3)
        
        ax.text(0, 0, 1.3, '|0⟩', fontsize=10, color='blue')
        ax.text(0, 0, -1.3, '|1⟩', fontsize=10, color='blue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'State: {name}')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_basics.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_basics.png")
    plt.show()


def plot_bloch_sphere_rotations():
    """
    Visualize quantum gate operations as rotations on the Bloch sphere.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Quantum Gates as Bloch Sphere Rotations', fontsize=16, fontweight='bold')
    
    gates = [
        ('X Gate', X, 'X-axis', 'red'),
        ('Y Gate', Y, 'Y-axis', 'green'),
        ('Z Gate', Z, 'Z-axis', 'blue'),
        ('H Gate', H, 'Combined', 'purple')
    ]
    
    for idx, (gate_name, gate, axis_name, color) in enumerate(gates):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        n_steps = 20
        initial_state = ket0
        trajectory_x, trajectory_y, trajectory_z = [], [], []
        
        for i in range(n_steps + 1):
            t = i / n_steps
            angle = t * np.pi  # 180 degree rotation
            if gate_name == 'X Gate':
                R = np.cos(angle/2) * I - 1j * np.sin(angle/2) * X
            elif gate_name == 'Y Gate':
                R = np.cos(angle/2) * I - 1j * np.sin(angle/2) * Y
            elif gate_name == 'Z Gate':
                R = np.cos(angle/2) * I - 1j * np.sin(angle/2) * Z
            else:  # Hadamard
                R = gate if t > 0.5 else I
            
            rotated_state = R @ initial_state
            x, y, z = state_to_bloch(rotated_state)
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_z.append(z)
        
        ax.plot(trajectory_x, trajectory_y, trajectory_z, color=color, linewidth=3, label='Rotation path')
        
        ax.quiver(0, 0, 0, trajectory_x[0], trajectory_y[0], trajectory_z[0], 
                 color='blue', arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
        ax.quiver(0, 0, 0, trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], 
                 color='red', arrow_length_ratio=0.15, linewidth=2)
        
        if axis_name == 'X-axis':
            ax.plot([-1.5, 1.5], [0, 0], [0, 0], color=color, linewidth=3, alpha=0.7, linestyle='--')
        elif axis_name == 'Y-axis':
            ax.plot([0, 0], [-1.5, 1.5], [0, 0], color=color, linewidth=3, alpha=0.7, linestyle='--')
        elif axis_name == 'Z-axis':
            ax.plot([0, 0], [0, 0], [-1.5, 1.5], color=color, linewidth=3, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{gate_name}: Rotation around {axis_name}')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_rotations.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_rotations.png")
    plt.show()


def plot_bloch_sphere_measurements():
    """
    Visualize measurement in different bases on the Bloch sphere.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Measurement on the Bloch Sphere', fontsize=16, fontweight='bold')
    
    test_theta = np.pi / 3
    test_phi = np.pi / 4
    test_state = bloch_to_state(test_theta, test_phi)
    
    measurement_bases = [
        ('Z-basis (Computational)', [(0, 0), (np.pi, 0)], ['|0⟩', '|1⟩'], 'blue'),
        ('X-basis', [(np.pi/2, 0), (np.pi/2, np.pi)], ['|+⟩', '|−⟩'], 'red'),
        ('Y-basis', [(np.pi/2, np.pi/2), (np.pi/2, 3*np.pi/2)], ['|+i⟩', '|−i⟩'], 'green')
    ]
    
    for idx, (basis_name, basis_states, labels, color) in enumerate(measurement_bases):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        x_test, y_test, z_test = state_to_bloch(test_state)
        ax.quiver(0, 0, 0, x_test, y_test, z_test, 
                 color='purple', arrow_length_ratio=0.15, linewidth=3, label='Test state')
        
        for (theta, phi), label in zip(basis_states, labels):
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            ax.quiver(0, 0, 0, x, y, z, 
                     color=color, arrow_length_ratio=0.15, linewidth=2, alpha=0.7)
            ax.text(x*1.3, y*1.3, z*1.3, label, fontsize=9, color=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{basis_name} Measurement')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.legend()
        
        ax2 = fig.add_subplot(2, 3, idx + 4)
        
        probs = []
        for (theta, phi) in basis_states:
            basis_state = bloch_to_state(theta, phi)
            prob = np.abs(np.vdot(basis_state, test_state))**2
            probs.append(prob)
        
        bars = ax2.bar(labels, probs, color=color, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'{basis_name} Probabilities')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_measurements.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_measurements.png")
    plt.show()


def plot_bloch_sphere_coverage():
    """
    Visualize coverage of the Bloch sphere with random states.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Bloch Sphere State Space Coverage', fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    
    n_states = 200
    theta_random = np.arccos(1 - 2*np.random.rand(n_states))
    phi_random = 2 * np.pi * np.random.rand(n_states)
    
    x_random = np.sin(theta_random) * np.cos(phi_random)
    y_random = np.sin(theta_random) * np.sin(phi_random)
    z_random = np.cos(theta_random)
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    ax.scatter(x_random, y_random, z_random, c='red', s=10, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Uniform Random Pure States')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    
    r_mixed = np.random.rand(n_states)  # Radius < 1
    theta_mixed = np.arccos(1 - 2*np.random.rand(n_states))
    phi_mixed = 2 * np.pi * np.random.rand(n_states)
    
    x_mixed = r_mixed * np.sin(theta_mixed) * np.cos(phi_mixed)
    y_mixed = r_mixed * np.sin(theta_mixed) * np.sin(phi_mixed)
    z_mixed = r_mixed * np.cos(theta_mixed)
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    ax.scatter(x_mixed, y_mixed, z_mixed, c=r_mixed, cmap='viridis', s=10, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mixed States (Interior Points)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    ax = fig.add_subplot(2, 2, 3)
    
    theta_bins = np.linspace(0, np.pi, 30)
    phi_bins = np.linspace(0, 2*np.pi, 60)
    
    hist, _, _ = np.histogram2d(theta_random, phi_random, bins=[theta_bins, phi_bins])
    
    im = ax.imshow(hist.T, extent=[0, 180, 0, 360], aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('φ (degrees)')
    ax.set_title('State Density on Bloch Sphere')
    plt.colorbar(im, ax=ax, label='Count')
    
    ax = fig.add_subplot(2, 2, 4)
    
    radii = np.linspace(0, 1, 100)
    purity = radii**2  # For single qubit: Tr(ρ²) = (1 + r²)/2, normalized to [0,1]
    
    ax.plot(radii, purity, 'b-', linewidth=2)
    ax.set_xlabel('Bloch Vector Length |r|')
    ax.set_ylabel('Purity')
    ax.set_title('Purity vs Bloch Vector Length')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax.plot(1, 1, 'ro', markersize=10, label='Pure state (surface)')
    ax.plot(0, 0, 'go', markersize=10, label='Maximally mixed (center)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_coverage.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_coverage.png")
    plt.show()


def plot_bloch_sphere_dynamics():
    """
    Visualize time evolution and dynamics on the Bloch sphere.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Quantum Dynamics on the Bloch Sphere', fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    
    initial_state = (ket0 + ket1) / np.sqrt(2)
    
    n_steps = 50
    omega = 2 * np.pi  # Frequency
    times = np.linspace(0, 2*np.pi/omega, n_steps)
    
    trajectory_x, trajectory_y, trajectory_z = [], [], []
    for t in times:
        phase = omega * t
        evolved_state = np.array([[initial_state[0, 0]], 
                                 [np.exp(-1j * phase) * initial_state[1, 0]]], dtype=complex)
        x, y, z = state_to_bloch(evolved_state)
        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=2, label='Precession')
    ax.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0], c='blue', s=100, marker='o')
    
    ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'b--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Free Precession (Z-axis)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.legend()
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    
    trajectory_x2, trajectory_y2, trajectory_z2 = [], [], []
    for t in times:
        angle = omega * t
        R = np.cos(angle/2) * I - 1j * np.sin(angle/2) * X
        evolved_state = R @ ket0
        x, y, z = state_to_bloch(evolved_state)
        trajectory_x2.append(x)
        trajectory_y2.append(y)
        trajectory_z2.append(z)
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    ax.plot(trajectory_x2, trajectory_y2, trajectory_z2, 'g-', linewidth=2, label='Rabi oscillation')
    ax.scatter(trajectory_x2[0], trajectory_y2[0], trajectory_z2[0], c='blue', s=100, marker='o')
    
    ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'r--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rabi Oscillations (X-axis)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.legend()
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    
    T2 = 10  # Decoherence time
    trajectory_x3, trajectory_y3, trajectory_z3 = [], [], []
    
    for t in times:
        phase = omega * t
        decay = np.exp(-t / T2)
        x = decay * np.cos(phase)
        y = decay * np.sin(phase)
        z = 0
        trajectory_x3.append(x)
        trajectory_y3.append(y)
        trajectory_z3.append(z)
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    ax.plot(trajectory_x3, trajectory_y3, trajectory_z3, 'purple', linewidth=2, label='Decoherence')
    ax.scatter(trajectory_x3[0], trajectory_y3[0], trajectory_z3[0], c='blue', s=100, marker='o')
    ax.scatter(0, 0, 0, c='red', s=100, marker='x', label='Mixed state')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Decoherence (Spiral to Center)')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.legend()
    
    ax = fig.add_subplot(2, 2, 4)
    
    pop_0 = [(1 + z) / 2 for z in trajectory_z2]
    pop_1 = [(1 - z) / 2 for z in trajectory_z2]
    
    ax.plot(times, pop_0, 'b-', linewidth=2, label='P(|0⟩)')
    ax.plot(times, pop_1, 'r-', linewidth=2, label='P(|1⟩)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('Rabi Oscillations: Population vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_dynamics.png', dpi=300, bbox_inches='tight')
    print("Saved: bloch_sphere_dynamics.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about the Bloch sphere.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: The Bloch Sphere")
    print("="*80)
    
    print("\n1. BLOCH SPHERE REPRESENTATION:")
    print("   • Geometric representation of single qubit pure states")
    print("   • Every point on surface = unique quantum state")
    print("   • Unit sphere in 3D: x² + y² + z² = 1")
    print("   • Parameterized by angles θ (polar) and φ (azimuthal)")
    
    print("\n2. MATHEMATICAL FORM:")
    print("   • |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print("   • Bloch coordinates: x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)")
    print("   • θ ∈ [0, π], φ ∈ [0, 2π)")
    print("   • Global phase not represented (physically unobservable)")
    
    print("\n3. SPECIAL POINTS:")
    print("   • |0⟩: North pole (0, 0, 1)")
    print("   • |1⟩: South pole (0, 0, -1)")
    print("   • |+⟩: Positive X-axis (1, 0, 0)")
    print("   • |−⟩: Negative X-axis (-1, 0, 0)")
    print("   • |+i⟩: Positive Y-axis (0, 1, 0)")
    print("   • |−i⟩: Negative Y-axis (0, -1, 0)")
    
    print("\n4. QUANTUM GATES AS ROTATIONS:")
    print("   • X gate: 180° rotation around X-axis")
    print("   • Y gate: 180° rotation around Y-axis")
    print("   • Z gate: 180° rotation around Z-axis")
    print("   • H gate: Combination of rotations")
    print("   • Rx(θ), Ry(θ), Rz(θ): Arbitrary rotations")
    
    print("\n5. MEASUREMENT:")
    print("   • Projects state onto measurement axis")
    print("   • Z-basis: P(|0⟩) = (1+z)/2, P(|1⟩) = (1-z)/2")
    print("   • X-basis: P(|+⟩) = (1+x)/2, P(|−⟩) = (1-x)/2")
    print("   • Y-basis: P(|+i⟩) = (1+y)/2, P(|−i⟩) = (1-y)/2")
    
    print("\n6. PURE VS MIXED STATES:")
    print("   • Pure states: On surface (|r| = 1)")
    print("   • Mixed states: Inside sphere (|r| < 1)")
    print("   • Maximally mixed: Center (r = 0)")
    print("   • Purity decreases toward center")
    
    print("\n7. PROPERTIES:")
    print("   • Antipodal points: Orthogonal states")
    print("   • Distance: Related to fidelity")
    print("   • Rotations: Unitary operations")
    print("   • Decoherence: Motion toward center")
    
    print("\n8. LIMITATIONS:")
    print("   • Single qubit only (no multi-qubit generalization)")
    print("   • Cannot represent entanglement")
    print("   • Pure states only on surface")
    print("   • Global phase not shown")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("BLOCH SPHERE - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates the Bloch sphere representation of")
    print("single qubit states, including visualization, rotations,")
    print("measurements, and properties.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating Bloch sphere basics visualization...")
    plot_bloch_sphere_basics()
    
    print("\n2. Creating Bloch sphere rotations visualization...")
    plot_bloch_sphere_rotations()
    
    print("\n3. Creating Bloch sphere measurements visualization...")
    plot_bloch_sphere_measurements()
    
    print("\n4. Creating Bloch sphere coverage visualization...")
    plot_bloch_sphere_coverage()
    
    print("\n5. Creating Bloch sphere dynamics visualization...")
    plot_bloch_sphere_dynamics()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • bloch_sphere_basics.png")
    print("  • bloch_sphere_rotations.png")
    print("  • bloch_sphere_measurements.png")
    print("  • bloch_sphere_coverage.png")
    print("  • bloch_sphere_dynamics.png")
    print("\nThese visualizations demonstrate the Bloch sphere representation")
    print("and its role in understanding single qubit quantum mechanics.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
