"""
Introduction to Qubits - Python Demonstrations
==============================================

This script provides comprehensive demonstrations of qubit concepts including:
- Classical bit vs quantum qubit comparison
- Superposition states
- Measurement probabilities
- Bloch sphere visualization
- Basic qubit operations

Author: QpiAI Quantum Engineer Course
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

def create_qubit_state(alpha, beta):
    """
    Create a normalized qubit state |ψ⟩ = α|0⟩ + β|1⟩
    
    Parameters:
    -----------
    alpha : complex
        Amplitude for |0⟩ state
    beta : complex
        Amplitude for |1⟩ state
    
    Returns:
    --------
    numpy.ndarray
        Normalized qubit state vector
    """
    state = alpha * ket_0 + beta * ket_1
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    return state / norm

def measure_qubit(state, num_measurements=1000):
    """
    Simulate measurement of a qubit state
    
    Parameters:
    -----------
    state : numpy.ndarray
        Qubit state vector
    num_measurements : int
        Number of measurements to simulate
    
    Returns:
    --------
    tuple
        (probability_0, probability_1, measurement_results)
    """
    prob_0 = np.abs(state[0, 0])**2
    prob_1 = np.abs(state[1, 0])**2
    
    results = np.random.choice([0, 1], size=num_measurements, p=[prob_0, prob_1])
    
    return prob_0, prob_1, results

def bloch_coordinates(theta, phi):
    """
    Convert spherical coordinates to Cartesian for Bloch sphere
    
    Parameters:
    -----------
    theta : float
        Polar angle (0 to π)
    phi : float
        Azimuthal angle (0 to 2π)
    
    Returns:
    --------
    tuple
        (x, y, z) Cartesian coordinates
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def plot_classical_vs_quantum():
    """
    Visualize the difference between classical bits and quantum qubits
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Classical Bit\n(Discrete States)', fontsize=16, fontweight='bold', pad=20)
    
    circle0 = plt.Circle((0, 1), 0.3, color='#ff6b6b', alpha=0.8)
    ax1.add_patch(circle0)
    ax1.text(0, 1, '0', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax1.text(0, 0.5, 'State |0⟩', ha='center', fontsize=12)
    
    circle1 = plt.Circle((1, 1), 0.3, color='#4ecdc4', alpha=0.8)
    ax1.add_patch(circle1)
    ax1.text(1, 1, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax1.text(1, 0.5, 'State |1⟩', ha='center', fontsize=12)
    
    ax1.annotate('', xy=(0.7, 1), xytext=(0.3, 1),
                arrowprops=dict(arrowstyle='<->', lw=2, color='gray'))
    ax1.text(0.5, 1.2, 'Discrete\nTransition', ha='center', fontsize=10, color='gray')
    
    ax2 = plt.subplot(122, projection='3d')
    ax2.set_title('Quantum Qubit\n(Continuous States)', fontsize=16, fontweight='bold', pad=20)
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    ax2.plot([0, 0], [0, 0], [-1.5, 1.5], 'k-', lw=2, label='Z-axis')
    ax2.plot([0, 1.5], [0, 0], [0, 0], 'r-', lw=2, label='X-axis')
    ax2.plot([0, 0], [0, 1.5], [0, 0], 'g-', lw=2, label='Y-axis')
    
    ax2.scatter([0], [0], [1], color='#ff6b6b', s=200, marker='o', label='|0⟩')
    ax2.scatter([0], [0], [-1], color='#4ecdc4', s=200, marker='o', label='|1⟩')
    
    angles = [(np.pi/4, 0), (np.pi/2, 0), (np.pi/2, np.pi/2), (np.pi/3, np.pi/4)]
    for theta, phi in angles:
        x, y, z = bloch_coordinates(theta, phi)
        ax2.scatter([x], [y], [z], color='#64ffda', s=100, alpha=0.6)
        ax2.plot([0, x], [0, y], [0, z], 'b--', alpha=0.3, lw=1)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('classical_vs_quantum.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: classical_vs_quantum.png")
    plt.close()

def plot_superposition_states():
    """
    Visualize various superposition states
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Qubit Superposition States', fontsize=18, fontweight='bold', y=1.02)
    
    states = [
        (1, 0, '|0⟩', 'Pure |0⟩ state'),
        (0, 1, '|1⟩', 'Pure |1⟩ state'),
        (1/np.sqrt(2), 1/np.sqrt(2), '|+⟩', 'Equal superposition'),
        (1/np.sqrt(2), -1/np.sqrt(2), '|−⟩', 'Equal superposition (negative)'),
        (1/np.sqrt(2), 1j/np.sqrt(2), '|+i⟩', 'Complex superposition'),
        (np.sqrt(0.7), np.sqrt(0.3), '|ψ⟩', 'General superposition')
    ]
    
    for idx, (alpha, beta, label, description) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        
        state = create_qubit_state(alpha, beta)
        prob_0, prob_1, _ = measure_qubit(state, 1000)
        
        bars = ax.bar(['|0⟩', '|1⟩'], [prob_0, prob_1], 
                     color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'{label}\n{description}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, prob in zip(bars, [prob_0, prob_1]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.text(0.5, 0.95, f'α = {alpha:.3f}\nβ = {beta:.3f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    plt.tight_layout()
    plt.savefig('superposition_states.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: superposition_states.png")
    plt.close()

def plot_measurement_statistics():
    """
    Demonstrate measurement statistics for a superposition state
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum Measurement Statistics', fontsize=18, fontweight='bold')
    
    alpha = 1/np.sqrt(2)
    beta = 1/np.sqrt(2)
    state = create_qubit_state(alpha, beta)
    
    sample_sizes = [10, 100, 1000, 10000]
    
    for idx, num_measurements in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]
        
        prob_0, prob_1, results = measure_qubit(state, num_measurements)
        
        count_0 = np.sum(results == 0)
        count_1 = np.sum(results == 1)
        
        bars = ax.bar(['|0⟩', '|1⟩'], [count_0, count_1], 
                     color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Measurements: {num_measurements}\n' + 
                    f'Observed: {count_0/num_measurements:.3f} |0⟩, {count_1/num_measurements:.3f} |1⟩',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax.axhline(y=num_measurements*0.5, color='green', linestyle='--', 
                  linewidth=2, label='Expected (0.5)')
        ax.legend()
        
        for bar, count in zip(bars, [count_0, count_1]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('measurement_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: measurement_statistics.png")
    plt.close()

def plot_bloch_sphere_states():
    """
    Visualize multiple qubit states on the Bloch sphere
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Qubit States on Bloch Sphere', fontsize=18, fontweight='bold', pad=20)
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    
    axis_length = 1.3
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', lw=3, label='X-axis')
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', lw=3, label='Y-axis')
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', lw=3, label='Z-axis')
    
    theta_eq = np.pi/2
    phi_eq = np.linspace(0, 2*np.pi, 100)
    x_eq = np.sin(theta_eq) * np.cos(phi_eq)
    y_eq = np.sin(theta_eq) * np.sin(phi_eq)
    z_eq = np.cos(theta_eq) * np.ones_like(phi_eq)
    ax.plot(x_eq, y_eq, z_eq, 'gray', lw=2, linestyle='--', alpha=0.5)
    
    states = [
        (0, 0, '|0⟩', '#ff6b6b', 200),           # North pole
        (np.pi, 0, '|1⟩', '#4ecdc4', 200),       # South pole
        (np.pi/2, 0, '|+⟩', '#64ffda', 150),     # +X
        (np.pi/2, np.pi, '|−⟩', '#ffd93d', 150), # -X
        (np.pi/2, np.pi/2, '|+i⟩', '#a8e6cf', 150),   # +Y
        (np.pi/2, 3*np.pi/2, '|−i⟩', '#ff9999', 150), # -Y
    ]
    
    for theta, phi, label, color, size in states:
        x, y, z = bloch_coordinates(theta, phi)
        ax.scatter([x], [y], [z], color=color, s=size, marker='o', 
                  edgecolors='black', linewidths=2, label=label, alpha=0.9)
        
        ax.plot([0, x], [0, y], [0, z], color=color, lw=2, alpha=0.6)
        
        ax.text(x*1.2, y*1.2, z*1.2, label, fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    for _ in range(10):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        x, y, z = bloch_coordinates(theta, phi)
        ax.scatter([x], [y], [z], color='purple', s=50, alpha=0.3, marker='.')
    
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1,1,1])
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('bloch_sphere_states.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: bloch_sphere_states.png")
    plt.close()

def plot_quantum_properties():
    """
    Visualize key quantum properties
    """
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Property 1: Superposition', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.7, 'Classical Bit', ha='center', fontsize=12, fontweight='bold',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.6, '0 OR 1', ha='center', fontsize=16, color='red',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.4, 'Quantum Qubit', ha='center', fontsize=12, fontweight='bold',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.3, 'α|0⟩ + β|1⟩', ha='center', fontsize=16, color='blue',
            transform=ax1.transAxes)
    ax1.text(0.5, 0.15, '(Both states simultaneously)', ha='center', fontsize=10,
            style='italic', transform=ax1.transAxes)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Property 2: Measurement Collapse', fontsize=14, fontweight='bold')
    
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.3
    x_circle = 0.3 + r * np.cos(theta)
    y_circle = 0.7 + r * np.sin(theta)
    ax2.fill(x_circle, y_circle, color='purple', alpha=0.5, label='Superposition')
    ax2.text(0.3, 0.7, '|ψ⟩', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax2.arrow(0.5, 0.7, 0.15, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax2.text(0.575, 0.75, 'Measure', ha='center', fontsize=10)
    
    ax2.scatter([0.8], [0.85], s=500, color='#ff6b6b', alpha=0.8, edgecolors='black', linewidths=2)
    ax2.text(0.8, 0.85, '|0⟩', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(0.8, 0.95, f'P = |α|²', ha='center', fontsize=9)
    
    ax2.scatter([0.8], [0.55], s=500, color='#4ecdc4', alpha=0.8, edgecolors='black', linewidths=2)
    ax2.text(0.8, 0.55, '|1⟩', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(0.8, 0.45, f'P = |β|²', ha='center', fontsize=9)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.3, 1)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Property 3: Intrinsic Parallelism', fontsize=14, fontweight='bold')
    
    ax3.text(0.1, 0.8, 'Classical:', ha='left', fontsize=12, fontweight='bold',
            transform=ax3.transAxes)
    ax3.text(0.1, 0.7, 'Process one path at a time', ha='left', fontsize=10,
            transform=ax3.transAxes)
    
    ax3.text(0.1, 0.5, 'Quantum:', ha='left', fontsize=12, fontweight='bold',
            transform=ax3.transAxes)
    ax3.text(0.1, 0.4, 'Process all paths simultaneously', ha='left', fontsize=10,
            transform=ax3.transAxes)
    ax3.text(0.1, 0.3, 'via superposition', ha='left', fontsize=10, style='italic',
            transform=ax3.transAxes)
    
    for i, y in enumerate([0.65, 0.60, 0.55]):
        ax3.arrow(0.6, y, 0.25, 0, head_width=0.02, head_length=0.03,
                 fc='gray', ec='gray', alpha=0.5, transform=ax3.transAxes)
    
    for i, y in enumerate([0.25, 0.20, 0.15, 0.10, 0.05]):
        ax3.arrow(0.6, y, 0.25, 0, head_width=0.02, head_length=0.03,
                 fc='blue', ec='blue', alpha=0.7, transform=ax3.transAxes)
    
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Property 4: Entanglement', fontsize=14, fontweight='bold')
    
    ax4.scatter([0.3], [0.5], s=800, color='#ff6b6b', alpha=0.6, edgecolors='black', linewidths=2)
    ax4.text(0.3, 0.5, 'Qubit A', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax4.scatter([0.7], [0.5], s=800, color='#4ecdc4', alpha=0.6, edgecolors='black', linewidths=2)
    ax4.text(0.7, 0.5, 'Qubit B', ha='center', va='center', fontsize=11, fontweight='bold')
    
    theta = np.linspace(0, np.pi, 50)
    x_curve = 0.3 + 0.4 * (theta / np.pi)
    y_curve = 0.5 + 0.15 * np.sin(theta)
    ax4.plot(x_curve, y_curve, 'purple', lw=3, alpha=0.7)
    ax4.text(0.5, 0.65, 'Entangled', ha='center', fontsize=11, 
            fontweight='bold', color='purple')
    
    ax4.text(0.5, 0.3, 'Measuring A instantly\naffects state of B', ha='center',
            fontsize=10, style='italic', transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0.2, 0.8)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('quantum_properties.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: quantum_properties.png")
    plt.close()

def print_key_concepts():
    """
    Print key concepts about qubits
    """
    print("\n" + "="*70)
    print("KEY CONCEPTS: INTRODUCTION TO QUBITS")
    print("="*70)
    
    print("\n1. WHAT IS A QUBIT?")
    print("   - Fundamental unit of quantum information")
    print("   - Quantum analog of classical bit")
    print("   - Two-level quantum system with quantum properties")
    
    print("\n2. QUANTUM PROPERTIES:")
    print("   a) Superposition:")
    print("      • Qubit can exist in multiple states simultaneously")
    print("      • |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1")
    print("   b) Entanglement:")
    print("      • Qubits can be correlated in non-classical ways")
    print("      • Measuring one affects the state of others")
    print("   c) Measurement:")
    print("      • Collapses superposition to definite state")
    print("      • Probabilistic outcome: P(0) = |α|², P(1) = |β|²")
    print("   d) Intrinsic Parallelism:")
    print("      • Process multiple computational paths simultaneously")
    
    print("\n3. MATHEMATICAL REPRESENTATION:")
    print("   • State vector: |ψ⟩ = α|0⟩ + β|1⟩")
    print("   • Basis states: |0⟩ = [1, 0]ᵀ, |1⟩ = [0, 1]ᵀ")
    print("   • Normalization: |α|² + |β|² = 1")
    print("   • Complex amplitudes: α, β ∈ ℂ")
    
    print("\n4. BLOCH SPHERE:")
    print("   • Geometric representation of qubit states")
    print("   • |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩")
    print("   • North pole: |0⟩, South pole: |1⟩")
    print("   • Equator: Equal superposition states")
    
    print("\n5. CLASSICAL BIT VS QUANTUM QUBIT:")
    print("   Classical Bit:")
    print("   • States: 0 OR 1 (discrete)")
    print("   • Deterministic operations")
    print("   • Can be copied")
    print("   Quantum Qubit:")
    print("   • States: α|0⟩ + β|1⟩ (continuous)")
    print("   • Probabilistic measurement")
    print("   • Cannot be cloned (No-cloning theorem)")
    
    print("\n6. APPLICATIONS:")
    print("   • Quantum computing and algorithms")
    print("   • Quantum cryptography and communication")
    print("   • Quantum simulation")
    print("   • Quantum sensing and metrology")
    
    print("\n" + "="*70)

def main():
    """
    Main function to run all demonstrations
    """
    print("\n" + "="*70)
    print("INTRODUCTION TO QUBITS - PYTHON DEMONSTRATIONS")
    print("="*70)
    
    print("\nGenerating visualizations...")
    print("-" * 70)
    
    plot_classical_vs_quantum()
    plot_superposition_states()
    plot_measurement_statistics()
    plot_bloch_sphere_states()
    plot_quantum_properties()
    
    print_key_concepts()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. classical_vs_quantum.png - Comparison of classical bits and quantum qubits")
    print("  2. superposition_states.png - Various superposition states")
    print("  3. measurement_statistics.png - Measurement statistics demonstration")
    print("  4. bloch_sphere_states.png - Qubit states on Bloch sphere")
    print("  5. quantum_properties.png - Key quantum properties visualization")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
