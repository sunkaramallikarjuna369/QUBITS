#!/usr/bin/env python3
"""
Summary - Comprehensive Review of Qubits Lecture

This script provides a comprehensive summary of all 13 concepts covered in the
Qubits lecture series, with visualizations and key takeaways.

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


def plot_concept_overview():
    """
    Visualize the 13 concepts and their relationships.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.suptitle('Qubits Lecture: 13 Concepts Overview', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    concepts = [
        (1, "Introduction", "Foundation of quantum computing", 5, 13),
        (2, "Complex Numbers", "Mathematical foundation", 2, 11),
        (3, "Vectors", "Hilbert space structure", 5, 11),
        (4, "Bit vs Qubit", "Classical vs quantum", 8, 11),
        (5, "Qubit from Bit", "Building quantum states", 2, 9),
        (6, "Mathematical View", "Rigorous formalism", 5, 9),
        (7, "Physical View", "Hardware implementations", 8, 9),
        (8, "Computational View", "Gates and circuits", 2, 7),
        (9, "Single Qubit", "One qubit operations", 5, 7),
        (10, "Multi-Qubits", "Multiple qubit systems", 8, 7),
        (11, "Entangled Qubits", "Quantum correlations", 2, 5),
        (12, "Bloch Sphere", "Geometric representation", 5, 5),
        (13, "Summary", "Comprehensive review", 5, 2)
    ]
    
    for num, name, desc, x, y in concepts:
        box = FancyBboxPatch((x-1, y-0.4), 2, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightblue', 
                            edgecolor='black', 
                            linewidth=2)
        ax.add_patch(box)
        
        ax.text(x, y+0.1, f"{num}. {name}", ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x, y-0.2, desc, ha='center', va='center', 
               fontsize=8, style='italic')
    
    connections = [
        (5, 13, 2, 11),  # Introduction to Complex Numbers
        (5, 13, 5, 11),  # Introduction to Vectors
        (5, 13, 8, 11),  # Introduction to Bit vs Qubit
        (2, 11, 2, 9),   # Complex Numbers to Qubit from Bit
        (5, 11, 5, 9),   # Vectors to Mathematical View
        (8, 11, 8, 9),   # Bit vs Qubit to Physical View
        (2, 9, 2, 7),    # Qubit from Bit to Computational View
        (5, 9, 5, 7),    # Mathematical View to Single Qubit
        (8, 9, 8, 7),    # Physical View to Multi-Qubits
        (2, 7, 2, 5),    # Computational View to Entangled Qubits
        (5, 7, 5, 5),    # Single Qubit to Bloch Sphere
        (8, 7, 2, 5),    # Multi-Qubits to Entangled Qubits
        (2, 5, 5, 2),    # Entangled Qubits to Summary
        (5, 5, 5, 2),    # Bloch Sphere to Summary
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5))
    
    ax.text(0.5, 12, 'Foundation', fontsize=12, fontweight='bold', color='blue')
    ax.text(0.5, 10, 'Mathematical\nFramework', fontsize=12, fontweight='bold', color='green')
    ax.text(0.5, 8, 'Perspectives', fontsize=12, fontweight='bold', color='purple')
    ax.text(0.5, 6, 'Advanced\nConcepts', fontsize=12, fontweight='bold', color='red')
    ax.text(0.5, 3, 'Integration', fontsize=12, fontweight='bold', color='orange')
    
    plt.tight_layout()
    plt.savefig('concept_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: concept_overview.png")
    plt.show()


def plot_key_principles():
    """
    Visualize the core principles of quantum computing.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Core Principles of Quantum Computing', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    ax.bar([0], [1], width=0.3, color='red', alpha=0.7, label='Classical: |0⟩ OR |1⟩')
    
    ax.bar([1, 1.3], [0.5, 0.5], width=0.3, color='blue', alpha=0.7, label='Quantum: |0⟩ AND |1⟩')
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Superposition')
    ax.set_xticks([0, 1.15])
    ax.set_xticklabels(['Classical', 'Quantum'])
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    
    separable = [0.25, 0.25, 0.25, 0.25]
    entangled = [0.5, 0, 0, 0.5]
    
    x = np.arange(4)
    width = 0.35
    
    ax.bar(x - width/2, separable, width, label='Separable', color='lightblue', alpha=0.7)
    ax.bar(x + width/2, entangled, width, label='Entangled', color='lightcoral', alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('Entanglement')
    ax.set_xticks(x)
    ax.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 2]
    
    x = np.linspace(0, 4*np.pi, 100)
    wave1 = np.sin(x)
    wave2 = np.sin(x + np.pi/4)
    constructive = wave1 + wave2
    destructive = wave1 - wave2
    
    ax.plot(x, wave1, 'b--', alpha=0.5, label='Wave 1')
    ax.plot(x, wave2, 'r--', alpha=0.5, label='Wave 2')
    ax.plot(x, constructive, 'g-', linewidth=2, label='Constructive')
    ax.plot(x, destructive, 'm-', linewidth=2, label='Destructive')
    
    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')
    ax.set_title('Interference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax = axes[1, 0]
    
    before = [0.3, 0.7]
    after = [0, 1]
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, before, width, label='Before', color='lightgreen', alpha=0.7)
    ax.bar(x + width/2, after, width, label='After', color='orange', alpha=0.7)
    
    ax.set_ylabel('Probability')
    ax.set_title('Measurement Collapse')
    ax.set_xticks(x)
    ax.set_xticklabels(['|0⟩', '|1⟩'])
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    
    angles = np.linspace(0, 2*np.pi, 100)
    prob_0 = np.cos(angles/2)**2
    prob_1 = np.sin(angles/2)**2
    total = prob_0 + prob_1
    
    ax.plot(angles, prob_0, 'b-', linewidth=2, label='P(|0⟩)')
    ax.plot(angles, prob_1, 'r-', linewidth=2, label='P(|1⟩)')
    ax.plot(angles, total, 'g--', linewidth=2, label='Total')
    
    ax.set_xlabel('Rotation Angle')
    ax.set_ylabel('Probability')
    ax.set_title('Unitarity (Probability Conservation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    ax = axes[1, 2]
    
    n_qubits = np.arange(1, 21)
    classical = n_qubits
    quantum = 2**n_qubits
    
    ax.semilogy(n_qubits, classical, 'r-', linewidth=2, marker='s', label='Classical')
    ax.semilogy(n_qubits, quantum, 'b-', linewidth=2, marker='o', label='Quantum')
    
    ax.set_xlabel('Number of Bits/Qubits')
    ax.set_ylabel('State Space Size (log scale)')
    ax.set_title('Exponential Quantum Advantage')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('key_principles.png', dpi=300, bbox_inches='tight')
    print("Saved: key_principles.png")
    plt.show()


def plot_quantum_gates_summary():
    """
    Summarize quantum gates and their properties.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Gates Summary', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    gates = ['X', 'Y', 'Z', 'H', 'S', 'T']
    properties = {
        'Pauli': [1, 1, 1, 0, 0, 0],
        'Clifford': [1, 1, 1, 1, 1, 0],
        'Universal': [0, 0, 0, 0, 0, 1]
    }
    
    x = np.arange(len(gates))
    width = 0.25
    
    for i, (prop, values) in enumerate(properties.items()):
        ax.bar(x + i*width, values, width, label=prop, alpha=0.7)
    
    ax.set_ylabel('Property')
    ax.set_title('Single-Qubit Gate Properties')
    ax.set_xticks(x + width)
    ax.set_xticklabels(gates)
    ax.legend()
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    
    two_qubit_gates = ['CNOT', 'CZ', 'SWAP', 'Toffoli']
    entangling = [1, 1, 0, 1]
    
    bars = ax.bar(two_qubit_gates, entangling, color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Entangling')
    ax.set_title('Two-Qubit Gates')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, entangling):
        height = bar.get_height()
        label = 'Yes' if val == 1 else 'No'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[1, 0]
    
    platforms = ['Superconducting', 'Trapped Ion', 'Photonic', 'Neutral Atom']
    single_qubit = [0.9995, 0.9999, 0.999, 0.999]
    two_qubit = [0.99, 0.995, 0.95, 0.99]
    
    x = np.arange(len(platforms))
    width = 0.35
    
    ax.bar(x - width/2, single_qubit, width, label='Single-Qubit', color='lightblue', alpha=0.7)
    ax.bar(x + width/2, two_qubit, width, label='Two-Qubit', color='lightgreen', alpha=0.7)
    
    ax.set_ylabel('Fidelity')
    ax.set_title('Typical Gate Fidelities by Platform')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0.9, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    
    operations = ['Arbitrary\nSingle-Qubit', 'CNOT', 'Toffoli', 'n-Control\nToffoli']
    depths = [3, 1, 6, 2**(n-1) for n in [3]]  # Simplified
    depths = [3, 1, 6, 8]
    
    bars = ax.bar(operations, depths, color='lightyellow', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Decomposition Depth')
    ax.set_title('Gate Decomposition Complexity')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, depths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('quantum_gates_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: quantum_gates_summary.png")
    plt.show()


def plot_applications_timeline():
    """
    Visualize quantum computing applications and timeline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Quantum Computing: Applications and Timeline', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    domains = ['Cryptography', 'Chemistry', 'Optimization', 'ML', 'Finance', 'Materials']
    impact = [9, 8, 7, 6, 7, 8]
    maturity = [7, 5, 4, 3, 4, 5]
    
    x = np.arange(len(domains))
    width = 0.35
    
    ax.bar(x - width/2, impact, width, label='Potential Impact', color='lightblue', alpha=0.7)
    ax.bar(x + width/2, maturity, width, label='Current Maturity', color='lightcoral', alpha=0.7)
    
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Application Domains')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    ax.set_xlim(1980, 2040)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Quantum Computing Timeline')
    
    milestones = [
        (1985, 'Deutsch Algorithm', 2),
        (1994, 'Shor\'s Algorithm', 3),
        (1996, 'Grover\'s Algorithm', 4),
        (2001, 'First 7-qubit QC', 5),
        (2019, 'Quantum Supremacy', 6),
        (2024, '1000+ Qubits', 7),
        (2030, 'Error Correction', 8),
        (2035, 'Fault-Tolerant QC', 9)
    ]
    
    for year, event, y in milestones:
        color = 'blue' if year <= 2024 else 'gray'
        ax.plot([year, year], [0, y], color=color, linewidth=2)
        ax.plot(year, y, 'o', color=color, markersize=10)
        ax.text(year, y+0.3, event, ha='center', fontsize=8, rotation=45)
    
    ax.axvline(x=2024, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2024, 0.5, 'Now', ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax = axes[1, 0]
    
    years = [2000, 2005, 2010, 2015, 2020, 2024, 2030]
    qubits = [7, 12, 50, 100, 500, 1000, 10000]
    
    ax.semilogy(years, qubits, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=2024, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Qubits (log scale)')
    ax.set_title('Qubit Count Growth')
    ax.grid(True, alpha=0.3, which='both')
    
    ax.fill_between([2024, 2030], [1, 1], [100000, 100000], alpha=0.2, color='gray')
    ax.text(2027, 5000, 'Projected', ha='center', fontsize=10, style='italic')
    
    ax = axes[1, 1]
    
    algorithms = ['Shor\'s', 'Grover\'s', 'HHL', 'Simulation']
    speedups = ['Exponential', 'Quadratic', 'Exponential', 'Exponential']
    speedup_values = [4, 2, 4, 4]  # Log scale representation
    
    colors = ['red' if s == 'Exponential' else 'blue' for s in speedups]
    bars = ax.bar(algorithms, speedup_values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Speedup Type')
    ax.set_title('Quantum Algorithm Speedups')
    ax.set_yticks([2, 4])
    ax.set_yticklabels(['Quadratic', 'Exponential'])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                speedup, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('applications_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved: applications_timeline.png")
    plt.show()


def print_comprehensive_summary():
    """
    Print comprehensive summary of all concepts.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY: QUBITS LECTURE")
    print("="*80)
    
    print("\n" + "="*80)
    print("PART 1: MATHEMATICAL FOUNDATIONS")
    print("="*80)
    
    print("\n1. COMPLEX NUMBERS:")
    print("   • z = a + bi, where i² = -1")
    print("   • Polar form: z = r·e^(iθ)")
    print("   • Essential for quantum amplitudes and phases")
    
    print("\n2. VECTORS AND HILBERT SPACES:")
    print("   • Quantum states as vectors in complex Hilbert space")
    print("   • Inner product: ⟨ψ|φ⟩")
    print("   • Orthonormal basis: ⟨i|j⟩ = δᵢⱼ")
    print("   • Completeness: Σᵢ |i⟩⟨i| = I")
    
    print("\n" + "="*80)
    print("PART 2: QUANTUM MECHANICS BASICS")
    print("="*80)
    
    print("\n3. QUBIT STATE:")
    print("   • |ψ⟩ = α|0⟩ + β|1⟩")
    print("   • Normalization: |α|² + |β|² = 1")
    print("   • Superposition: Both states simultaneously")
    print("   • Measurement: Probabilistic collapse")
    
    print("\n4. BLOCH SPHERE:")
    print("   • Geometric representation: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩")
    print("   • Pure states on surface, mixed states inside")
    print("   • Gates as rotations")
    print("   • Measurement as projection")
    
    print("\n" + "="*80)
    print("PART 3: QUANTUM OPERATIONS")
    print("="*80)
    
    print("\n5. SINGLE-QUBIT GATES:")
    print("   • Pauli gates: X (bit flip), Y (combined), Z (phase flip)")
    print("   • Hadamard: H = (X + Z)/√2 (creates superposition)")
    print("   • Phase gates: S (π/2), T (π/4)")
    print("   • Rotations: Rx(θ), Ry(θ), Rz(θ)")
    
    print("\n6. TWO-QUBIT GATES:")
    print("   • CNOT: Controlled-NOT (creates entanglement)")
    print("   • CZ: Controlled-Z")
    print("   • SWAP: Exchange qubit states")
    print("   • Universal: {H, T, CNOT} or {Rx, Ry, Rz, CNOT}")
    
    print("\n" + "="*80)
    print("PART 4: MULTI-QUBIT SYSTEMS")
    print("="*80)
    
    print("\n7. TENSOR PRODUCTS:")
    print("   • n qubits → 2ⁿ dimensional space")
    print("   • |ψ⟩ ⊗ |φ⟩ = |ψφ⟩")
    print("   • Exponential scaling: Power and challenge")
    
    print("\n8. ENTANGLEMENT:")
    print("   • Cannot factor: |ψ⟩ ≠ |ψ₁⟩ ⊗ |ψ₂⟩")
    print("   • Bell states: Maximally entangled")
    print("   • |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print("   • Perfect correlations, no classical analog")
    
    print("\n" + "="*80)
    print("PART 5: PHYSICAL IMPLEMENTATIONS")
    print("="*80)
    
    print("\n9. QUBIT TECHNOLOGIES:")
    print("   • Superconducting: Fast, cryogenic (IBM, Google)")
    print("   • Trapped ions: High fidelity (IonQ, Honeywell)")
    print("   • Photonic: Room temperature (Xanadu)")
    print("   • Neutral atoms: Scalable (QuEra)")
    print("   • Topological: Error-resistant (Microsoft)")
    
    print("\n10. CHALLENGES:")
    print("   • Decoherence: ~100 μs (superconducting)")
    print("   • Gate errors: ~0.1-1%")
    print("   • Scalability: Connectivity, control")
    print("   • Error correction: Large overhead")
    
    print("\n" + "="*80)
    print("PART 6: QUANTUM ALGORITHMS")
    print("="*80)
    
    print("\n11. MAJOR ALGORITHMS:")
    print("   • Shor's: Factor N in O(log³N) time")
    print("   • Grover's: Search in O(√N) time")
    print("   • VQE: Variational quantum eigensolver")
    print("   • QAOA: Quantum optimization")
    print("   • Quantum simulation: Exponential speedup")
    
    print("\n12. APPLICATIONS:")
    print("   • Cryptography: Breaking RSA, QKD")
    print("   • Chemistry: Molecular simulation")
    print("   • Optimization: Logistics, finance")
    print("   • Machine learning: Quantum ML")
    print("   • Materials: Design new materials")
    
    print("\n" + "="*80)
    print("PART 7: QUANTUM INFORMATION")
    print("="*80)
    
    print("\n13. KEY CONCEPTS:")
    print("   • Quantum entropy: S(ρ) = -Tr(ρ log ρ)")
    print("   • No-cloning theorem")
    print("   • Quantum teleportation")
    print("   • Superdense coding")
    print("   • Error correction codes")
    
    print("\n" + "="*80)
    print("CURRENT STATE (2024)")
    print("="*80)
    
    print("\n• ~1000 qubit systems demonstrated")
    print("• Quantum advantage for specific problems")
    print("• NISQ era: Noisy Intermediate-Scale Quantum")
    print("• Active research in error correction")
    print("• Growing software ecosystem (Qiskit, Cirq, Q#)")
    
    print("\n" + "="*80)
    print("FUTURE OUTLOOK")
    print("="*80)
    
    print("\n• Near-term (1-5 years): Better NISQ algorithms")
    print("• Medium-term (5-10 years): Logical qubits")
    print("• Long-term (10+ years): Fault-tolerant QC")
    print("• Revolutionary impact on computation")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    
    print("\n1. Qubits use superposition and entanglement")
    print("2. Exponential state space enables quantum advantage")
    print("3. Quantum gates are unitary operations")
    print("4. Measurement collapses superposition")
    print("5. Multiple physical implementations exist")
    print("6. Powerful algorithms for specific problems")
    print("7. Challenges: decoherence, errors, scaling")
    print("8. Rapid progress toward practical quantum computing")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("SUMMARY - COMPREHENSIVE REVIEW OF QUBITS LECTURE")
    print("="*80)
    print("\nThis script provides a comprehensive summary of all 13 concepts")
    print("covered in the Qubits lecture series.\n")
    
    print_comprehensive_summary()
    
    print("\nGenerating summary visualizations...")
    print("-" * 80)
    
    print("\n1. Creating concept overview...")
    plot_concept_overview()
    
    print("\n2. Creating key principles visualization...")
    plot_key_principles()
    
    print("\n3. Creating quantum gates summary...")
    plot_quantum_gates_summary()
    
    print("\n4. Creating applications timeline...")
    plot_applications_timeline()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • concept_overview.png")
    print("  • key_principles.png")
    print("  • quantum_gates_summary.png")
    print("  • applications_timeline.png")
    print("\nThese visualizations provide a comprehensive summary of the")
    print("entire Qubits lecture series, covering all 13 concepts.")
    print("\n" + "="*80)
    print("THANK YOU FOR COMPLETING THE QUBITS LECTURE!")
    print("="*80)
    print("\nYou now have a solid foundation in:")
    print("  • Quantum mechanics basics")
    print("  • Qubit mathematics and operations")
    print("  • Multi-qubit systems and entanglement")
    print("  • Physical implementations")
    print("  • Quantum algorithms and applications")
    print("\nContinue learning and exploring quantum computing!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
