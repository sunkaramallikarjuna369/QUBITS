"""
Bit vs Qubit - Python Demonstrations
====================================

This script provides comprehensive demonstrations comparing classical bits and quantum qubits:
- Classical bit states and operations
- Quantum qubit superposition
- Measurement statistics
- State space comparison
- Quantum advantage visualization

Author: QpiAI Quantum Engineer Course
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def plot_bit_vs_qubit_states():
    """
    Compare classical bit and quantum qubit state spaces
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Classical Bit vs Quantum Qubit State Spaces', fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Classical Bit', fontsize=14, fontweight='bold')
    ax.set_xlabel('State', fontsize=12)
    ax.axis('off')
    
    bit0 = Circle((0, 0.5), 0.3, color='#ff6b6b', ec='black', linewidth=3)
    bit1 = Circle((1, 0.5), 0.3, color='#4ecdc4', ec='black', linewidth=3)
    ax.add_patch(bit0)
    ax.add_patch(bit1)
    
    ax.text(0, 0.5, '0', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax.text(1, 0.5, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    
    ax.text(0, -0.1, 'State 0', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, -0.1, 'State 1', ha='center', fontsize=12, fontweight='bold')
    
    ax.text(0.5, 1.2, 'Only 2 possible states\n(discrete)', 
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Quantum Qubit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    
    circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ax.plot(0, 1, 'o', color='#ff6b6b', markersize=15, markeredgecolor='black', markeredgewidth=2, label='|0⟩')
    ax.plot(0, -1, 'o', color='#4ecdc4', markersize=15, markeredgecolor='black', markeredgewidth=2, label='|1⟩')
    
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot(x, y, 'o', color='#64ffda', markersize=10, alpha=0.7)
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#64ffda', alpha=0.5))
    
    ax.axhline(y=0, color='k', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=1, alpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=11, loc='upper right')
    
    ax.text(0, 1.35, 'Infinite possible states\n(continuous)', 
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('bit_vs_qubit_states.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: bit_vs_qubit_states.png")
    plt.close()

def plot_superposition():
    """
    Visualize quantum superposition
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Quantum Superposition', fontsize=18, fontweight='bold')
    
    states = [
        ([1, 0], '|0⟩', 'Pure |0⟩ state'),
        ([0, 1], '|1⟩', 'Pure |1⟩ state'),
        ([1/np.sqrt(2), 1/np.sqrt(2)], '|+⟩', 'Equal superposition'),
        ([1/np.sqrt(2), -1/np.sqrt(2)], '|−⟩', 'Equal superposition (opposite phase)')
    ]
    
    for idx, (amplitudes, label, description) in enumerate(states):
        ax = axes[idx // 2, idx % 2]
        
        probs = [abs(amp)**2 for amp in amplitudes]
        
        basis_labels = ['|0⟩', '|1⟩']
        x = np.arange(len(basis_labels))
        
        bars = ax.bar(x, probs, color=['#ff6b6b', '#4ecdc4'], 
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'{label} - {description}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(i, prob + 0.05, f'{prob:.3f}', 
                   ha='center', fontsize=11, fontweight='bold')
        
        state_str = f'[{amplitudes[0]:.3f}, {amplitudes[1]:.3f}]ᵀ'
        ax.text(0.95, 0.95, f'State:\n{state_str}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('superposition.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: superposition.png")
    plt.close()

def plot_measurement_collapse():
    """
    Visualize measurement collapse
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Quantum Measurement Collapse', fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_title('Before Measurement\n(Superposition)', fontsize=13, fontweight='bold')
    
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    probs = [abs(amp)**2 for amp in state]
    
    bars = ax.bar(['|0⟩', '|1⟩'], probs, color=['#ff6b6b', '#4ecdc4'], 
                 alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(i, prob + 0.05, f'{prob:.3f}', 
               ha='center', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.85, 'Qubit in superposition\nBoth states possible',
           transform=ax.transAxes, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=10, fontweight='bold')
    
    ax = axes[1]
    ax.set_title('Measurement\n(Collapse)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    ax.text(0.5, 0.6, '⚡ MEASUREMENT ⚡', 
           ha='center', va='center', fontsize=20, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    
    ax.text(0.5, 0.4, 'Superposition collapses\nto definite state',
           ha='center', va='center', fontsize=12, style='italic')
    
    ax = axes[2]
    ax.set_title('After Measurement\n(Collapsed to |0⟩)', fontsize=13, fontweight='bold')
    
    collapsed_probs = [1.0, 0.0]  # Collapsed to |0⟩
    
    bars = ax.bar(['|0⟩', '|1⟩'], collapsed_probs, color=['#ff6b6b', '#4ecdc4'], 
                 alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, prob) in enumerate(zip(bars, collapsed_probs)):
        if prob > 0:
            ax.text(i, prob + 0.05, f'{prob:.3f}', 
                   ha='center', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.85, 'Qubit now in |0⟩\nSuperposition destroyed',
           transform=ax.transAxes, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('measurement_collapse.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: measurement_collapse.png")
    plt.close()

def plot_quantum_parallelism():
    """
    Visualize quantum parallelism advantage
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Quantum Parallelism: Exponential State Space', 
                fontsize=16, fontweight='bold', pad=20)
    
    n_qubits = np.arange(1, 11)
    classical_states = n_qubits  # Classical: n bits store n values
    quantum_states = 2**n_qubits  # Quantum: n qubits represent 2^n states
    
    ax.semilogy(n_qubits, classical_states, 'o-', color='#ff6b6b', 
               linewidth=3, markersize=10, label='Classical (n bits)')
    ax.semilogy(n_qubits, quantum_states, 's-', color='#64ffda', 
               linewidth=3, markersize=10, label='Quantum (n qubits)')
    
    ax.set_xlabel('Number of Bits/Qubits', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of States (log scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=13, loc='upper left')
    
    ax.annotate('Exponential growth!', 
               xy=(7, 2**7), xytext=(5, 2**9),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    text_str = 'Examples:\n'
    text_str += '• 10 qubits: 1,024 states\n'
    text_str += '• 20 qubits: 1,048,576 states\n'
    text_str += '• 50 qubits: 1.1 × 10¹⁵ states\n'
    text_str += '• 300 qubits: More states than\n  atoms in universe!'
    
    ax.text(0.98, 0.35, text_str,
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           fontsize=11, fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig('quantum_parallelism.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: quantum_parallelism.png")
    plt.close()

def plot_no_cloning():
    """
    Visualize no-cloning theorem
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cloning: Classical vs Quantum', fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_title('Classical Bit Cloning\n✓ POSSIBLE', fontsize=14, fontweight='bold', color='green')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    circle1 = Circle((2, 5), 0.8, color='#ff6b6b', ec='black', linewidth=3)
    ax.add_patch(circle1)
    ax.text(2, 5, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax.text(2, 3.5, 'Original', ha='center', fontsize=11, fontweight='bold')
    
    ax.annotate('', xy=(5.5, 5), xytext=(3.5, 5),
               arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax.text(4.5, 5.5, 'COPY', ha='center', fontsize=12, fontweight='bold', color='green')
    
    circle2 = Circle((7, 6.5), 0.8, color='#ff6b6b', ec='black', linewidth=3)
    circle3 = Circle((7, 5), 0.8, color='#ff6b6b', ec='black', linewidth=3)
    circle4 = Circle((7, 3.5), 0.8, color='#ff6b6b', ec='black', linewidth=3)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.text(7, 6.5, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax.text(7, 5, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax.text(7, 3.5, '1', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    ax.text(7, 2, 'Perfect Copies', ha='center', fontsize=11, fontweight='bold')
    
    ax = axes[1]
    ax.set_title('Quantum Qubit Cloning\n✗ IMPOSSIBLE', fontsize=14, fontweight='bold', color='red')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    circle1 = Circle((2, 5), 0.8, color='#64ffda', ec='black', linewidth=3)
    ax.add_patch(circle1)
    ax.text(2, 5, '|ψ⟩', ha='center', va='center', fontsize=18, fontweight='bold', color='black')
    ax.text(2, 3.5, 'Original', ha='center', fontsize=11, fontweight='bold')
    
    ax.annotate('', xy=(5.5, 5), xytext=(3.5, 5),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(4.5, 5.5, 'COPY?', ha='center', fontsize=12, fontweight='bold', color='red')
    ax.text(4.5, 4.5, '✗', ha='center', fontsize=30, fontweight='bold', color='red')
    
    ax.text(7, 5, '❌', ha='center', va='center', fontsize=60, color='red', alpha=0.7)
    ax.text(7, 2, 'No-Cloning\nTheorem', ha='center', fontsize=11, fontweight='bold')
    
    ax.text(5, 1, 'Cannot create identical copy of unknown quantum state',
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('no_cloning.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: no_cloning.png")
    plt.close()

def plot_comparison_table():
    """
    Create visual comparison table
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.set_title('Classical Bit vs Quantum Qubit: Comprehensive Comparison', 
                fontsize=18, fontweight='bold', pad=20)
    
    properties = [
        'States',
        'Superposition',
        'Measurement',
        'Cloning',
        'Interference',
        'Entanglement',
        'Parallelism',
        'Gates'
    ]
    
    classical = [
        '0 or 1\n(discrete)',
        'No\n(always definite)',
        'Non-destructive\n(doesn\'t change state)',
        'Possible\n(perfect copies)',
        'No\n(no wave behavior)',
        'No\n(independent)',
        'Sequential\n(one at a time)',
        'Boolean\n(AND, OR, NOT)'
    ]
    
    quantum = [
        'α|0⟩ + β|1⟩\n(continuous)',
        'Yes\n(both states)',
        'Destructive\n(collapses state)',
        'Impossible\n(no-cloning)',
        'Yes\n(quantum interference)',
        'Yes\n(correlated)',
        'Exponential\n(2ⁿ states)',
        'Unitary\n(H, CNOT, Phase)'
    ]
    
    n_rows = len(properties)
    row_height = 0.8
    col_widths = [0.2, 0.35, 0.35]
    
    y_pos = 0.95
    ax.text(0.1, y_pos, 'Property', ha='center', va='center', 
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9, edgecolor='white', linewidth=2))
    ax.text(0.4, y_pos, 'Classical Bit', ha='center', va='center', 
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ff6b6b', alpha=0.7, edgecolor='white', linewidth=2))
    ax.text(0.75, y_pos, 'Quantum Qubit', ha='center', va='center', 
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#64ffda', alpha=0.7, edgecolor='white', linewidth=2))
    
    y_pos = 0.85
    for i, (prop, clas, quan) in enumerate(zip(properties, classical, quantum)):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y_pos - 0.05), 1, 0.1, 
                                      facecolor='#0f0f1e', alpha=0.3, zorder=0))
        
        ax.text(0.1, y_pos, prop, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.4, y_pos, clas, ha='center', va='center', 
               fontsize=10, multialignment='center')
        ax.text(0.75, y_pos, quan, ha='center', va='center', 
               fontsize=10, multialignment='center')
        
        y_pos -= 0.1
    
    plt.tight_layout()
    plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_table.png")
    plt.close()

def print_key_concepts():
    """
    Print key concepts about bits vs qubits
    """
    print("\n" + "="*70)
    print("KEY CONCEPTS: CLASSICAL BIT VS QUANTUM QUBIT")
    print("="*70)
    
    print("\n1. CLASSICAL BIT:")
    print("   • Two states: 0 or 1")
    print("   • Always in definite state")
    print("   • Measurement doesn't change state")
    print("   • Can be copied perfectly")
    print("   • No interference or entanglement")
    
    print("\n2. QUANTUM QUBIT:")
    print("   • Superposition: α|0⟩ + β|1⟩")
    print("   • Complex probability amplitudes")
    print("   • Measurement collapses state")
    print("   • Cannot be cloned (no-cloning theorem)")
    print("   • Exhibits interference and entanglement")
    
    print("\n3. KEY DIFFERENCES:")
    print("   • State space: Discrete vs Continuous")
    print("   • Superposition: No vs Yes")
    print("   • Measurement: Non-destructive vs Destructive")
    print("   • Cloning: Possible vs Impossible")
    print("   • Parallelism: Sequential vs Exponential")
    
    print("\n4. QUANTUM ADVANTAGE:")
    print("   • n qubits represent 2ⁿ states simultaneously")
    print("   • Quantum parallelism enables exponential speedup")
    print("   • Interference amplifies correct answers")
    print("   • Entanglement creates non-classical correlations")
    
    print("\n5. APPLICATIONS:")
    print("   • Shor's algorithm: Factor large numbers")
    print("   • Grover's algorithm: Search databases")
    print("   • Quantum simulation: Simulate quantum systems")
    print("   • Quantum cryptography: Secure communication")
    
    print("\n" + "="*70)

def main():
    """
    Main function to run all demonstrations
    """
    print("\n" + "="*70)
    print("BIT VS QUBIT - PYTHON DEMONSTRATIONS")
    print("="*70)
    
    print("\nGenerating visualizations...")
    print("-" * 70)
    
    plot_bit_vs_qubit_states()
    plot_superposition()
    plot_measurement_collapse()
    plot_quantum_parallelism()
    plot_no_cloning()
    plot_comparison_table()
    
    print_key_concepts()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. bit_vs_qubit_states.png - State space comparison")
    print("  2. superposition.png - Quantum superposition states")
    print("  3. measurement_collapse.png - Measurement collapse process")
    print("  4. quantum_parallelism.png - Exponential state space growth")
    print("  5. no_cloning.png - No-cloning theorem")
    print("  6. comparison_table.png - Comprehensive comparison table")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
