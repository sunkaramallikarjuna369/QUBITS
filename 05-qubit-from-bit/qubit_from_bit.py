#!/usr/bin/env python3
"""
Qubit from Bit - Comprehensive Python Demonstrations

This script demonstrates how to construct a qubit from classical bits,
showing the transition from discrete classical states to continuous
quantum superposition states.

Author: Devin (AI Software Engineer)
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib plots"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def create_qubit_state(alpha, beta):
    """
    Create a normalized qubit state |ψ⟩ = α|0⟩ + β|1⟩
    
    Args:
        alpha: Complex amplitude for |0⟩
        beta: Complex amplitude for |1⟩
    
    Returns:
        Normalized qubit state as numpy array
    """
    state = np.array([alpha, beta], dtype=complex)
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    if norm > 0:
        state = state / norm
    return state


def measure_qubit(state, num_measurements=1000):
    """
    Simulate measurement of a qubit state
    
    Args:
        state: Qubit state [α, β]
        num_measurements: Number of measurements to simulate
    
    Returns:
        Array of measurement outcomes (0s and 1s)
    """
    prob_0 = np.abs(state[0])**2
    outcomes = np.random.choice([0, 1], size=num_measurements, p=[prob_0, 1-prob_0])
    return outcomes


def plot_classical_to_quantum_transition():
    """Visualize the transition from classical bits to quantum qubits"""
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter([0, 1], [0, 0], s=500, c=['#ff0000', '#00ffff'], 
                edgecolors='white', linewidths=2, zorder=3)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Bit Value', fontsize=12, color='#64ffda')
    ax1.set_title('Classical Bits (Discrete States)', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax1.text(0, -0.3, '|0⟩', ha='center', fontsize=14, color='#ff0000')
    ax1.text(1, -0.3, '|1⟩', ha='center', fontsize=14, color='#00ffff')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(2, 3, 2)
    alphas = np.linspace(0, 1, 100)
    betas = np.sqrt(1 - alphas**2)
    ax2.plot(alphas, betas, 'c-', linewidth=3, label='Allowed states')
    ax2.scatter([1, 0], [0, 1], s=500, c=['#ff0000', '#00ffff'], 
                edgecolors='white', linewidths=2, zorder=3)
    ax2.scatter([1/np.sqrt(2)], [1/np.sqrt(2)], s=500, c='#64ffda', 
                edgecolors='white', linewidths=2, zorder=3, marker='*')
    ax2.set_xlabel('|α| (Amplitude for |0⟩)', fontsize=12, color='#64ffda')
    ax2.set_ylabel('|β| (Amplitude for |1⟩)', fontsize=12, color='#64ffda')
    ax2.set_title('Quantum Qubits (Continuous States)', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax2.text(1, -0.1, '|0⟩', ha='center', fontsize=12, color='#ff0000')
    ax2.text(-0.1, 1, '|1⟩', ha='center', fontsize=12, color='#00ffff')
    ax2.text(1/np.sqrt(2), 1/np.sqrt(2)+0.1, '|+⟩', ha='center', 
             fontsize=12, color='#64ffda')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    
    ax3 = fig.add_subplot(2, 3, 3)
    states = {
        '|0⟩': (1, 0),
        '|1⟩': (0, 1),
        '|+⟩': (1/np.sqrt(2), 1/np.sqrt(2)),
        '|−⟩': (1/np.sqrt(2), -1/np.sqrt(2)),
        '|+i⟩': (1/np.sqrt(2), 1j/np.sqrt(2)),
        '|−i⟩': (1/np.sqrt(2), -1j/np.sqrt(2))
    }
    
    x_pos = np.arange(len(states))
    probs_0 = [np.abs(alpha)**2 for alpha, beta in states.values()]
    probs_1 = [np.abs(beta)**2 for alpha, beta in states.values()]
    
    width = 0.35
    ax3.bar(x_pos - width/2, probs_0, width, label='P(|0⟩)', color='#ff0000', alpha=0.7)
    ax3.bar(x_pos + width/2, probs_1, width, label='P(|1⟩)', color='#00ffff', alpha=0.7)
    ax3.set_xlabel('Quantum State', fontsize=12, color='#64ffda')
    ax3.set_ylabel('Measurement Probability', fontsize=12, color='#64ffda')
    ax3.set_title('Measurement Probabilities', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(states.keys())
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.1)
    
    ax4 = fig.add_subplot(2, 3, 4)
    categories = ['Classical\nBit', 'Quantum\nQubit']
    state_counts = [2, np.inf]
    colors = ['#ff0000', '#64ffda']
    
    bars = ax4.bar(categories, [2, 100], color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    ax4.set_ylabel('State Space Size', fontsize=12, color='#64ffda')
    ax4.set_title('State Space Comparison', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax4.text(0, 2.5, '2 states\n(discrete)', ha='center', fontsize=10)
    ax4.text(1, 105, '∞ states\n(continuous)', ha='center', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    x_before = [0, 1]
    y_before = [0.5, 0.5]
    ax5.bar(x_before, y_before, width=0.3, color='#64ffda', alpha=0.7, 
            label='Before (Superposition)')
    
    y_after_0 = [1, 0]
    y_after_1 = [0, 1]
    ax5.bar([0.35], [1], width=0.3, color='#ff0000', alpha=0.5, 
            label='After (Collapsed to |0⟩)')
    ax5.bar([1.35], [1], width=0.3, color='#00ffff', alpha=0.5, 
            label='After (Collapsed to |1⟩)')
    
    ax5.set_xlabel('Basis State', fontsize=12, color='#64ffda')
    ax5.set_ylabel('Probability', fontsize=12, color='#64ffda')
    ax5.set_title('Measurement Collapse', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['|0⟩', '|1⟩'])
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1.2)
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    info_data = {
        'Classical Bit': 1,
        'Qubit (extractable)': 1,
        'Qubit (total)': 100
    }
    
    colors_info = ['#ff0000', '#ffaa00', '#64ffda']
    bars = ax6.bar(range(len(info_data)), list(info_data.values()), 
                   color=colors_info, alpha=0.7, edgecolor='white', linewidth=2)
    ax6.set_ylabel('Information (bits)', fontsize=12, color='#64ffda')
    ax6.set_title('Information Content', fontsize=14, 
                  color='#64ffda', fontweight='bold')
    ax6.set_xticks(range(len(info_data)))
    ax6.set_xticklabels(list(info_data.keys()), rotation=15, ha='right')
    ax6.text(0, 2, '1 bit', ha='center', fontsize=10)
    ax6.text(1, 2, '1 bit\n(via measurement)', ha='center', fontsize=9)
    ax6.text(2, 105, '∞ bits\n(continuous α, β)', ha='center', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('classical_to_quantum_transition.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: classical_to_quantum_transition.png")
    plt.close()


def plot_superposition_construction():
    """Visualize the construction of superposition states"""
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.quiver(0, 0, 0, 0, 0, 1, color='#ff0000', arrow_length_ratio=0.2, linewidth=3)
    ax1.quiver(0, 0, 0, 0, 0, -1, color='#00ffff', arrow_length_ratio=0.2, linewidth=3)
    ax1.text(0, 0, 1.3, '|0⟩', fontsize=14, color='#ff0000')
    ax1.text(0, 0, -1.3, '|1⟩', fontsize=14, color='#00ffff')
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X', color='#64ffda')
    ax1.set_ylabel('Y', color='#64ffda')
    ax1.set_zlabel('Z', color='#64ffda')
    ax1.set_title('Step 1: Embed Classical States', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1, linewidth=0.5)
    
    ax2.quiver(0, 0, 0, 0, 0, 1, color='#ff0000', arrow_length_ratio=0.2, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0, -1, color='#00ffff', arrow_length_ratio=0.2, linewidth=2)
    
    ax2.quiver(0, 0, 0, 1, 0, 0, color='#64ffda', arrow_length_ratio=0.2, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 1, 0, color='#ffaa00', arrow_length_ratio=0.2, linewidth=2)
    
    ax2.text(0, 0, 1.3, '|0⟩', fontsize=12, color='#ff0000')
    ax2.text(0, 0, -1.3, '|1⟩', fontsize=12, color='#00ffff')
    ax2.text(1.3, 0, 0, '|+⟩', fontsize=12, color='#64ffda')
    ax2.text(0, 1.3, 0, '|+i⟩', fontsize=12, color='#ffaa00')
    
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X', color='#64ffda')
    ax2.set_ylabel('Y', color='#64ffda')
    ax2.set_zlabel('Z', color='#64ffda')
    ax2.set_title('Step 2: Add Superposition', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    
    ax3 = fig.add_subplot(2, 3, 3)
    
    phases = np.linspace(0, 2*np.pi, 100)
    alpha = 1/np.sqrt(2)
    
    real_parts = [alpha * np.cos(phi) for phi in phases]
    imag_parts = [alpha * np.sin(phi) for phi in phases]
    
    ax3.plot(real_parts, imag_parts, 'c-', linewidth=3, label='Phase circle')
    ax3.scatter([alpha], [0], s=200, c='#64ffda', edgecolors='white', 
                linewidths=2, zorder=3, label='|+⟩')
    ax3.scatter([0], [alpha], s=200, c='#ffaa00', edgecolors='white', 
                linewidths=2, zorder=3, label='|+i⟩')
    ax3.scatter([0], [-alpha], s=200, c='#ff00ff', edgecolors='white', 
                linewidths=2, zorder=3, label='|−i⟩')
    
    ax3.set_xlabel('Real(β)', fontsize=12, color='#64ffda')
    ax3.set_ylabel('Imag(β)', fontsize=12, color='#64ffda')
    ax3.set_title('Step 3: Add Complex Phases', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
    ax3.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    alphas_unnorm = np.linspace(0, 2, 50)
    betas_unnorm = np.linspace(0, 2, 50)
    
    alphas_norm = np.linspace(0, 1, 100)
    betas_norm = np.sqrt(1 - alphas_norm**2)
    
    ax4.plot(alphas_norm, betas_norm, 'c-', linewidth=3, label='|α|² + |β|² = 1')
    ax4.fill_between(alphas_norm, 0, betas_norm, alpha=0.2, color='cyan')
    
    valid_states = [
        (1, 0, '|0⟩', '#ff0000'),
        (0, 1, '|1⟩', '#00ffff'),
        (1/np.sqrt(2), 1/np.sqrt(2), '|+⟩', '#64ffda'),
        (np.sqrt(0.7), np.sqrt(0.3), 'custom', '#ffaa00')
    ]
    
    for alpha, beta, label, color in valid_states:
        ax4.scatter([alpha], [beta], s=200, c=color, edgecolors='white', 
                   linewidths=2, zorder=3)
        ax4.text(alpha+0.05, beta+0.05, label, fontsize=10, color=color)
    
    ax4.set_xlabel('|α|', fontsize=12, color='#64ffda')
    ax4.set_ylabel('|β|', fontsize=12, color='#64ffda')
    ax4.set_title('Step 4: Enforce Normalization', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(-0.1, 1.5)
    ax4.set_ylim(-0.1, 1.5)
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    plus_state = create_qubit_state(1/np.sqrt(2), 1/np.sqrt(2))
    measurements = measure_qubit(plus_state, 1000)
    
    counts = [np.sum(measurements == 0), np.sum(measurements == 1)]
    ax5.bar([0, 1], counts, color=['#ff0000', '#00ffff'], alpha=0.7, 
            edgecolor='white', linewidth=2)
    ax5.axhline(y=500, color='#64ffda', linestyle='--', linewidth=2, 
                label='Expected (50%)')
    ax5.set_xlabel('Measurement Outcome', fontsize=12, color='#64ffda')
    ax5.set_ylabel('Count (out of 1000)', fontsize=12, color='#64ffda')
    ax5.set_title('Measurement Statistics for |+⟩', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['|0⟩', '|1⟩'])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax6.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1)
    
    states_3d = [
        (0, 0, 1, '#ff0000', '|0⟩'),
        (0, 0, -1, '#00ffff', '|1⟩'),
        (1, 0, 0, '#64ffda', '|+⟩'),
        (-1, 0, 0, '#ff00ff', '|−⟩'),
        (0, 1, 0, '#ffaa00', '|+i⟩'),
        (0, -1, 0, '#00ff00', '|−i⟩')
    ]
    
    for x, y, z, color, label in states_3d:
        ax6.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.15, linewidth=2)
        ax6.text(x*1.3, y*1.3, z*1.3, label, fontsize=10, color=color)
    
    ax6.set_xlim([-1.5, 1.5])
    ax6.set_ylim([-1.5, 1.5])
    ax6.set_zlim([-1.5, 1.5])
    ax6.set_xlabel('X', color='#64ffda')
    ax6.set_ylabel('Y', color='#64ffda')
    ax6.set_zlabel('Z', color='#64ffda')
    ax6.set_title('Complete Qubit State Space', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('superposition_construction.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: superposition_construction.png")
    plt.close()


def plot_measurement_process():
    """Visualize the measurement and collapse process"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = axes[0, 0]
    x = [0, 1]
    y_super = [0.5, 0.5]
    bars1 = ax1.bar(x, y_super, color='#64ffda', alpha=0.7, edgecolor='white', linewidth=2)
    ax1.set_ylim(0, 1.2)
    ax1.set_xlabel('Basis State', fontsize=12, color='#64ffda')
    ax1.set_ylabel('Probability Amplitude²', fontsize=12, color='#64ffda')
    ax1.set_title('Before Measurement\n|ψ⟩ = (|0⟩ + |1⟩)/√2', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.5, 0.6, 'Superposition', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='#64ffda', alpha=0.3))
    
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.7, '⚡ MEASUREMENT ⚡', ha='center', va='center', 
             fontsize=16, color='#64ffda', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=3))
    ax2.text(0.5, 0.5, 'Quantum → Classical', ha='center', va='center', 
             fontsize=12, color='white')
    ax2.text(0.5, 0.3, 'Interaction', ha='center', va='center', 
             fontsize=12, color='white')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    ax3 = axes[0, 2]
    y_collapsed_0 = [1, 0]
    bars3 = ax3.bar(x, y_collapsed_0, color=['#ff0000', '#00ffff'], alpha=0.7, 
                    edgecolor='white', linewidth=2)
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel('Basis State', fontsize=12, color='#64ffda')
    ax3.set_ylabel('Probability', fontsize=12, color='#64ffda')
    ax3.set_title('After Measurement (50% chance)\n|ψ⟩ = |0⟩', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['|0⟩', '|1⟩'])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0, 1.1, 'Collapsed!', ha='center', fontsize=11, color='#ff0000',
             bbox=dict(boxstyle='round', facecolor='#ff0000', alpha=0.3))
    
    ax4 = axes[1, 0]
    y_collapsed_1 = [0, 1]
    bars4 = ax4.bar(x, y_collapsed_1, color=['#ff0000', '#00ffff'], alpha=0.7, 
                    edgecolor='white', linewidth=2)
    ax4.set_ylim(0, 1.2)
    ax4.set_xlabel('Basis State', fontsize=12, color='#64ffda')
    ax4.set_ylabel('Probability', fontsize=12, color='#64ffda')
    ax4.set_title('After Measurement (50% chance)\n|ψ⟩ = |1⟩', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['|0⟩', '|1⟩'])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(1, 1.1, 'Collapsed!', ha='center', fontsize=11, color='#00ffff',
             bbox=dict(boxstyle='round', facecolor='#00ffff', alpha=0.3))
    
    ax5 = axes[1, 1]
    
    num_trials = [10, 100, 1000, 10000]
    results_0 = []
    results_1 = []
    
    plus_state = create_qubit_state(1/np.sqrt(2), 1/np.sqrt(2))
    
    for n in num_trials:
        measurements = measure_qubit(plus_state, n)
        results_0.append(np.sum(measurements == 0) / n)
        results_1.append(np.sum(measurements == 1) / n)
    
    x_pos = np.arange(len(num_trials))
    width = 0.35
    
    bars_0 = ax5.bar(x_pos - width/2, results_0, width, label='P(|0⟩)', 
                     color='#ff0000', alpha=0.7)
    bars_1 = ax5.bar(x_pos + width/2, results_1, width, label='P(|1⟩)', 
                     color='#00ffff', alpha=0.7)
    ax5.axhline(y=0.5, color='#64ffda', linestyle='--', linewidth=2, 
                label='Expected (50%)')
    
    ax5.set_xlabel('Number of Measurements', fontsize=12, color='#64ffda')
    ax5.set_ylabel('Measured Probability', fontsize=12, color='#64ffda')
    ax5.set_title('Convergence to Expected Probabilities', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(num_trials)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 0.7)
    
    ax6 = axes[1, 2]
    
    stages = ['Superposition\nBefore', 'Measurement', 'Classical\nAfter', 'Cannot\nReverse']
    y_pos = [3, 2, 1, 0]
    colors_stages = ['#64ffda', '#ffaa00', '#ff0000', '#ff0000']
    
    for i, (stage, y, color) in enumerate(zip(stages, y_pos, colors_stages)):
        if i < 3:
            ax6.add_patch(FancyBboxPatch((0.1, y-0.3), 0.8, 0.6, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=color, edgecolor='white', 
                                         alpha=0.3, linewidth=2))
            ax6.text(0.5, y, stage, ha='center', va='center', fontsize=11, 
                    color='white', fontweight='bold')
            if i < 2:
                ax6.annotate('', xy=(0.5, y-0.4), xytext=(0.5, y-0.9),
                           arrowprops=dict(arrowstyle='->', color='white', lw=2))
        else:
            ax6.text(0.5, y, stage, ha='center', va='center', fontsize=11, 
                    color='#ff0000', fontweight='bold')
            ax6.annotate('', xy=(0.5, y+0.5), xytext=(0.5, y+0.9),
                       arrowprops=dict(arrowstyle='->', color='#ff0000', lw=2))
            ax6.text(0.5, y+0.7, '✗', ha='center', va='center', fontsize=20, 
                    color='#ff0000', fontweight='bold')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(-0.5, 3.5)
    ax6.set_title('Irreversibility of Measurement', fontsize=12, 
                  color='#64ffda', fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('measurement_process.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: measurement_process.png")
    plt.close()


def plot_quantum_advantage():
    """Visualize the quantum advantage from qubits"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = axes[0, 0]
    n_qubits = np.arange(1, 11)
    classical_states = 2 * n_qubits  # Can only be in one state at a time
    quantum_states = 2**n_qubits  # Can be in superposition of all states
    
    ax1.semilogy(n_qubits, classical_states, 'o-', color='#ff0000', linewidth=3, 
                 markersize=8, label='Classical (sequential)')
    ax1.semilogy(n_qubits, quantum_states, 's-', color='#64ffda', linewidth=3, 
                 markersize=8, label='Quantum (parallel)')
    ax1.set_xlabel('Number of Qubits/Bits', fontsize=12, color='#64ffda')
    ax1.set_ylabel('Accessible States', fontsize=12, color='#64ffda')
    ax1.set_title('Exponential State Space Growth', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2 = axes[0, 1]
    
    n_bits = [1, 2, 3, 4, 5]
    classical_ops = [2, 4, 8, 16, 32]  # Need to evaluate each state separately
    quantum_ops = [1, 1, 1, 1, 1]  # Can evaluate all states in parallel
    
    x_pos = np.arange(len(n_bits))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, classical_ops, width, label='Classical', 
                    color='#ff0000', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, quantum_ops, width, label='Quantum', 
                    color='#64ffda', alpha=0.7)
    
    ax2.set_xlabel('Number of Qubits/Bits', fontsize=12, color='#64ffda')
    ax2.set_ylabel('Operations Needed', fontsize=12, color='#64ffda')
    ax2.set_title('Quantum Parallelism Advantage', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(n_bits)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    ax3 = axes[1, 0]
    
    categories = ['1 Classical\nBit', '1 Qubit\n(before measure)', '1 Qubit\n(after measure)']
    info_values = [1, 100, 1]  # Representing continuous vs discrete
    colors_info = ['#ff0000', '#64ffda', '#ffaa00']
    
    bars = ax3.bar(range(len(categories)), info_values, color=colors_info, 
                   alpha=0.7, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Information Content (arbitrary units)', fontsize=12, color='#64ffda')
    ax3.set_title('Information Density Comparison', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax3.text(0, info_values[0]+5, '1 bit\n(discrete)', ha='center', fontsize=10)
    ax3.text(1, info_values[1]+5, '∞ bits\n(continuous α, β)', ha='center', fontsize=10)
    ax3.text(2, info_values[2]+5, '1 bit\n(collapsed)', ha='center', fontsize=10)
    
    ax4 = axes[1, 1]
    
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Property', 'Classical Bit', 'Quantum Qubit'],
        ['State Space', 'Discrete {0,1}', 'Continuous'],
        ['Superposition', 'No', 'Yes'],
        ['Parallelism', 'Sequential', 'Intrinsic'],
        ['Interference', 'No', 'Yes'],
        ['Entanglement', 'No', 'Yes'],
        ['Measurement', 'Deterministic', 'Probabilistic'],
        ['Cloning', 'Allowed', 'Forbidden'],
        ['Information', '1 bit', '∞ (continuous)']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#64ffda')
                cell.set_text_props(weight='bold', color='#0a0a0a')
            else:
                if j == 0:  # Property column
                    cell.set_facecolor('#1a1a2e')
                    cell.set_text_props(color='#64ffda', weight='bold')
                elif j == 1:  # Classical column
                    cell.set_facecolor('#2a0a0a')
                    cell.set_text_props(color='#ff6666')
                else:  # Quantum column
                    cell.set_facecolor('#0a2a2a')
                    cell.set_text_props(color='#66ffdd')
            cell.set_edgecolor('white')
            cell.set_linewidth(1)
    
    ax4.set_title('Quantum Advantage Summary', fontsize=13, 
                  color='#64ffda', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('quantum_advantage.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: quantum_advantage.png")
    plt.close()


def print_key_concepts():
    """Print key concepts about constructing qubits from bits"""
    print("\n" + "="*70)
    print("KEY CONCEPTS: QUBIT FROM BIT")
    print("="*70)
    
    print("\n1. CLASSICAL BIT PROPERTIES:")
    print("   • Discrete states: {0, 1}")
    print("   • Deterministic measurement")
    print("   • Can be copied freely")
    print("   • Sequential processing")
    print("   • Boolean logic operations")
    
    print("\n2. QUANTUM QUBIT CONSTRUCTION:")
    print("   • Step 1: Embed classical states as |0⟩ and |1⟩")
    print("   • Step 2: Allow superposition: α|0⟩ + β|1⟩")
    print("   • Step 3: Add complex phases: α|0⟩ + e^(iφ)β|1⟩")
    print("   • Step 4: Enforce normalization: |α|² + |β|² = 1")
    
    print("\n3. KEY DIFFERENCES:")
    print("   • State space: Discrete → Continuous")
    print("   • Information: 1 bit → ∞ (continuous parameters)")
    print("   • Processing: Sequential → Parallel (superposition)")
    print("   • Measurement: Deterministic → Probabilistic")
    print("   • Copying: Allowed → Forbidden (no-cloning theorem)")
    
    print("\n4. SUPERPOSITION:")
    print("   • Qubit exists in multiple states simultaneously")
    print("   • Not the same as unknown state")
    print("   • Enables quantum parallelism")
    print("   • Example: |+⟩ = (|0⟩ + |1⟩)/√2")
    
    print("\n5. MEASUREMENT AND COLLAPSE:")
    print("   • Measurement destroys superposition")
    print("   • Collapses to classical state (|0⟩ or |1⟩)")
    print("   • Probabilistic outcome: P(|0⟩) = |α|², P(|1⟩) = |β|²")
    print("   • Irreversible process")
    print("   • Returns classical bit")
    
    print("\n6. QUANTUM ADVANTAGE:")
    print("   • Exponential state space: n qubits → 2^n states")
    print("   • Quantum parallelism: Process all states simultaneously")
    print("   • Interference: Amplify correct, cancel wrong answers")
    print("   • Entanglement: Correlations impossible classically")
    
    print("\n7. PHYSICAL REALIZATIONS:")
    print("   • Photon polarization (horizontal/vertical)")
    print("   • Electron spin (up/down)")
    print("   • Superconducting circuits (current direction)")
    print("   • Trapped ions (energy levels)")
    print("   • Nuclear spin (NMR)")
    
    print("\n8. MATHEMATICAL REPRESENTATION:")
    print("   • Vector notation: |ψ⟩ = [α, β]ᵀ")
    print("   • Bra-ket notation: |ψ⟩ = α|0⟩ + β|1⟩")
    print("   • Bloch sphere: Geometric representation")
    print("   • Normalization: ⟨ψ|ψ⟩ = 1")
    
    print("\n" + "="*70)


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("QUBIT FROM BIT - COMPREHENSIVE DEMONSTRATIONS")
    print("="*70)
    
    print("\n📊 Generating visualizations...")
    print("\n1. Classical to Quantum Transition...")
    plot_classical_to_quantum_transition()
    
    print("\n2. Superposition Construction...")
    plot_superposition_construction()
    
    print("\n3. Measurement Process...")
    plot_measurement_process()
    
    print("\n4. Quantum Advantage...")
    plot_quantum_advantage()
    
    print_key_concepts()
    
    print("\n✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  • classical_to_quantum_transition.png")
    print("  • superposition_construction.png")
    print("  • measurement_process.png")
    print("  • quantum_advantage.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
