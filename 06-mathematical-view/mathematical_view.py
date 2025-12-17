#!/usr/bin/env python3
"""
Mathematical View of Qubits - Comprehensive Python Demonstrations

This script demonstrates the mathematical foundations of qubits using
linear algebra, Hilbert space theory, and quantum mechanics formalism.

Author: Devin (AI Software Engineer)
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
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


def plot_hilbert_space():
    """Visualize the Hilbert space structure for qubits"""
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(2, 3, 1)
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    ax1.plot(x_circle, y_circle, 'c-', linewidth=3, label='|α|² + |β|² = 1')
    ax1.fill(x_circle, y_circle, alpha=0.1, color='cyan')
    
    states = [
        (1, 0, '|0⟩', '#ff0000'),
        (0, 1, '|1⟩', '#00ffff'),
        (1/np.sqrt(2), 1/np.sqrt(2), '|+⟩', '#64ffda'),
        (np.sqrt(0.7), np.sqrt(0.3), 'custom', '#ffaa00')
    ]
    
    for alpha, beta, label, color in states:
        ax1.scatter([alpha], [beta], s=200, c=color, edgecolors='white', 
                   linewidths=2, zorder=3)
        ax1.annotate(label, (alpha, beta), xytext=(10, 10), 
                    textcoords='offset points', fontsize=11, color=color)
    
    ax1.set_xlabel('|α| (Amplitude for |0⟩)', fontsize=12, color='#64ffda')
    ax1.set_ylabel('|β| (Amplitude for |1⟩)', fontsize=12, color='#64ffda')
    ax1.set_title('Qubit State Space (Hilbert Space ℋ²)', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    
    ax2.quiver(0, 0, 0, 0, 0, 1, color='#ff0000', arrow_length_ratio=0.2, linewidth=3)
    ax2.quiver(0, 0, 0, 0, 0, -1, color='#00ffff', arrow_length_ratio=0.2, linewidth=3)
    ax2.quiver(0, 0, 0, 1, 0, 0, color='#64ffda', arrow_length_ratio=0.2, linewidth=3)
    ax2.quiver(0, 0, 0, 0, 1, 0, color='#ffaa00', arrow_length_ratio=0.2, linewidth=3)
    
    ax2.text(0, 0, 1.3, '|0⟩', fontsize=12, color='#ff0000')
    ax2.text(0, 0, -1.3, '|1⟩', fontsize=12, color='#00ffff')
    ax2.text(1.3, 0, 0, '|+⟩', fontsize=12, color='#64ffda')
    ax2.text(0, 1.3, 0, '|+i⟩', fontsize=12, color='#ffaa00')
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1, linewidth=0.5)
    
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X', color='#64ffda')
    ax2.set_ylabel('Y', color='#64ffda')
    ax2.set_zlabel('Z', color='#64ffda')
    ax2.set_title('Vector Representation', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    
    ax3 = fig.add_subplot(2, 3, 3)
    
    angles = np.linspace(0, 2*np.pi, 100)
    inner_products = []
    
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    for angle in angles:
        state = np.array([np.cos(angle/2), np.sin(angle/2)])
        inner_prod = np.abs(np.vdot(plus_state, state))**2
        inner_products.append(inner_prod)
    
    ax3.plot(angles * 180/np.pi, inner_products, 'c-', linewidth=3)
    ax3.fill_between(angles * 180/np.pi, 0, inner_products, alpha=0.3, color='cyan')
    
    special_angles = [0, 90, 180, 270, 360]
    special_labels = ['|0⟩', '|+i⟩', '|1⟩', '|−i⟩', '|0⟩']
    
    for angle, label in zip(special_angles, special_labels):
        angle_rad = angle * np.pi / 180
        state = np.array([np.cos(angle_rad/2), np.sin(angle_rad/2)])
        inner_prod = np.abs(np.vdot(plus_state, state))**2
        ax3.scatter([angle], [inner_prod], s=150, c='#64ffda', 
                   edgecolors='white', linewidths=2, zorder=3)
        ax3.text(angle, inner_prod+0.1, label, ha='center', fontsize=10)
    
    ax3.set_xlabel('State Parameter (degrees)', fontsize=12, color='#64ffda')
    ax3.set_ylabel('|⟨+|ψ⟩|²', fontsize=12, color='#64ffda')
    ax3.set_title('Inner Product with |+⟩', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    ax3.set_ylim(0, 1.1)
    
    ax4 = fig.add_subplot(2, 3, 4)
    
    basis_states = {
        '|0⟩': np.array([1, 0]),
        '|1⟩': np.array([0, 1]),
        '|+⟩': np.array([1, 1])/np.sqrt(2),
        '|−⟩': np.array([1, -1])/np.sqrt(2)
    }
    
    labels = list(basis_states.keys())
    n = len(labels)
    inner_prod_matrix = np.zeros((n, n))
    
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            inner_prod_matrix[i, j] = np.abs(np.vdot(basis_states[label1], 
                                                      basis_states[label2]))**2
    
    im = ax4.imshow(inner_prod_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(n))
    ax4.set_yticks(range(n))
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    
    for i in range(n):
        for j in range(n):
            text = ax4.text(j, i, f'{inner_prod_matrix[i, j]:.2f}',
                           ha="center", va="center", color="white", fontsize=11)
    
    ax4.set_title('Inner Products |⟨φ|ψ⟩|²', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Probability')
    
    ax5 = fig.add_subplot(2, 3, 5)
    
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    I = P0 + P1
    
    matrices = [P0, P1, I]
    titles = ['|0⟩⟨0|', '|1⟩⟨1|', '|0⟩⟨0| + |1⟩⟨1| = I']
    
    for idx, (mat, title) in enumerate(zip(matrices, titles)):
        ax_sub = plt.subplot(2, 3, 5, frameon=False)
        
        x_offset = idx * 0.35 - 0.35
        
        rect = Rectangle((x_offset, 0.3), 0.25, 0.4, 
                        facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=2)
        ax5.add_patch(rect)
        
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = '#64ffda' if val > 0 else '#666666'
                ax5.text(x_offset + 0.06 + j*0.13, 0.55 - i*0.15, 
                        f'{val:.0f}', ha='center', va='center', 
                        fontsize=14, color=color, fontweight='bold')
        
        ax5.text(x_offset + 0.125, 0.15, title, ha='center', 
                fontsize=10, color='#64ffda')
        
        if idx < 2:
            ax5.text(x_offset + 0.3, 0.5, '+', ha='center', 
                    fontsize=16, color='white', fontweight='bold')
        elif idx == 2:
            ax5.text(x_offset - 0.05, 0.5, '=', ha='center', 
                    fontsize=16, color='white', fontweight='bold')
    
    ax5.set_xlim(-0.5, 0.6)
    ax5.set_ylim(0, 1)
    ax5.set_title('Completeness Relation', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 3, 6)
    
    n_qubits = np.arange(1, 6)
    dimensions = 2**n_qubits
    
    ax6.semilogy(n_qubits, dimensions, 'o-', color='#64ffda', linewidth=3, 
                markersize=10, label='Hilbert space dimension')
    
    for n, dim in zip(n_qubits, dimensions):
        ax6.text(n, dim*1.5, f'2^{n}={dim}', ha='center', fontsize=10, color='#64ffda')
    
    ax6.set_xlabel('Number of Qubits', fontsize=12, color='#64ffda')
    ax6.set_ylabel('Hilbert Space Dimension', fontsize=12, color='#64ffda')
    ax6.set_title('Exponential Growth via Tensor Product', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('hilbert_space.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: hilbert_space.png")
    plt.close()


def plot_operators_and_gates():
    """Visualize quantum operators and gates"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = axes[0, 0]
    
    pauli_matrices = {
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]])
    }
    
    x_pos = 0
    for name, matrix in pauli_matrices.items():
        rect = Rectangle((x_pos, 0.3), 0.25, 0.4, 
                        facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=2)
        ax1.add_patch(rect)
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if np.iscomplex(val):
                    text = f'{val.imag:.0f}i' if val.imag != 0 else '0'
                else:
                    text = f'{val.real:.0f}'
                color = '#64ffda' if val != 0 else '#666666'
                ax1.text(x_pos + 0.06 + j*0.13, 0.55 - i*0.15, 
                        text, ha='center', va='center', 
                        fontsize=12, color=color, fontweight='bold')
        
        ax1.text(x_pos + 0.125, 0.15, name, ha='center', 
                fontsize=12, color='#64ffda', fontweight='bold')
        
        x_pos += 0.35
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Pauli Matrices', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax1.axis('off')
    
    ax2 = axes[0, 1]
    
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    
    gates = {'H': H, 'S': S}
    x_pos = 0
    
    for name, matrix in gates.items():
        rect = Rectangle((x_pos, 0.3), 0.35, 0.4, 
                        facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=2)
        ax2.add_patch(rect)
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if np.iscomplex(val):
                    if val.real != 0 and val.imag != 0:
                        text = f'{val.real:.2f}+{val.imag:.2f}i'
                    elif val.imag != 0:
                        text = f'{val.imag:.2f}i'
                    else:
                        text = f'{val.real:.2f}'
                else:
                    text = f'{val.real:.2f}'
                
                color = '#64ffda' if val != 0 else '#666666'
                ax2.text(x_pos + 0.08 + j*0.18, 0.55 - i*0.15, 
                        text, ha='center', va='center', 
                        fontsize=10, color=color, fontweight='bold')
        
        ax2.text(x_pos + 0.175, 0.15, name, ha='center', 
                fontsize=12, color='#64ffda', fontweight='bold')
        
        x_pos += 0.5
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Hadamard and Phase Gates', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax2.axis('off')
    
    ax3 = axes[0, 2]
    
    state_0 = np.array([1, 0])
    
    gates_action = {
        'I': np.eye(2),
        'X': pauli_matrices['X'],
        'H': H,
        'S': S
    }
    
    x_positions = np.arange(len(gates_action))
    
    for idx, (name, gate) in enumerate(gates_action.items()):
        result = gate @ state_0
        prob_0 = np.abs(result[0])**2
        prob_1 = np.abs(result[1])**2
        
        ax3.bar([idx - 0.15], [prob_0], width=0.3, color='#ff0000', alpha=0.7, label='P(|0⟩)' if idx == 0 else '')
        ax3.bar([idx + 0.15], [prob_1], width=0.3, color='#00ffff', alpha=0.7, label='P(|1⟩)' if idx == 0 else '')
    
    ax3.set_xlabel('Gate', fontsize=12, color='#64ffda')
    ax3.set_ylabel('Measurement Probability', fontsize=12, color='#64ffda')
    ax3.set_title('Gate Action on |0⟩', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(gates_action.keys())
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.1)
    
    ax4 = axes[1, 0]
    
    H_dagger = H.conj().T
    product = H_dagger @ H
    
    matrices_unitary = [H, H_dagger, product]
    titles_unitary = ['H', 'H†', 'H†H = I']
    
    for idx, (mat, title) in enumerate(zip(matrices_unitary, titles_unitary)):
        x_offset = idx * 0.35 - 0.35
        
        rect = Rectangle((x_offset, 0.3), 0.25, 0.4, 
                        facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=2)
        ax4.add_patch(rect)
        
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                if np.abs(val.imag) < 1e-10:
                    text = f'{val.real:.2f}'
                else:
                    text = f'{val.real:.1f}+{val.imag:.1f}i'
                
                color = '#64ffda' if np.abs(val) > 0.1 else '#666666'
                ax4.text(x_offset + 0.06 + j*0.13, 0.55 - i*0.15, 
                        text, ha='center', va='center', 
                        fontsize=9, color=color, fontweight='bold')
        
        ax4.text(x_offset + 0.125, 0.15, title, ha='center', 
                fontsize=10, color='#64ffda')
        
        if idx < 2:
            ax4.text(x_offset + 0.3, 0.5, '×', ha='center', 
                    fontsize=16, color='white', fontweight='bold')
        elif idx == 2:
            ax4.text(x_offset - 0.05, 0.5, '=', ha='center', 
                    fontsize=16, color='white', fontweight='bold')
    
    ax4.set_xlim(-0.5, 0.6)
    ax4.set_ylim(0, 1)
    ax4.set_title('Unitary Property: U†U = I', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax4.axis('off')
    
    ax5 = axes[1, 1]
    
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    
    projectors = {'P₀ = |0⟩⟨0|': P0, 'P₁ = |1⟩⟨1|': P1}
    x_pos = 0
    
    for name, proj in projectors.items():
        rect = Rectangle((x_pos, 0.3), 0.35, 0.4, 
                        facecolor='#1a1a2e', edgecolor='#64ffda', linewidth=2)
        ax5.add_patch(rect)
        
        for i in range(2):
            for j in range(2):
                val = proj[i, j]
                color = '#64ffda' if val > 0 else '#666666'
                ax5.text(x_pos + 0.08 + j*0.18, 0.55 - i*0.15, 
                        f'{val:.0f}', ha='center', va='center', 
                        fontsize=14, color=color, fontweight='bold')
        
        ax5.text(x_pos + 0.175, 0.15, name, ha='center', 
                fontsize=10, color='#64ffda')
        
        x_pos += 0.5
    
    ax5.set_xlim(-0.1, 1.1)
    ax5.set_ylim(0, 1)
    ax5.set_title('Measurement Projectors', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax5.axis('off')
    
    ax6 = axes[1, 2]
    
    X = pauli_matrices['X']
    Z = pauli_matrices['Z']
    HXH = H @ X @ H
    
    difference = np.max(np.abs(HXH - Z))
    
    ax6.text(0.5, 0.7, 'HXH = Z', ha='center', va='center', 
            fontsize=18, color='#64ffda', fontweight='bold')
    ax6.text(0.5, 0.5, f'Max difference: {difference:.2e}', ha='center', va='center', 
            fontsize=12, color='white')
    ax6.text(0.5, 0.3, 'Operators can be composed\nto create new operations', 
            ha='center', va='center', fontsize=11, color='white')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_title('Operator Composition', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('operators_and_gates.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: operators_and_gates.png")
    plt.close()


def plot_density_matrices():
    """Visualize density matrix formalism"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    ax1 = axes[0, 0]
    
    plus_state = np.array([[1], [1]]) / np.sqrt(2)
    rho_pure = plus_state @ plus_state.conj().T
    
    im1 = ax1.imshow(np.abs(rho_pure), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['|0⟩', '|1⟩'])
    ax1.set_yticklabels(['⟨0|', '⟨1|'])
    
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{rho_pure[i, j].real:.2f}',
                           ha="center", va="center", color="white", fontsize=12)
    
    ax1.set_title('Pure State: ρ = |+⟩⟨+|', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Magnitude')
    
    purity = np.trace(rho_pure @ rho_pure).real
    ax1.text(0.5, -0.5, f'Tr(ρ²) = {purity:.3f} (pure)', 
            ha='center', fontsize=11, color='#64ffda', 
            transform=ax1.transData)
    
    ax2 = axes[0, 1]
    
    state_0 = np.array([[1], [0]])
    state_1 = np.array([[0], [1]])
    rho_mixed = 0.5 * (state_0 @ state_0.conj().T) + 0.5 * (state_1 @ state_1.conj().T)
    
    im2 = ax2.imshow(np.abs(rho_mixed), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['|0⟩', '|1⟩'])
    ax2.set_yticklabels(['⟨0|', '⟨1|'])
    
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{rho_mixed[i, j].real:.2f}',
                           ha="center", va="center", color="white", fontsize=12)
    
    ax2.set_title('Mixed State: ρ = 0.5|0⟩⟨0| + 0.5|1⟩⟨1|', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Magnitude')
    
    purity_mixed = np.trace(rho_mixed @ rho_mixed).real
    ax2.text(0.5, -0.5, f'Tr(ρ²) = {purity_mixed:.3f} (mixed)', 
            ha='center', fontsize=11, color='#64ffda', 
            transform=ax2.transData)
    
    ax3 = axes[1, 0]
    
    purities = []
    mixedness = np.linspace(0, 1, 50)
    
    for p in mixedness:
        rho = (1-p) * (plus_state @ plus_state.conj().T) + p * rho_mixed
        purity = np.trace(rho @ rho).real
        purities.append(purity)
    
    ax3.plot(mixedness, purities, 'c-', linewidth=3)
    ax3.fill_between(mixedness, 0.5, purities, alpha=0.3, color='cyan')
    
    ax3.axhline(y=1, color='#ff0000', linestyle='--', linewidth=2, label='Pure (Tr(ρ²)=1)')
    ax3.axhline(y=0.5, color='#00ffff', linestyle='--', linewidth=2, label='Maximally mixed (Tr(ρ²)=0.5)')
    
    ax3.set_xlabel('Mixedness Parameter', fontsize=12, color='#64ffda')
    ax3.set_ylabel('Purity Tr(ρ²)', fontsize=12, color='#64ffda')
    ax3.set_title('Purity Spectrum', fontsize=13, 
                  color='#64ffda', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.4, 1.1)
    
    ax4 = axes[1, 1]
    
    properties = [
        'Hermitian: ρ† = ρ',
        'Positive: ⟨ψ|ρ|ψ⟩ ≥ 0',
        'Unit trace: Tr(ρ) = 1',
        'Pure: Tr(ρ²) = 1',
        'Mixed: Tr(ρ²) < 1'
    ]
    
    y_pos = 0.9
    for prop in properties:
        ax4.text(0.1, y_pos, '• ' + prop, fontsize=12, color='#64ffda', 
                va='top', fontfamily='monospace')
        y_pos -= 0.15
    
    ax4.text(0.5, 0.1, 'Density Matrix Formalism', ha='center', 
            fontsize=14, color='#64ffda', fontweight='bold')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('density_matrices.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: density_matrices.png")
    plt.close()


def plot_mathematical_summary():
    """Create a comprehensive mathematical summary visualization"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    fig.suptitle('Mathematical View of Qubits - Summary', fontsize=18, 
                color='#64ffda', fontweight='bold', y=0.98)
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_text = """
MATHEMATICAL FOUNDATIONS OF QUBITS

1. STATE SPACE
   • Hilbert space: ℋ² (2-dimensional complex vector space)
   • States: Unit vectors |ψ⟩ with ⟨ψ|ψ⟩ = 1
   • Representation: |ψ⟩ = α|0⟩ + β|1⟩ where α, β ∈ ℂ
   • Normalization: |α|² + |β|² = 1

2. INNER PRODUCT STRUCTURE
   • Definition: ⟨φ|ψ⟩ = α*₁α₂ + β*₁β₂
   • Probability: P(φ|ψ) = |⟨φ|ψ⟩|²
   • Orthonormality: ⟨i|j⟩ = δᵢⱼ (Kronecker delta)
   • Completeness: Σᵢ |i⟩⟨i| = I

3. OPERATORS AND GATES
   • Unitary operators: U†U = I (preserve normalization)
   • Pauli matrices: X, Y, Z (basis for single-qubit operations)
   • Hadamard: H = (X + Z)/√2 (creates superposition)
   • Phase gates: S, T (add relative phases)

4. MEASUREMENT
   • Projectors: Pᵢ = |i⟩⟨i| (project onto basis states)
   • Probability: P(i) = ⟨ψ|Pᵢ|ψ⟩
   • Collapse: |ψ⟩ → Pᵢ|ψ⟩/√⟨ψ|Pᵢ|ψ⟩
   • Completeness: Σᵢ Pᵢ = I

5. MULTI-QUBIT SYSTEMS
   • Tensor product: ℋ² ⊗ ℋ² = ℋ⁴
   • n qubits: (ℋ²)⊗ⁿ = ℋ^(2ⁿ)
   • Exponential growth: Enables quantum advantage
   • Entanglement: Non-separable states

6. DENSITY MATRIX FORMALISM
   • Pure state: ρ = |ψ⟩⟨ψ| with Tr(ρ²) = 1
   • Mixed state: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ| with Tr(ρ²) < 1
   • Properties: Hermitian, positive, unit trace
   • Generalization: Handles statistical mixtures

7. KEY MATHEMATICAL PROPERTIES
   • Linearity: Quantum mechanics is fundamentally linear
   • Unitarity: Time evolution preserves normalization
   • Hermiticity: Observables are Hermitian operators
   • Completeness: Basis sets span the entire space
   • Tensor structure: Enables multi-particle systems
"""
    
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
           fontsize=11, color='white', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', 
                    edgecolor='#64ffda', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('mathematical_summary.png', dpi=300, bbox_inches='tight', 
                facecolor='#0a0a0a')
    print("✓ Saved: mathematical_summary.png")
    plt.close()


def print_key_concepts():
    """Print key mathematical concepts"""
    print("\n" + "="*70)
    print("KEY CONCEPTS: MATHEMATICAL VIEW OF QUBITS")
    print("="*70)
    
    print("\n1. HILBERT SPACE STRUCTURE:")
    print("   • State space: ℋ² (2-dimensional complex vector space)")
    print("   • Inner product: ⟨φ|ψ⟩ defines angles and probabilities")
    print("   • Completeness: All Cauchy sequences converge")
    print("   • Normalization: ⟨ψ|ψ⟩ = 1 for all valid states")
    
    print("\n2. VECTOR REPRESENTATION:")
    print("   • Ket: |ψ⟩ = [α, β]ᵀ (column vector)")
    print("   • Bra: ⟨ψ| = [α*, β*] (row vector)")
    print("   • Inner product: ⟨φ|ψ⟩ (scalar)")
    print("   • Outer product: |ψ⟩⟨φ| (operator/matrix)")
    
    print("\n3. BASIS STATES:")
    print("   • Computational: {|0⟩, |1⟩}")
    print("   • Hadamard: {|+⟩, |−⟩}")
    print("   • Circular: {|+i⟩, |−i⟩}")
    print("   • All bases are orthonormal and complete")
    
    print("\n4. OPERATORS:")
    print("   • Unitary: U†U = I (preserve normalization)")
    print("   • Hermitian: A† = A (observables)")
    print("   • Projectors: P² = P (measurement)")
    print("   • Composition: Operators can be multiplied")
    
    print("\n5. MEASUREMENT:")
    print("   • Projectors: Pᵢ = |i⟩⟨i|")
    print("   • Probability: P(i) = ⟨ψ|Pᵢ|ψ⟩")
    print("   • Collapse: |ψ⟩ → Pᵢ|ψ⟩/√⟨ψ|Pᵢ|ψ⟩")
    print("   • Completeness: Σᵢ Pᵢ = I")
    
    print("\n6. TENSOR PRODUCT:")
    print("   • Multi-qubit: ℋ² ⊗ ℋ² = ℋ⁴")
    print("   • n qubits: (ℋ²)⊗ⁿ = ℋ^(2ⁿ)")
    print("   • Exponential growth")
    print("   • Enables entanglement")
    
    print("\n7. DENSITY MATRICES:")
    print("   • Pure state: ρ = |ψ⟩⟨ψ|, Tr(ρ²) = 1")
    print("   • Mixed state: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|, Tr(ρ²) < 1")
    print("   • Properties: Hermitian, positive, Tr(ρ) = 1")
    print("   • Handles statistical mixtures")
    
    print("\n8. KEY PROPERTIES:")
    print("   • Linearity: Superposition principle")
    print("   • Unitarity: Reversible evolution")
    print("   • Hermiticity: Real eigenvalues")
    print("   • Completeness: Basis expansion")
    
    print("\n" + "="*70)


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("MATHEMATICAL VIEW OF QUBITS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*70)
    
    print("\n📊 Generating visualizations...")
    print("\n1. Hilbert Space Structure...")
    plot_hilbert_space()
    
    print("\n2. Operators and Gates...")
    plot_operators_and_gates()
    
    print("\n3. Density Matrices...")
    plot_density_matrices()
    
    print("\n4. Mathematical Summary...")
    plot_mathematical_summary()
    
    print_key_concepts()
    
    print("\n✅ All visualizations complete!")
    print("\nGenerated files:")
    print("  • hilbert_space.png")
    print("  • operators_and_gates.png")
    print("  • density_matrices.png")
    print("  • mathematical_summary.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
