"""
Vectors Recap - Python Demonstrations
=====================================

This script provides comprehensive demonstrations of vectors including:
- Vector operations (addition, scalar multiplication, inner product)
- Vector spaces and basis vectors
- Orthogonality and normalization
- Quantum state vectors
- Tensor products for multi-qubit systems

Author: QpiAI Quantum Engineer Course
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_vector_operations():
    """
    Visualize basic vector operations
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Vector Operations', fontsize=18, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Vector Addition', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    v1 = np.array([2, 3])
    v2 = np.array([3, 1])
    v_sum = v1 + v2
    
    ax.annotate('', xy=v1, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.text(v1[0]/2, v1[1]/2 + 0.3, 'v₁', fontsize=12, fontweight='bold', color='#ff6b6b')
    
    ax.annotate('', xy=v2, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ecdc4'))
    ax.text(v2[0]/2 + 0.3, v2[1]/2, 'v₂', fontsize=12, fontweight='bold', color='#4ecdc4')
    
    ax.annotate('', xy=v_sum, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#64ffda'))
    ax.text(v_sum[0]/2, v_sum[1]/2 + 0.5, 'v₁+v₂', fontsize=12, fontweight='bold', color='#64ffda')
    
    ax.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], '--', color='gray', alpha=0.5)
    ax.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], '--', color='gray', alpha=0.5)
    
    ax = axes[0, 1]
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Scalar Multiplication', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    v = np.array([2, 3])
    scalars = [0.5, 1, 1.5, 2]
    colors = ['#ff9999', '#ff6b6b', '#cc5555', '#aa3333']
    
    for scalar, color in zip(scalars, colors):
        v_scaled = scalar * v
        ax.annotate('', xy=v_scaled, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        ax.text(v_scaled[0] + 0.3, v_scaled[1], f'{scalar}v', 
               fontsize=10, fontweight='bold', color=color)
    
    ax = axes[1, 0]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Inner Product (Dot Product)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    v1 = np.array([3, 2])
    v2 = np.array([2, 3])
    
    ax.annotate('', xy=v1, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.text(v1[0]/2, v1[1]/2 + 0.3, 'v₁', fontsize=12, fontweight='bold', color='#ff6b6b')
    
    ax.annotate('', xy=v2, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ecdc4'))
    ax.text(v2[0]/2 + 0.3, v2[1]/2, 'v₂', fontsize=12, fontweight='bold', color='#4ecdc4')
    
    inner_prod = np.dot(v1, v2)
    
    angle = np.arccos(inner_prod / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    theta = np.linspace(0, angle, 50)
    r = 0.8
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'g-', linewidth=2)
    
    ax.text(0.5, 0.95, f'⟨v₁|v₂⟩ = {inner_prod}\nAngle = {np.degrees(angle):.1f}°',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=11, fontweight='bold')
    
    ax = axes[1, 1]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Normalization', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    v = np.array([3, 4])
    v_norm = v / np.linalg.norm(v)
    
    ax.annotate('', xy=v, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.text(v[0]/2, v[1]/2 + 0.3, 'v', fontsize=12, fontweight='bold', color='#ff6b6b')
    
    ax.annotate('', xy=v_norm, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#64ffda'))
    ax.text(v_norm[0]/2 + 0.2, v_norm[1]/2, 'v̂', fontsize=12, fontweight='bold', color='#64ffda')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ax.text(0.5, 0.95, f'||v|| = {np.linalg.norm(v):.2f}\n||v̂|| = {np.linalg.norm(v_norm):.2f}',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('vector_operations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: vector_operations.png")
    plt.close()

def plot_3d_vectors():
    """
    Visualize vectors in 3D space
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Vectors in 3D Space', fontsize=16, fontweight='bold', pad=20)
    
    vectors = [
        ([3, 2, 1], '#ff6b6b', 'v₁'),
        ([1, 3, 2], '#4ecdc4', 'v₂'),
        ([-2, 2, 3], '#ffd93d', 'v₃'),
        ([2, -1, 2], '#a8e6cf', 'v₄')
    ]
    
    for v, color, label in vectors:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], 
                 color=color, arrow_length_ratio=0.15, linewidth=2.5, label=label)
        
        ax.scatter([v[0]], [v[1]], [v[2]], color=color, s=100, 
                  edgecolors='black', linewidths=1.5)
    
    basis_length = 4
    ax.quiver(0, 0, 0, basis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3, alpha=0.5)
    ax.quiver(0, 0, 0, 0, basis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=3, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, basis_length, color='blue', arrow_length_ratio=0.1, linewidth=3, alpha=0.5)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-1, 5])
    ax.legend(fontsize=11, loc='upper right')
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('3d_vectors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 3d_vectors.png")
    plt.close()

def plot_quantum_state_vectors():
    """
    Visualize quantum state vectors
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Quantum State Vectors', fontsize=18, fontweight='bold')
    
    states = [
        (np.array([1, 0]), '|0⟩', 'Computational basis'),
        (np.array([0, 1]), '|1⟩', 'Computational basis'),
        (np.array([1, 1])/np.sqrt(2), '|+⟩', 'Hadamard basis'),
        (np.array([1, -1])/np.sqrt(2), '|−⟩', 'Hadamard basis'),
        (np.array([1, 1j])/np.sqrt(2), '|+i⟩', 'Circular basis'),
        (np.array([1, -1j])/np.sqrt(2), '|−i⟩', 'Circular basis')
    ]
    
    for idx, (state, label, description) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        
        basis_labels = ['|0⟩', '|1⟩']
        real_parts = [state[0].real, state[1].real]
        imag_parts = [state[0].imag, state[1].imag]
        
        x = np.arange(len(basis_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_parts, width, label='Real', 
                      color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, imag_parts, width, label='Imaginary',
                      color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'{label} - {description}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1)
        
        norm = np.linalg.norm(state)
        ax.text(0.95, 0.95, f'||ψ|| = {norm:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('quantum_state_vectors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: quantum_state_vectors.png")
    plt.close()

def plot_basis_vectors():
    """
    Visualize different basis sets
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Different Basis Sets for Qubits', fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_title('Computational Basis', fontsize=14, fontweight='bold')
    ax.set_xlabel('|0⟩ component', fontsize=11)
    ax.set_ylabel('|1⟩ component', fontsize=11)
    
    ax.annotate('', xy=(1, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ff6b6b'))
    ax.text(0.5, -0.15, '|0⟩', fontsize=14, fontweight='bold', color='#ff6b6b', ha='center')
    
    ax.annotate('', xy=(0, 1), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#4ecdc4'))
    ax.text(-0.15, 0.5, '|1⟩', fontsize=14, fontweight='bold', color='#4ecdc4', va='center')
    
    ax = axes[1]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_title('Hadamard Basis', fontsize=14, fontweight='bold')
    ax.set_xlabel('|0⟩ component', fontsize=11)
    ax.set_ylabel('|1⟩ component', fontsize=11)
    
    plus = np.array([1, 1]) / np.sqrt(2)
    ax.annotate('', xy=plus, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#64ffda'))
    ax.text(plus[0] + 0.1, plus[1], '|+⟩', fontsize=14, fontweight='bold', color='#64ffda')
    
    minus = np.array([1, -1]) / np.sqrt(2)
    ax.annotate('', xy=minus, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ffd93d'))
    ax.text(minus[0] + 0.1, minus[1], '|−⟩', fontsize=14, fontweight='bold', color='#ffd93d')
    
    ax = axes[2]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_title('Circular Basis (Real Part)', fontsize=14, fontweight='bold')
    ax.set_xlabel('|0⟩ component', fontsize=11)
    ax.set_ylabel('|1⟩ component (Real)', fontsize=11)
    
    plus_i = np.array([1, 0]) / np.sqrt(2)  # Real part only
    ax.annotate('', xy=plus_i, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#a8e6cf'))
    ax.text(plus_i[0], plus_i[1] + 0.15, '|+i⟩', fontsize=14, fontweight='bold', color='#a8e6cf', ha='center')
    
    minus_i = np.array([1, 0]) / np.sqrt(2)  # Real part only
    ax.annotate('', xy=minus_i, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ff9999'))
    ax.text(minus_i[0], minus_i[1] - 0.15, '|−i⟩', fontsize=14, fontweight='bold', color='#ff9999', ha='center')
    
    ax.text(0.5, 0.05, 'Note: Imaginary parts not shown',
           transform=ax.transAxes, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
           fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('basis_vectors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: basis_vectors.png")
    plt.close()

def plot_orthogonality():
    """
    Visualize orthogonal and orthonormal vectors
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Orthogonality and Orthonormality', fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_title('Orthogonal Vectors', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    v1 = np.array([3, 0])
    v2 = np.array([0, 3])
    
    ax.annotate('', xy=v1, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ff6b6b'))
    ax.text(v1[0]/2, -0.3, 'v₁', fontsize=14, fontweight='bold', color='#ff6b6b', ha='center')
    
    ax.annotate('', xy=v2, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#4ecdc4'))
    ax.text(-0.3, v2[1]/2, 'v₂', fontsize=14, fontweight='bold', color='#4ecdc4', va='center')
    
    square_size = 0.5
    square = plt.Rectangle((0, 0), square_size, square_size, 
                           fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(square)
    
    inner_prod = np.dot(v1, v2)
    ax.text(0.5, 0.95, f'⟨v₁|v₂⟩ = {inner_prod}\nOrthogonal!',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           fontsize=12, fontweight='bold')
    
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_title('Orthonormal Vectors', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    ax.annotate('', xy=v1, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ff6b6b'))
    ax.text(v1[0]/2, -0.2, 'v₁', fontsize=14, fontweight='bold', color='#ff6b6b', ha='center')
    
    ax.annotate('', xy=v2, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#4ecdc4'))
    ax.text(-0.2, v2[1]/2, 'v₂', fontsize=14, fontweight='bold', color='#4ecdc4', va='center')
    
    square_size = 0.3
    square = plt.Rectangle((0, 0), square_size, square_size, 
                           fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(square)
    
    inner_prod = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    ax.text(0.5, 0.95, f'⟨v₁|v₂⟩ = {inner_prod}\n||v₁|| = {norm1:.1f}\n||v₂|| = {norm2:.1f}\nOrthonormal!',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('orthogonality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: orthogonality.png")
    plt.close()

def plot_tensor_products():
    """
    Visualize tensor products for multi-qubit systems
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Tensor Products for Multi-Qubit Systems', fontsize=18, fontweight='bold')
    
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    
    two_qubit_states = [
        (np.kron(ket_0, ket_0), '|00⟩', 'Both qubits in |0⟩'),
        (np.kron(ket_0, ket_1), '|01⟩', 'First |0⟩, second |1⟩'),
        (np.kron(ket_1, ket_0), '|10⟩', 'First |1⟩, second |0⟩'),
        (np.kron(ket_1, ket_1), '|11⟩', 'Both qubits in |1⟩')
    ]
    
    for idx, (state, label, description) in enumerate(two_qubit_states):
        ax = axes[idx // 2, idx % 2]
        
        basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        amplitudes = state
        
        bars = ax.bar(basis_labels, amplitudes, 
                     color='#64ffda', alpha=0.8, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'{label} - {description}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        for i, (bar, amp) in enumerate(zip(bars, amplitudes)):
            if amp > 0:
                bar.set_color('#ff6b6b')
                ax.text(i, amp + 0.05, f'{amp:.1f}', 
                       ha='center', fontsize=12, fontweight='bold')
        
        state_str = '[' + ', '.join([f'{x:.0f}' for x in state]) + ']ᵀ'
        ax.text(0.95, 0.95, f'State vector:\n{state_str}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tensor_products.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tensor_products.png")
    plt.close()

def print_key_concepts():
    """
    Print key concepts about vectors
    """
    print("\n" + "="*70)
    print("KEY CONCEPTS: VECTORS IN QUANTUM COMPUTING")
    print("="*70)
    
    print("\n1. VECTOR BASICS:")
    print("   • Ordered list of numbers: v = [v₁, v₂, ..., vₙ]ᵀ")
    print("   • Can be real or complex")
    print("   • Quantum states are vectors in Hilbert space")
    
    print("\n2. VECTOR OPERATIONS:")
    print("   • Addition: u + v (component-wise)")
    print("   • Scalar multiplication: c·v")
    print("   • Inner product: ⟨u|v⟩ = Σᵢ uᵢ*vᵢ")
    print("   • Norm: ||v|| = √⟨v|v⟩")
    print("   • Normalization: v̂ = v/||v||")
    
    print("\n3. BASIS VECTORS:")
    print("   • Computational: {|0⟩, |1⟩}")
    print("   • Hadamard: {|+⟩, |−⟩}")
    print("   • Circular: {|+i⟩, |−i⟩}")
    print("   • Any state: |ψ⟩ = α|0⟩ + β|1⟩")
    
    print("\n4. ORTHOGONALITY:")
    print("   • Orthogonal: ⟨u|v⟩ = 0")
    print("   • Orthonormal: ⟨eᵢ|eⱼ⟩ = δᵢⱼ")
    print("   • Basis vectors are orthonormal")
    
    print("\n5. TENSOR PRODUCTS:")
    print("   • Multi-qubit states: |ψ⟩ ⊗ |φ⟩ = |ψφ⟩")
    print("   • Two qubits: ℂ² ⊗ ℂ² = ℂ⁴")
    print("   • Basis: {|00⟩, |01⟩, |10⟩, |11⟩}")
    
    print("\n6. QUANTUM APPLICATIONS:")
    print("   • State representation: Vectors in Hilbert space")
    print("   • Superposition: Linear combinations")
    print("   • Measurement: Inner products → probabilities")
    print("   • Evolution: Linear transformations (gates)")
    print("   • Entanglement: Tensor products")
    
    print("\n" + "="*70)

def main():
    """
    Main function to run all demonstrations
    """
    print("\n" + "="*70)
    print("VECTORS RECAP - PYTHON DEMONSTRATIONS")
    print("="*70)
    
    print("\nGenerating visualizations...")
    print("-" * 70)
    
    plot_vector_operations()
    plot_3d_vectors()
    plot_quantum_state_vectors()
    plot_basis_vectors()
    plot_orthogonality()
    plot_tensor_products()
    
    print_key_concepts()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. vector_operations.png - Basic vector operations")
    print("  2. 3d_vectors.png - Vectors in 3D space")
    print("  3. quantum_state_vectors.png - Quantum state representations")
    print("  4. basis_vectors.png - Different basis sets")
    print("  5. orthogonality.png - Orthogonal and orthonormal vectors")
    print("  6. tensor_products.png - Multi-qubit tensor products")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
