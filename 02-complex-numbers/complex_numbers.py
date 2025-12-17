"""
Complex Numbers Recap - Python Demonstrations
=============================================

This script provides comprehensive demonstrations of complex numbers including:
- Complex plane visualization
- Complex number operations
- Euler's formula
- Polar and Cartesian forms
- Applications in quantum computing

Author: QpiAI Quantum Engineer Course
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
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

def plot_complex_plane():
    """
    Visualize complex numbers on the complex plane
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_xlabel('Real Part (Re)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Imaginary Part (Im)', fontsize=14, fontweight='bold')
    ax.set_title('Complex Plane Representation', fontsize=16, fontweight='bold', pad=20)
    
    complex_numbers = [
        (3 + 4j, '#ff6b6b', '3+4i'),
        (-2 + 3j, '#4ecdc4', '-2+3i'),
        (2 - 2j, '#ffd93d', '2-2i'),
        (-3 - 1j, '#a8e6cf', '-3-i'),
        (1 + 1j, '#ff9999', '1+i'),
        (4 + 0j, '#64ffda', '4 (real)'),
        (0 + 3j, '#c7b3e5', '3i (imaginary)')
    ]
    
    for z, color, label in complex_numbers:
        ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        
        ax.plot(z.real, z.imag, 'o', color=color, markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label=label)
        
        offset_x = 0.3 if z.real >= 0 else -0.8
        offset_y = 0.3 if z.imag >= 0 else -0.5
        ax.text(z.real + offset_x, z.imag + offset_y, label, 
               fontsize=10, fontweight='bold', color=color)
        
        modulus = abs(z)
        ax.plot([0, z.real], [0, z.imag], '--', color=color, alpha=0.3, linewidth=1)
    
    circle = Circle((0, 0), 1, fill=False, color='green', linewidth=2, 
                   linestyle='--', alpha=0.5, label='Unit Circle')
    ax.add_patch(circle)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('complex_plane.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: complex_plane.png")
    plt.close()

def plot_complex_operations():
    """
    Visualize complex number operations
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Complex Number Operations', fontsize=18, fontweight='bold', y=1.00)
    
    z1 = 2 + 3j
    z2 = 1 + 1j
    
    ax = axes[0, 0]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Addition: z₁ + z₂', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    
    ax.annotate('', xy=(z1.real, z1.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.plot(z1.real, z1.imag, 'o', color='#ff6b6b', markersize=10, label='z₁ = 2+3i')
    
    ax.annotate('', xy=(z2.real, z2.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ecdc4'))
    ax.plot(z2.real, z2.imag, 'o', color='#4ecdc4', markersize=10, label='z₂ = 1+i')
    
    z_sum = z1 + z2
    ax.annotate('', xy=(z_sum.real, z_sum.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#64ffda'))
    ax.plot(z_sum.real, z_sum.imag, 'o', color='#64ffda', markersize=12, 
           markeredgecolor='black', markeredgewidth=2, label=f'z₁+z₂ = {z_sum.real:.0f}+{z_sum.imag:.0f}i')
    
    ax.plot([z1.real, z_sum.real], [z1.imag, z_sum.imag], '--', color='gray', alpha=0.5)
    ax.plot([z2.real, z_sum.real], [z2.imag, z_sum.imag], '--', color='gray', alpha=0.5)
    
    ax.legend(fontsize=10)
    
    ax = axes[0, 1]
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Multiplication: z₁ × z₂', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    
    ax.annotate('', xy=(z1.real, z1.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.plot(z1.real, z1.imag, 'o', color='#ff6b6b', markersize=10, label='z₁ = 2+3i')
    
    ax.annotate('', xy=(z2.real, z2.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ecdc4'))
    ax.plot(z2.real, z2.imag, 'o', color='#4ecdc4', markersize=10, label='z₂ = 1+i')
    
    z_prod = z1 * z2
    ax.annotate('', xy=(z_prod.real, z_prod.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#64ffda'))
    ax.plot(z_prod.real, z_prod.imag, 'o', color='#64ffda', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label=f'z₁×z₂ = {z_prod.real:.0f}+{z_prod.imag:.0f}i')
    
    ax.legend(fontsize=10)
    ax.text(0.5, 0.95, 'Magnitudes multiply\nAngles add', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)
    
    ax = axes[1, 0]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Complex Conjugate: z and z*', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    
    ax.annotate('', xy=(z1.real, z1.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#ff6b6b'))
    ax.plot(z1.real, z1.imag, 'o', color='#ff6b6b', markersize=10, label='z = 2+3i')
    
    z1_conj = np.conj(z1)
    ax.annotate('', xy=(z1_conj.real, z1_conj.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='#4ecdc4'))
    ax.plot(z1_conj.real, z1_conj.imag, 'o', color='#4ecdc4', markersize=10, label='z* = 2-3i')
    
    ax.plot([z1.real, z1_conj.real], [z1.imag, z1_conj.imag], '--', 
           color='gray', alpha=0.5, linewidth=2)
    
    ax.legend(fontsize=10)
    ax.text(0.5, 0.95, 'Reflection across\nreal axis', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)
    
    ax = axes[1, 1]
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_title('Modulus and Argument', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    
    ax.annotate('', xy=(z1.real, z1.imag), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color='#ff6b6b'))
    ax.plot(z1.real, z1.imag, 'o', color='#ff6b6b', markersize=12,
           markeredgecolor='black', markeredgewidth=2, label='z = 2+3i')
    
    modulus = abs(z1)
    ax.plot([0, z1.real], [0, z1.imag], 'g--', linewidth=2, label=f'|z| = {modulus:.2f}')
    
    argument = np.angle(z1)
    theta = np.linspace(0, argument, 50)
    r = 0.5
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'b-', linewidth=2, 
           label=f'arg(z) = {np.degrees(argument):.1f}°')
    
    ax.legend(fontsize=10)
    
    ax.text(z1.real/2 - 0.3, z1.imag/2 + 0.3, f'r = {modulus:.2f}', 
           fontsize=10, color='green', fontweight='bold')
    ax.text(0.8, 0.3, f'θ = {np.degrees(argument):.1f}°', 
           fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('complex_operations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: complex_operations.png")
    plt.close()

def plot_eulers_formula():
    """
    Visualize Euler's formula: e^(iθ) = cos(θ) + i sin(θ)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Euler's Formula: e^(iθ) = cos(θ) + i sin(θ)", 
                fontsize=18, fontweight='bold')
    
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=1.5)
    ax.axvline(x=0, color='k', linewidth=1.5)
    ax.set_xlabel('Real Part (cos θ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imaginary Part (sin θ)', fontsize=12, fontweight='bold')
    ax.set_title('Unit Circle Representation', fontsize=14, fontweight='bold')
    
    circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
    
    for theta, color in zip(angles, colors):
        z = np.exp(1j * theta)
        ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        ax.plot(z.real, z.imag, 'o', color=color, markersize=10,
               markeredgecolor='black', markeredgewidth=1.5)
        
        label_r = 1.3
        ax.text(label_r * np.cos(theta), label_r * np.sin(theta), 
               f'{np.degrees(theta):.0f}°',
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax = axes[1]
    theta_range = np.linspace(0, 2*np.pi, 200)
    real_part = np.cos(theta_range)
    imag_part = np.sin(theta_range)
    
    ax.plot(theta_range, real_part, 'b-', linewidth=2, label='Re(e^(iθ)) = cos(θ)')
    ax.plot(theta_range, imag_part, 'r-', linewidth=2, label='Im(e^(iθ)) = sin(θ)')
    ax.axhline(y=0, color='k', linewidth=1, linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('θ (radians)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Real and Imaginary Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    
    special_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    special_labels = ['0', 'π/2', 'π', '3π/2', '2π']
    ax.set_xticks(special_angles)
    ax.set_xticklabels(special_labels)
    
    plt.tight_layout()
    plt.savefig('eulers_formula.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: eulers_formula.png")
    plt.close()

def plot_polar_cartesian():
    """
    Visualize conversion between polar and Cartesian forms
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Polar and Cartesian Forms of Complex Numbers', 
                fontsize=18, fontweight='bold', y=1.00)
    
    examples = [
        (3 + 4j, 'z₁ = 3+4i'),
        (1 + 1j, 'z₂ = 1+i'),
        (-2 + 2j, 'z₃ = -2+2i'),
        (2 - 3j, 'z₄ = 2-3i')
    ]
    
    for idx, (z, label) in enumerate(examples):
        ax = axes[idx // 2, idx % 2]
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1.5)
        ax.axvline(x=0, color='k', linewidth=1.5)
        ax.set_xlabel('Real', fontsize=11)
        ax.set_ylabel('Imaginary', fontsize=11)
        
        r = abs(z)
        theta = np.angle(z)
        
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        
        circle = Circle((0, 0), 1, fill=False, color='gray', 
                       linewidth=1, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        
        ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', lw=3, color='#ff6b6b'))
        ax.plot(z.real, z.imag, 'o', color='#ff6b6b', markersize=12,
               markeredgecolor='black', markeredgewidth=2)
        
        ax.plot([0, z.real], [0, z.imag], 'g--', linewidth=2, alpha=0.7)
        
        if theta != 0:
            angle_arc = np.linspace(0, theta, 50)
            arc_r = min(r * 0.3, 0.8)
            ax.plot(arc_r * np.cos(angle_arc), arc_r * np.sin(angle_arc), 
                   'b-', linewidth=2)
        
        ax.text(0.05, 0.95, f'Cartesian:\n{z.real:.1f} + {z.imag:.1f}i',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        ax.text(0.05, 0.75, f'Polar:\nr = {r:.2f}\nθ = {np.degrees(theta):.1f}°',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               fontsize=10, fontweight='bold')
        
        ax.text(0.05, 0.50, f'Exponential:\n{r:.2f}e^(i·{theta:.2f})',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('polar_cartesian.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: polar_cartesian.png")
    plt.close()

def plot_quantum_states():
    """
    Visualize complex amplitudes in quantum states
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Complex Amplitudes in Quantum States', fontsize=18, fontweight='bold')
    
    states = [
        ([1, 0], '|0⟩', 'Computational basis'),
        ([0, 1], '|1⟩', 'Computational basis'),
        ([1/np.sqrt(2), 1/np.sqrt(2)], '|+⟩', 'Hadamard basis'),
        ([1/np.sqrt(2), -1/np.sqrt(2)], '|−⟩', 'Hadamard basis'),
        ([1/np.sqrt(2), 1j/np.sqrt(2)], '|+i⟩', 'Circular basis'),
        ([1/np.sqrt(2), -1j/np.sqrt(2)], '|−i⟩', 'Circular basis')
    ]
    
    for idx, (amplitudes, label, description) in enumerate(states):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)
        ax.set_xlabel('Real', fontsize=10)
        ax.set_ylabel('Imaginary', fontsize=10)
        ax.set_title(f'{label} - {description}', fontsize=12, fontweight='bold')
        
        circle = Circle((0, 0), 1, fill=False, color='gray', 
                       linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        colors = ['#ff6b6b', '#4ecdc4']
        labels_amp = ['α (for |0⟩)', 'β (for |1⟩)']
        
        for i, (amp, color, lbl) in enumerate(zip(amplitudes, colors, labels_amp)):
            if isinstance(amp, complex):
                real, imag = amp.real, amp.imag
            else:
                real, imag = amp, 0
            
            if abs(real) > 0.01 or abs(imag) > 0.01:
                ax.annotate('', xy=(real, imag), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', lw=2, color=color))
                ax.plot(real, imag, 'o', color=color, markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5)
            
            prob = abs(amp)**2
            ax.text(0.05, 0.95 - i*0.15, f'{lbl}\n|amp|² = {prob:.3f}',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig('quantum_states_complex.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: quantum_states_complex.png")
    plt.close()

def print_key_concepts():
    """
    Print key concepts about complex numbers
    """
    print("\n" + "="*70)
    print("KEY CONCEPTS: COMPLEX NUMBERS IN QUANTUM COMPUTING")
    print("="*70)
    
    print("\n1. COMPLEX NUMBER BASICS:")
    print("   • z = a + bi where i² = -1")
    print("   • Real part: Re(z) = a")
    print("   • Imaginary part: Im(z) = b")
    print("   • Complex conjugate: z* = a - bi")
    
    print("\n2. POLAR FORM:")
    print("   • z = r·e^(iθ) = r(cos θ + i sin θ)")
    print("   • Modulus: r = |z| = √(a² + b²)")
    print("   • Argument: θ = arg(z) = arctan(b/a)")
    
    print("\n3. EULER'S FORMULA:")
    print("   • e^(iθ) = cos θ + i sin θ")
    print("   • Euler's Identity: e^(iπ) + 1 = 0")
    print("   • Connects exponentials with trigonometry")
    
    print("\n4. OPERATIONS:")
    print("   • Addition: (a+bi) + (c+di) = (a+c) + (b+d)i")
    print("   • Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i")
    print("   • Conjugate: z·z* = |z|²")
    print("   • Division: z₁/z₂ = (z₁·z₂*)/(z₂·z₂*)")
    
    print("\n5. QUANTUM COMPUTING APPLICATIONS:")
    print("   • Quantum states: |ψ⟩ = α|0⟩ + β|1⟩ (α, β ∈ ℂ)")
    print("   • Normalization: |α|² + |β|² = 1")
    print("   • Probabilities: P(0) = |α|², P(1) = |β|²")
    print("   • Phase factors: e^(iφ) in quantum gates")
    print("   • Interference: Complex amplitudes enable quantum interference")
    
    print("\n6. IMPORTANT PROPERTIES:")
    print("   • |z₁·z₂| = |z₁|·|z₂| (moduli multiply)")
    print("   • arg(z₁·z₂) = arg(z₁) + arg(z₂) (arguments add)")
    print("   • (e^(iθ))^n = e^(inθ) (De Moivre's theorem)")
    print("   • |e^(iθ)| = 1 (unit modulus)")
    
    print("\n" + "="*70)

def main():
    """
    Main function to run all demonstrations
    """
    print("\n" + "="*70)
    print("COMPLEX NUMBERS RECAP - PYTHON DEMONSTRATIONS")
    print("="*70)
    
    print("\nGenerating visualizations...")
    print("-" * 70)
    
    plot_complex_plane()
    plot_complex_operations()
    plot_eulers_formula()
    plot_polar_cartesian()
    plot_quantum_states()
    
    print_key_concepts()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. complex_plane.png - Complex numbers on the complex plane")
    print("  2. complex_operations.png - Addition, multiplication, conjugate, modulus")
    print("  3. eulers_formula.png - Euler's formula visualization")
    print("  4. polar_cartesian.png - Polar and Cartesian form conversions")
    print("  5. quantum_states_complex.png - Complex amplitudes in quantum states")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
