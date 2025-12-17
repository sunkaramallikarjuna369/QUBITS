#!/usr/bin/env python3
"""
Physical View of Qubits - Comprehensive Python Demonstrations

This script demonstrates various physical implementations of qubits and their properties.
It covers different physical platforms, their characteristics, and visualization of
physical qubit systems.

Author: Devin AI
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle, Wedge
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


def plot_physical_implementations():
    """
    Visualize different physical qubit implementations and their characteristics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Physical Qubit Implementations', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Photon Polarization Qubit')
    
    x = np.linspace(-1.5, 1.5, 100)
    y = 0.5 * np.sin(4 * np.pi * x)
    ax.plot(x, y, 'b-', linewidth=2, label='Photon wave')
    
    ax.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    ax.text(0.3, 1, '|0⟩ (H)', fontsize=12, color='red')
    ax.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=2)
    ax.text(1, 0.3, '|1⟩ (V)', fontsize=12, color='green')
    
    ax.set_xlabel('Horizontal')
    ax.set_ylabel('Vertical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Electron Spin Qubit')
    
    circle = Circle((0, 0), 0.5, color='cyan', alpha=0.6)
    ax.add_patch(circle)
    ax.text(0, 0, 'e⁻', fontsize=16, ha='center', va='center', fontweight='bold')
    
    ax.arrow(0, 0.5, 0, 1, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=3)
    ax.text(0.3, 1.5, '|0⟩ (↑)', fontsize=12, color='red')
    ax.arrow(0, -0.5, 0, -1, head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=3)
    ax.text(0.3, -1.5, '|1⟩ (↓)', fontsize=12, color='green')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Spin axis)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Superconducting Qubit')
    
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, 'b-', linewidth=4, label='Superconducting loop')
    
    ax.plot([0.9, 1.1], [0, 0], 'r-', linewidth=6, label='Josephson junction')
    
    ax.plot([-1.5, -0.5], [-1, -1], 'g-', linewidth=3)
    ax.text(-1, -1.3, 'E₀ (|0⟩)', fontsize=10, color='green')
    ax.plot([-1.5, -0.5], [1, 1], 'r-', linewidth=3)
    ax.text(-1, 1.3, 'E₁ (|1⟩)', fontsize=10, color='red')
    
    ax.set_xlabel('Circuit dimension')
    ax.set_ylabel('Energy levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Trapped Ion Qubit')
    
    circle = Circle((0, 0), 0.3, color='yellow', alpha=0.8)
    ax.add_patch(circle)
    ax.text(0, 0, 'Ion', fontsize=10, ha='center', va='center', fontweight='bold')
    
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        x = 1.5 * np.cos(rad)
        y = 1.5 * np.sin(rad)
        rect = Rectangle((x-0.1, y-0.3), 0.2, 0.6, angle=angle, color='gray', alpha=0.7)
        ax.add_patch(rect)
    
    ax.arrow(-1.5, 0, 1, 0, head_width=0.15, head_length=0.15, fc='red', ec='red', 
             linewidth=2, alpha=0.7, label='Laser')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Quantum Dot Qubit')
    
    circle1 = Circle((-0.7, 0), 0.5, color='blue', alpha=0.5)
    circle2 = Circle((0.7, 0), 0.5, color='blue', alpha=0.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    electron = Circle((-0.7, 0), 0.15, color='red', alpha=0.8)
    ax.add_patch(electron)
    ax.text(-0.7, -1, '|0⟩', fontsize=12, ha='center', color='red')
    
    ax.plot([0.7], [0], 'go', markersize=10, alpha=0.5)
    ax.text(0.7, -1, '|1⟩', fontsize=12, ha='center', color='green')
    
    ax.plot([0, 0], [-0.8, 0.8], 'k--', linewidth=2, label='Tunnel barrier')
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.axis('off')
    ax.set_title('Implementation Comparison')
    
    comparison_text = """
    Platform Comparison:
    
    Coherence Time:
    • Trapped Ions: seconds
    • Photons: milliseconds
    • Superconducting: 10-100 μs
    • Quantum Dots: 1-10 μs
    
    Gate Speed:
    • Superconducting: nanoseconds
    • Quantum Dots: nanoseconds
    • Trapped Ions: microseconds
    • Photons: picoseconds
    
    Temperature:
    • Superconducting: ~20 mK
    • Quantum Dots: ~100 mK
    • Trapped Ions: room temp
    • Photons: room temp
    
    Scalability:
    • Superconducting: High
    • Quantum Dots: High
    • Trapped Ions: Medium
    • Photons: Low
    """
    
    ax.text(0.1, 0.9, comparison_text, fontsize=10, verticalalignment='top',
            family='monospace', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('physical_implementations.png', dpi=300, bbox_inches='tight')
    print("Saved: physical_implementations.png")
    plt.show()


def plot_decoherence_analysis():
    """
    Visualize decoherence processes and coherence times for different implementations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Decoherence Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    t = np.linspace(0, 100, 1000)
    
    platforms = {
        'Trapped Ion': 1000000,  # 1 second in microseconds
        'Photon': 1000,  # 1 ms in microseconds
        'Superconducting': 50,
        'Quantum Dot': 5
    }
    
    colors = ['green', 'blue', 'red', 'orange']
    for (platform, T2), color in zip(platforms.items(), colors):
        coherence = np.exp(-t / T2)
        ax.plot(t, coherence, label=f'{platform} (T₂={T2}μs)', linewidth=2, color=color)
    
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Coherence')
    ax.set_title('Coherence Decay Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    
    ax = axes[0, 1]
    noise_sources = ['Thermal', 'EM\nInterference', 'Material\nDefects', 
                     'Control\nErrors', 'Measurement\nBackaction']
    impact = [0.3, 0.25, 0.2, 0.15, 0.1]
    colors_noise = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen']
    
    bars = ax.bar(noise_sources, impact, color=colors_noise, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Relative Impact')
    ax.set_title('Decoherence Noise Sources')
    ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, impact):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    ax = axes[1, 0]
    platforms_list = list(platforms.keys())
    T1_times = [2000000, 2000, 100, 10]  # T1 times in microseconds
    T2_times = [1000000, 1000, 50, 5]    # T2 times in microseconds
    
    x = np.arange(len(platforms_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, T1_times, width, label='T₁ (Energy relaxation)', 
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, T2_times, width, label='T₂ (Dephasing)', 
                   color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Time (μs, log scale)')
    ax.set_title('T₁ vs T₂ Coherence Times')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms_list, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    
    gate_times = np.logspace(-3, 2, 100)  # From ns to 100 μs
    
    for (platform, T2), color in zip(platforms.items(), colors):
        fidelity = np.exp(-gate_times / T2)
        ax.plot(gate_times, fidelity, label=platform, linewidth=2, color=color)
    
    ax.set_xlabel('Gate Time (μs)')
    ax.set_ylabel('Gate Fidelity')
    ax.set_title('Gate Fidelity vs Operation Time')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax.axhline(y=0.99, color='red', linestyle='--', linewidth=2, 
               label='Fault-tolerant threshold', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('decoherence_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: decoherence_analysis.png")
    plt.show()


def plot_control_mechanisms():
    """
    Visualize different control mechanisms for physical qubits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Qubit Control Mechanisms', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    t = np.linspace(0, 10, 1000)
    
    omega_rabi = 2 * np.pi * 0.5  # Rabi frequency
    population_0 = np.cos(omega_rabi * t / 2) ** 2
    population_1 = np.sin(omega_rabi * t / 2) ** 2
    
    ax.plot(t, population_0, 'b-', linewidth=2, label='P(|0⟩)')
    ax.plot(t, population_1, 'r-', linewidth=2, label='P(|1⟩)')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title('Microwave Control: Rabi Oscillations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    pi_time = np.pi / omega_rabi
    pi_2_time = np.pi / (2 * omega_rabi)
    ax.axvline(x=pi_2_time, color='green', linestyle='--', alpha=0.7, label='π/2 pulse')
    ax.axvline(x=pi_time, color='orange', linestyle='--', alpha=0.7, label='π pulse')
    
    ax = axes[0, 1]
    
    time = np.linspace(0, 20, 1000)
    pulse1 = np.exp(-((time - 5) ** 2) / 0.5)
    pulse2 = np.exp(-((time - 10) ** 2) / 0.5)
    pulse3 = np.exp(-((time - 15) ** 2) / 0.5)
    
    ax.plot(time, pulse1, 'r-', linewidth=2, alpha=0.7)
    ax.plot(time, pulse2, 'r-', linewidth=2, alpha=0.7)
    ax.plot(time, pulse3, 'r-', linewidth=2, alpha=0.7)
    ax.fill_between(time, 0, pulse1, alpha=0.3, color='red')
    ax.fill_between(time, 0, pulse2, alpha=0.3, color='red')
    ax.fill_between(time, 0, pulse3, alpha=0.3, color='red')
    
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Laser Intensity')
    ax.set_title('Laser Control: Pulse Sequence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    ax.text(5, 1.1, 'Init', ha='center', fontsize=10)
    ax.text(10, 1.1, 'Gate', ha='center', fontsize=10)
    ax.text(15, 1.1, 'Measure', ha='center', fontsize=10)
    
    ax = axes[1, 0]
    
    time = np.linspace(0, 100, 1000)
    voltage = np.zeros_like(time)
    
    voltage[(time > 20) & (time < 40)] = 1.0
    voltage[(time > 50) & (time < 60)] = 0.5
    voltage[(time > 70) & (time < 90)] = 1.0
    
    ax.plot(time, voltage, 'b-', linewidth=2)
    ax.fill_between(time, 0, voltage, alpha=0.3, color='blue')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Gate Voltage (V)')
    ax.set_title('Voltage Control: Gate Pulses')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.3)
    
    ax.text(30, 1.15, 'X gate', ha='center', fontsize=10)
    ax.text(55, 0.65, 'H gate', ha='center', fontsize=10)
    ax.text(80, 1.15, 'X gate', ha='center', fontsize=10)
    
    ax = axes[1, 1]
    
    platforms_list = ['Trapped\nIon', 'Superconducting', 'Quantum\nDot', 'Photonic']
    single_qubit_fidelity = [0.9999, 0.999, 0.99, 0.98]
    two_qubit_fidelity = [0.999, 0.99, 0.95, 0.90]
    
    x = np.arange(len(platforms_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_qubit_fidelity, width, 
                   label='Single-qubit gates', color='lightgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, two_qubit_fidelity, width, 
                   label='Two-qubit gates', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Gate Fidelity')
    ax.set_title('Control Fidelity by Platform')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms_list)
    ax.legend()
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.axhline(y=0.99, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(3.5, 0.991, 'Fault-tolerant\nthreshold', fontsize=9, color='red')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('control_mechanisms.png', dpi=300, bbox_inches='tight')
    print("Saved: control_mechanisms.png")
    plt.show()


def plot_measurement_techniques():
    """
    Visualize different measurement techniques for physical qubits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Qubit Measurement Techniques', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    freq = np.linspace(4.5, 5.5, 1000)
    state_0 = np.exp(-((freq - 4.9) ** 2) / 0.01)
    state_1 = np.exp(-((freq - 5.1) ** 2) / 0.01)
    
    ax.plot(freq, state_0, 'b-', linewidth=2, label='|0⟩ state')
    ax.plot(freq, state_1, 'r-', linewidth=2, label='|1⟩ state')
    ax.fill_between(freq, 0, state_0, alpha=0.3, color='blue')
    ax.fill_between(freq, 0, state_1, alpha=0.3, color='red')
    
    ax.set_xlabel('Resonator Frequency (GHz)')
    ax.set_ylabel('Response Amplitude')
    ax.set_title('Dispersive Readout: Frequency Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.annotate('', xy=(5.1, 0.5), xytext=(4.9, 0.5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(5.0, 0.55, 'χ', fontsize=14, ha='center', color='green')
    
    ax = axes[0, 1]
    
    time = np.arange(0, 100, 1)
    bright_state = np.random.poisson(10, len(time))  # |0⟩ state - bright
    dark_state = np.random.poisson(0.5, len(time))   # |1⟩ state - dark
    
    ax.plot(time[:50], bright_state[:50], 'go-', linewidth=1, markersize=3, 
            label='|0⟩ (bright)', alpha=0.7)
    ax.plot(time[50:], dark_state[50:], 'ro-', linewidth=1, markersize=3, 
            label='|1⟩ (dark)', alpha=0.7)
    
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Photon Counts')
    ax.set_title('Fluorescence Detection: Photon Counting')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(90, 5.5, 'Threshold', fontsize=10)
    
    ax = axes[1, 0]
    
    voltage = np.linspace(-1, 1, 1000)
    current_0 = 1 / (1 + np.exp(-10 * (voltage + 0.3)))
    current_1 = 1 / (1 + np.exp(-10 * (voltage - 0.3)))
    
    ax.plot(voltage, current_0, 'b-', linewidth=2, label='|0⟩ state')
    ax.plot(voltage, current_1, 'r-', linewidth=2, label='|1⟩ state')
    
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Sensor Current (nA)')
    ax.set_title('Charge Sensing: Current vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0.05, 0.9, 'Operating\npoint', fontsize=10, color='green')
    
    ax = axes[1, 1]
    
    platforms_list = ['Trapped\nIon', 'Superconducting', 'Quantum\nDot', 'Photonic']
    readout_fidelity = [0.9999, 0.98, 0.95, 0.99]
    measurement_time = [10, 0.5, 1, 0.001]  # in microseconds
    
    ax2 = ax.twinx()
    
    x = np.arange(len(platforms_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, readout_fidelity, width, 
                   label='Readout fidelity', color='lightblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, measurement_time, width, 
                    label='Measurement time', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Readout Fidelity', color='blue')
    ax2.set_ylabel('Measurement Time (μs)', color='red')
    ax.set_xlabel('Platform')
    ax.set_title('Measurement Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms_list)
    ax.set_ylim(0.9, 1.0)
    ax2.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    for bar, val in zip(bars1, readout_fidelity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('measurement_techniques.png', dpi=300, bbox_inches='tight')
    print("Saved: measurement_techniques.png")
    plt.show()


def plot_scalability_analysis():
    """
    Visualize scalability challenges and progress for different platforms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    superconducting = np.array([5, 9, 16, 20, 53, 65, 127, 433, 1121, 1386])
    trapped_ion = np.array([2, 5, 5, 11, 11, 32, 32, 32, 56, 100])
    photonic = np.array([1, 2, 4, 8, 12, 20, 25, 30, 40, 50])
    
    ax.plot(years, superconducting, 'o-', linewidth=2, markersize=6, 
            label='Superconducting', color='blue')
    ax.plot(years, trapped_ion, 's-', linewidth=2, markersize=6, 
            label='Trapped Ion', color='green')
    ax.plot(years, photonic, '^-', linewidth=2, markersize=6, 
            label='Photonic', color='red')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Qubits')
    ax.set_title('Qubit Count Progress')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    
    challenges = ['Connectivity', 'Crosstalk', 'Control\nComplexity', 
                  'Cooling', 'Fabrication']
    supercond_challenge = [0.7, 0.6, 0.8, 0.9, 0.5]
    ion_challenge = [0.9, 0.3, 0.7, 0.2, 0.6]
    
    x = np.arange(len(challenges))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, supercond_challenge, width, 
                   label='Superconducting', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ion_challenge, width, 
                   label='Trapped Ion', color='lightgreen', alpha=0.8)
    
    ax.set_ylabel('Challenge Level')
    ax.set_title('Scalability Challenges (0=easy, 1=hard)')
    ax.set_xticks(x)
    ax.set_xticklabels(challenges)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    
    n_qubits = 5
    
    conn_super = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits - 1):
        conn_super[i, i+1] = 1
        conn_super[i+1, i] = 1
    
    im = ax.imshow(conn_super, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Superconducting: Nearest-Neighbor')
    ax.set_xlabel('Qubit')
    ax.set_ylabel('Qubit')
    
    for i in range(n_qubits + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    ax = axes[1, 1]
    
    conn_ion = np.ones((n_qubits, n_qubits)) - np.eye(n_qubits)
    
    im = ax.imshow(conn_ion, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Trapped Ion: All-to-All')
    ax.set_xlabel('Qubit')
    ax.set_ylabel('Qubit')
    
    for i in range(n_qubits + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: scalability_analysis.png")
    plt.show()


def print_key_concepts():
    """
    Print key concepts about physical qubit implementations.
    """
    print("\n" + "="*80)
    print("KEY CONCEPTS: Physical View of Qubits")
    print("="*80)
    
    print("\n1. PHYSICAL IMPLEMENTATIONS:")
    print("   • Photon Polarization: H/V polarization as |0⟩/|1⟩")
    print("   • Electron Spin: Spin up/down as |0⟩/|1⟩")
    print("   • Superconducting Circuits: Energy levels as |0⟩/|1⟩")
    print("   • Trapped Ions: Hyperfine states as |0⟩/|1⟩")
    print("   • Quantum Dots: Electron position as |0⟩/|1⟩")
    print("   • Topological: Anyonic states as |0⟩/|1⟩")
    
    print("\n2. KEY REQUIREMENTS:")
    print("   • Two-level quantum system")
    print("   • Superposition capability")
    print("   • Long coherence times")
    print("   • Precise control mechanisms")
    print("   • High-fidelity measurement")
    print("   • Scalability potential")
    
    print("\n3. DECOHERENCE:")
    print("   • Loss of quantum properties due to environment")
    print("   • Sources: thermal noise, EM interference, defects")
    print("   • T₁: Energy relaxation time")
    print("   • T₂: Dephasing time (T₂ ≤ 2T₁)")
    print("   • Limits computational time")
    
    print("\n4. CONTROL METHODS:")
    print("   • Microwave pulses: Superconducting qubits")
    print("   • Laser pulses: Trapped ions, NV centers")
    print("   • Voltage pulses: Quantum dots")
    print("   • Magnetic fields: Spin qubits")
    
    print("\n5. MEASUREMENT TECHNIQUES:")
    print("   • Dispersive readout: Superconducting (frequency shift)")
    print("   • Fluorescence: Trapped ions (photon counting)")
    print("   • Charge sensing: Quantum dots (current measurement)")
    print("   • Photon detection: Photonic qubits")
    
    print("\n6. PLATFORM COMPARISON:")
    print("   Superconducting:")
    print("     ✓ Fast gates (ns), scalable")
    print("     ✗ Short coherence (μs), requires extreme cooling")
    print("   Trapped Ion:")
    print("     ✓ Long coherence (s), high fidelity")
    print("     ✗ Slow gates (μs), difficult to scale")
    print("   Photonic:")
    print("     ✓ Room temperature, long coherence")
    print("     ✗ Probabilistic gates, hard to store")
    
    print("\n7. CURRENT STATE (2024):")
    print("   • Superconducting: 1000+ qubits (IBM, Google)")
    print("   • Trapped Ion: 100+ qubits (IonQ, Honeywell)")
    print("   • Photonic: Experimental stage")
    print("   • Neutral Atoms: 100+ qubits (QuEra)")
    print("   • Topological: Experimental stage (Microsoft)")
    
    print("\n" + "="*80 + "\n")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("PHYSICAL VIEW OF QUBITS - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    print("\nThis script demonstrates various physical implementations of qubits")
    print("and their properties, including decoherence, control, and measurement.\n")
    
    print_key_concepts()
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    print("\n1. Creating physical implementations visualization...")
    plot_physical_implementations()
    
    print("\n2. Creating decoherence analysis...")
    plot_decoherence_analysis()
    
    print("\n3. Creating control mechanisms visualization...")
    plot_control_mechanisms()
    
    print("\n4. Creating measurement techniques visualization...")
    plot_measurement_techniques()
    
    print("\n5. Creating scalability analysis...")
    plot_scalability_analysis()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • physical_implementations.png")
    print("  • decoherence_analysis.png")
    print("  • control_mechanisms.png")
    print("  • measurement_techniques.png")
    print("  • scalability_analysis.png")
    print("\nThese visualizations demonstrate the physical realization of qubits")
    print("across different platforms and their key characteristics.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
