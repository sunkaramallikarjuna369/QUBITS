"""
Qubits Practice Exercises - Python Verification and Demonstrations
This module provides answer verification and demonstrations for all exercises.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('exercise_outputs', exist_ok=True)


def verify_ex1_1():
    """Exercise 1.1: Fundamental difference between bit and qubit"""
    correct_answer = 'b'
    explanation = """
    The correct answer is B: A qubit can be in a superposition of 0 and 1, 
    while a bit is always either 0 or 1.
    
    Mathematical representation:
    - Classical bit: b ∈ {0, 1}
    - Qubit: |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
    """
    return correct_answer, explanation

def verify_ex1_2():
    """Exercise 1.2: Quantum properties"""
    correct_answer = 'c'
    explanation = """
    The correct answer is C: Deterministic measurement is NOT a quantum property.
    
    Quantum measurements are probabilistic:
    - Measuring |ψ⟩ = α|0⟩ + β|1⟩ gives |0⟩ with probability |α|²
    - And |1⟩ with probability |β|²
    
    True quantum properties:
    - Superposition: existing in multiple states simultaneously
    - Entanglement: correlations between particles
    - Interference: wave-like behavior of probability amplitudes
    """
    return correct_answer, explanation

def verify_ex1_3():
    """Exercise 1.3: Classical bits needed to describe n qubits"""
    correct_answer = 'c'
    explanation = """
    The correct answer is C: 2^n bits are needed.
    
    Reasoning:
    - An n-qubit system has 2^n basis states
    - Each basis state needs a complex amplitude (2 real numbers)
    - Total: 2^(n+1) real numbers ≈ 2^n complex numbers
    
    Example:
    - 1 qubit: 2¹ = 2 states (|0⟩, |1⟩)
    - 2 qubits: 2² = 4 states (|00⟩, |01⟩, |10⟩, |11⟩)
    - 3 qubits: 2³ = 8 states
    - 10 qubits: 2¹⁰ = 1024 states
    - 300 qubits: 2³⁰⁰ ≈ 10⁹⁰ states (more than atoms in universe!)
    """
    return correct_answer, explanation


def verify_ex2_1():
    """Exercise 2.1: Magnitude of complex number"""
    z = 3 + 4j
    correct_answer = abs(z)
    explanation = f"""
    Calculate |z| for z = 3 + 4i:
    
    |z| = √(Re(z)² + Im(z)²)
    |z| = √(3² + 4²)
    |z| = √(9 + 16)
    |z| = √25
    |z| = 5
    
    Verification: {abs(z)}
    """
    return correct_answer, explanation

def verify_ex2_2():
    """Exercise 2.2: Complex multiplication"""
    result = (1 + 1j) * (1 - 1j)
    correct_answer = result.real
    explanation = f"""
    Calculate (1 + i)(1 - i):
    
    Using FOIL method:
    = 1·1 + 1·(-i) + i·1 + i·(-i)
    = 1 - i + i - i²
    = 1 - i²
    = 1 - (-1)    [since i² = -1]
    = 2
    
    Verification: {result}
    """
    return correct_answer, explanation

def verify_ex2_3():
    """Exercise 2.3: Euler's formula"""
    result = np.exp(1j * np.pi / 2)
    correct_answer = result.real
    explanation = f"""
    Express e^(iπ/2) in the form a + bi:
    
    Using Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    
    e^(iπ/2) = cos(π/2) + i·sin(π/2)
    e^(iπ/2) = 0 + i·1
    e^(iπ/2) = i
    
    Therefore: a = 0, b = 1
    
    Verification: {result}
    """
    return correct_answer, explanation


def verify_ex3_1():
    """Exercise 3.1: Inner product"""
    psi = np.array([1, 1j])
    phi = np.array([1j, 1])
    inner_product = np.vdot(psi, phi)
    explanation = f"""
    Calculate ⟨ψ|φ⟩ where |ψ⟩ = (1, i) and |φ⟩ = (i, 1):
    
    ⟨ψ|φ⟩ = ψ₁*·φ₁ + ψ₂*·φ₂
    
    where * denotes complex conjugation:
    ψ₁* = (1)* = 1
    ψ₂* = (i)* = -i
    
    ⟨ψ|φ⟩ = (1)·(i) + (-i)·(1)
    ⟨ψ|φ⟩ = i + (-i)
    ⟨ψ|φ⟩ = 0
    
    Verification: {inner_product}
    Real part: {inner_product.real}
    Imaginary part: {inner_product.imag}
    """
    return inner_product.real, inner_product.imag, explanation

def verify_ex3_2():
    """Exercise 3.2: Vector normalization"""
    psi = np.array([0.6, 0.8])
    norm_squared = np.vdot(psi, psi).real
    is_normalized = np.isclose(norm_squared, 1.0)
    explanation = f"""
    Check if |ψ⟩ = (0.6, 0.8) is normalized:
    
    ||ψ||² = |ψ₁|² + |ψ₂|²
    ||ψ||² = (0.6)² + (0.8)²
    ||ψ||² = 0.36 + 0.64
    ||ψ||² = 1.0
    
    Since ||ψ||² = 1, the vector IS normalized.
    
    Verification: {norm_squared}
    Is normalized: {is_normalized}
    """
    return 1 if is_normalized else 0, explanation

def verify_ex3_3():
    """Exercise 3.3: Hilbert space dimension"""
    correct_answer = 2
    explanation = """
    Dimension of Hilbert space for a single qubit:
    
    A single qubit lives in a 2-dimensional complex Hilbert space ℂ².
    
    Basis states: {|0⟩, |1⟩}
    
    General state: |ψ⟩ = α|0⟩ + β|1⟩
    
    In vector form:
    |0⟩ = [1]    |1⟩ = [0]
          [0]          [1]
    
    Any qubit state can be written as a linear combination of these two basis vectors.
    """
    return correct_answer, explanation


def verify_ex4_1():
    """Exercise 4.1: No-cloning theorem"""
    correct_answer = 0  # No, you cannot clone
    explanation = """
    Can you clone an arbitrary quantum state?
    
    Answer: NO (0)
    
    The No-Cloning Theorem states that it is impossible to create an 
    identical copy of an arbitrary unknown quantum state.
    
    Proof sketch:
    - Suppose we have a unitary operator U that clones: U|ψ⟩|0⟩ = |ψ⟩|ψ⟩
    - For two different states |ψ⟩ and |φ⟩:
      U|ψ⟩|0⟩ = |ψ⟩|ψ⟩
      U|φ⟩|0⟩ = |φ⟩|φ⟩
    - Taking inner product: ⟨ψ|φ⟩ = ⟨ψ|φ⟩²
    - This only holds if ⟨ψ|φ⟩ = 0 or 1 (orthogonal or identical)
    - Therefore, universal cloning is impossible!
    
    This is fundamentally different from classical bits, which can be copied freely.
    """
    return correct_answer, explanation

def verify_ex4_2():
    """Exercise 4.2: Classical bit states"""
    correct_answer = 1
    explanation = """
    How many states can a classical bit be in at any given time?
    
    Answer: 1 (exactly ONE state)
    
    A classical bit is always definitively in one state:
    - Either 0
    - Or 1
    
    It cannot be in both states simultaneously.
    
    This contrasts with a qubit, which can be in a superposition:
    |ψ⟩ = α|0⟩ + β|1⟩
    
    The qubit exists in both states with amplitudes α and β until measured.
    """
    return correct_answer, explanation

def verify_ex4_3():
    """Exercise 4.3: Measurement collapse"""
    correct_answer = 'b'
    explanation = """
    What happens when you measure a qubit in superposition?
    
    Answer: B - It collapses to one of the basis states probabilistically
    
    Measurement process:
    1. Before measurement: |ψ⟩ = α|0⟩ + β|1⟩ (superposition)
    2. Measurement performed
    3. After measurement: 
       - |0⟩ with probability |α|²
       - |1⟩ with probability |β|²
    
    This is called "wavefunction collapse" or "state reduction."
    
    Example:
    |ψ⟩ = (|0⟩ + |1⟩)/√2
    - P(0) = |1/√2|² = 1/2 = 50%
    - P(1) = |1/√2|² = 1/2 = 50%
    
    The superposition is destroyed by measurement!
    """
    return correct_answer, explanation


def verify_ex5_1():
    """Exercise 5.1: Computational basis states"""
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    explanation = f"""
    Quantum state |0⟩ as a column vector:
    
    |0⟩ = [1]
          [0]
    
    |1⟩ = [0]
          [1]
    
    These are the computational basis states.
    They are orthonormal:
    - ⟨0|0⟩ = 1, ⟨1|1⟩ = 1 (normalized)
    - ⟨0|1⟩ = 0 (orthogonal)
    
    Verification:
    |0⟩ = {ket_0}
    |1⟩ = {ket_1}
    """
    return ket_0, explanation

def verify_ex5_2():
    """Exercise 5.2: Measurement probability"""
    correct_answer = 1.0
    explanation = """
    Probability of measuring |1⟩ when the state is |1⟩:
    
    P(1) = |⟨1|ψ⟩|²
    
    When |ψ⟩ = |1⟩:
    P(1) = |⟨1|1⟩|²
    P(1) = |1|²
    P(1) = 1
    
    The probability is 100% (certainty).
    
    This is because |1⟩ is an eigenstate of the computational basis measurement.
    Measuring an eigenstate always yields the corresponding eigenvalue with certainty.
    """
    return correct_answer, explanation

def verify_ex5_3():
    """Exercise 5.3: Plus state"""
    ket_plus = np.array([1, 1]) / np.sqrt(2)
    first_element = ket_plus[0]
    explanation = f"""
    The |+⟩ state in vector form:
    
    |+⟩ = (|0⟩ + |1⟩)/√2
    
    |+⟩ = 1/√2 [1] + 1/√2 [0]
                [0]         [1]
    
    |+⟩ = [1/√2]  ≈ [0.7071]
          [1/√2]     [0.7071]
    
    First element: 1/√2 ≈ 0.7071
    
    Verification: {ket_plus}
    First element: {first_element}
    """
    return first_element, explanation


def verify_ex6_1():
    """Exercise 6.1: Unitary operators"""
    correct_answer = 'b'
    explanation = """
    What property must quantum operators satisfy?
    
    Answer: B - U must be unitary (U†U = I)
    
    Unitary operators preserve:
    1. Normalization: ||Uψ|| = ||ψ||
    2. Inner products: ⟨Uψ|Uφ⟩ = ⟨ψ|φ⟩
    3. Probability: Total probability remains 1
    
    Mathematical definition:
    U is unitary if U†U = UU† = I
    
    where U† is the conjugate transpose (Hermitian adjoint) of U.
    
    Examples of unitary gates:
    - Pauli X, Y, Z
    - Hadamard H
    - Phase gates S, T
    - Rotation gates Rx, Ry, Rz
    """
    return correct_answer, explanation

def verify_ex6_2():
    """Exercise 6.2: Trace of Pauli X"""
    X = np.array([[0, 1], [1, 0]])
    trace_X = np.trace(X)
    explanation = f"""
    Trace of the Pauli X matrix:
    
    X = [0  1]
        [1  0]
    
    Tr(X) = sum of diagonal elements
    Tr(X) = 0 + 0
    Tr(X) = 0
    
    Verification: {trace_X}
    
    Note: All Pauli matrices have trace 0:
    Tr(X) = Tr(Y) = Tr(Z) = 0
    """
    return trace_X, explanation

def verify_ex6_3():
    """Exercise 6.3: Eigenvalues of Pauli Z"""
    Z = np.array([[1, 0], [0, -1]])
    eigenvalues = np.linalg.eigvals(Z)
    positive_eigenvalue = max(eigenvalues)
    explanation = f"""
    Eigenvalues of the Pauli Z matrix:
    
    Z = [1   0]
        [0  -1]
    
    For a diagonal matrix, eigenvalues are the diagonal elements.
    
    Eigenvalues: +1 and -1
    Eigenvectors: |0⟩ and |1⟩
    
    Z|0⟩ = (+1)|0⟩
    Z|1⟩ = (-1)|1⟩
    
    Positive eigenvalue: +1
    
    Verification: {eigenvalues}
    """
    return positive_eigenvalue, explanation


def verify_ex7_1():
    """Exercise 7.1: Coherence times"""
    correct_answer = 'b'
    explanation = """
    Which physical qubit has the longest coherence time?
    
    Answer: B - Trapped ion qubits
    
    Typical coherence times:
    - Superconducting qubits: ~10-100 microseconds
    - Trapped ions: seconds to minutes
    - NV centers: milliseconds to seconds
    - Photonic qubits: limited by propagation
    
    Why trapped ions have long coherence:
    1. Excellent isolation from environment
    2. Vacuum chamber reduces noise
    3. Laser cooling minimizes thermal effects
    4. Electromagnetic traps provide stable confinement
    
    Trade-off: Longer coherence but slower gate operations
    """
    return correct_answer, explanation

def verify_ex7_2():
    """Exercise 7.2: Operating temperature"""
    correct_answer = 15  # millikelvin
    tolerance = 10
    explanation = """
    Operating temperature of superconducting qubits:
    
    Typical range: 10-20 millikelvin (mK)
    
    This is approximately 0.01-0.02 Kelvin, which is:
    - Colder than outer space (~2.7 K)
    - Close to absolute zero (0 K = -273.15°C)
    
    Why so cold?
    - Superconductivity requires very low temperatures
    - Reduces thermal noise and decoherence
    - Achieved using dilution refrigerators
    
    Comparison:
    - Room temperature: ~300 K
    - Liquid nitrogen: 77 K
    - Liquid helium: 4.2 K
    - Superconducting qubits: 0.01-0.02 K
    """
    return correct_answer, tolerance, explanation

def verify_ex7_3():
    """Exercise 7.3: Decoherence"""
    correct_answer = 'b'
    explanation = """
    What is decoherence in quantum systems?
    
    Answer: B - Loss of quantum information due to environmental interaction
    
    Decoherence process:
    1. Quantum system starts in pure state |ψ⟩
    2. Unwanted interaction with environment
    3. System becomes entangled with environment
    4. Quantum coherence is lost
    5. System behaves more classically
    
    Types of decoherence:
    - T1 (energy relaxation): qubit loses energy
    - T2 (dephasing): relative phase is lost
    
    Causes:
    - Thermal fluctuations
    - Electromagnetic noise
    - Vibrations
    - Cosmic rays
    
    This is the main challenge in building quantum computers!
    """
    return correct_answer, explanation


def verify_ex8_1():
    """Exercise 8.1: Hadamard gate"""
    correct_answer = 'b'
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    ket_0 = np.array([1, 0])
    result = H @ ket_0
    explanation = f"""
    What does the Hadamard gate do to |0⟩?
    
    Answer: B - Creates the superposition (|0⟩ + |1⟩)/√2
    
    Hadamard matrix:
    H = 1/√2 [1   1]
             [1  -1]
    
    H|0⟩ = 1/√2 [1   1] [1]
                [1  -1] [0]
    
    H|0⟩ = 1/√2 [1]
                [1]
    
    H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩
    
    Verification: {result}
    
    The Hadamard gate creates equal superposition!
    """
    return correct_answer, explanation

def verify_ex8_2():
    """Exercise 8.2: X gate squared"""
    correct_answer = 'a'
    X = np.array([[0, 1], [1, 0]])
    X_squared = X @ X
    explanation = f"""
    What is X²|ψ⟩?
    
    Answer: A - |ψ⟩ (unchanged)
    
    Pauli X matrix:
    X = [0  1]
        [1  0]
    
    X² = X·X = [0  1] [0  1]
                [1  0] [1  0]
    
    X² = [1  0] = I (identity)
        [0  1]
    
    Therefore: X²|ψ⟩ = I|ψ⟩ = |ψ⟩
    
    Verification: X² = 
    {X_squared}
    
    The X gate is its own inverse!
    """
    return correct_answer, explanation

def verify_ex8_3():
    """Exercise 8.3: Universal gate set"""
    correct_answer = 'b'
    explanation = """
    Which gate set is universal for quantum computation?
    
    Answer: B - {H, T, CNOT}
    
    Universal gate set properties:
    - Can approximate any unitary operation to arbitrary precision
    - Combination of single-qubit and two-qubit gates
    - Forms a complete basis for quantum computation
    
    Why {H, T, CNOT} is universal:
    - H (Hadamard): Creates superposition
    - T (π/8 gate): Provides non-Clifford operation
    - CNOT: Enables entanglement
    
    Other universal sets:
    - {H, S, T, CNOT}
    - {Rx, Ry, Rz, CNOT}
    - {U3, CNOT} (where U3 is arbitrary single-qubit rotation)
    
    The key is having:
    1. Arbitrary single-qubit rotations
    2. At least one entangling two-qubit gate
    """
    return correct_answer, explanation


def verify_ex9_1():
    """Exercise 9.1: Measurement probability"""
    alpha = 1/np.sqrt(2)
    prob_0 = abs(alpha)**2
    explanation = f"""
    Probability of measuring |0⟩ for |ψ⟩ = (|0⟩ + |1⟩)/√2:
    
    |ψ⟩ = 1/√2 |0⟩ + 1/√2 |1⟩
    
    P(0) = |amplitude of |0⟩|²
    P(0) = |1/√2|²
    P(0) = 1/2
    P(0) = 0.5
    
    Similarly:
    P(1) = |1/√2|² = 0.5
    
    Total probability: P(0) + P(1) = 0.5 + 0.5 = 1 ✓
    
    Verification: {prob_0}
    """
    return prob_0, explanation

def verify_ex9_2():
    """Exercise 9.2: Bloch sphere utility"""
    correct_answer = 'a'
    explanation = """
    What is the Bloch sphere useful for?
    
    Answer: A - Visualizing single qubit states geometrically
    
    Bloch sphere representation:
    - Maps qubit states to points on a unit sphere
    - North pole: |0⟩
    - South pole: |1⟩
    - Equator: superposition states like |+⟩, |−⟩, |+i⟩, |−i⟩
    
    General state:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    Bloch coordinates:
    - θ: polar angle (0 to π)
    - φ: azimuthal angle (0 to 2π)
    
    Benefits:
    - Intuitive geometric visualization
    - Quantum gates become rotations
    - Easy to see state evolution
    
    Limitation: Only works for single qubits!
    """
    return correct_answer, explanation

def verify_ex9_3():
    """Exercise 9.3: Parameters for qubit state"""
    correct_answer = 2
    explanation = """
    How many real parameters specify a single qubit pure state?
    
    Answer: 2 parameters (ignoring global phase)
    
    General qubit state:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    
    Parameters:
    1. θ (theta): polar angle, range [0, π]
    2. φ (phi): azimuthal angle, range [0, 2π]
    
    These correspond to:
    - θ: "how much" |0⟩ vs |1⟩
    - φ: relative phase between |0⟩ and |1⟩
    
    On the Bloch sphere:
    - θ: angle from north pole
    - φ: angle around equator
    
    Note: We ignore global phase e^(iγ) because it has no physical effect.
    """
    return correct_answer, explanation


def verify_ex10_1():
    """Exercise 10.1: Hilbert space dimension"""
    n_qubits = 3
    dimension = 2**n_qubits
    explanation = f"""
    Dimension of Hilbert space for 3 qubits:
    
    Formula: dim = 2^n
    
    For n = 3:
    dim = 2³ = 8
    
    Basis states:
    |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
    
    General state:
    |ψ⟩ = Σ αᵢⱼₖ |ijk⟩
    
    where i,j,k ∈ {{0,1}} and Σ|αᵢⱼₖ|² = 1
    
    Exponential growth:
    - 1 qubit: 2¹ = 2
    - 2 qubits: 2² = 4
    - 3 qubits: 2³ = 8
    - 10 qubits: 2¹⁰ = 1,024
    - 50 qubits: 2⁵⁰ ≈ 10¹⁵
    - 300 qubits: 2³⁰⁰ ≈ 10⁹⁰ (more than atoms in universe!)
    """
    return dimension, explanation

def verify_ex10_2():
    """Exercise 10.2: CNOT gate"""
    correct_answer = 'b'
    CNOT = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
    explanation = f"""
    What does the CNOT gate do?
    
    Answer: B - Flips the target qubit if the control qubit is |1⟩
    
    CNOT truth table:
    |00⟩ → |00⟩  (control=0, target unchanged)
    |01⟩ → |01⟩  (control=0, target unchanged)
    |10⟩ → |11⟩  (control=1, target flipped)
    |11⟩ → |10⟩  (control=1, target flipped)
    
    Matrix representation:
    CNOT = [1  0  0  0]
           [0  1  0  0]
           [0  0  0  1]
           [0  0  1  0]
    
    Verification:
    {CNOT}
    
    CNOT is crucial for:
    - Creating entanglement
    - Quantum error correction
    - Universal quantum computation
    """
    return correct_answer, explanation

def verify_ex10_3():
    """Exercise 10.3: Tensor product notation"""
    correct_answer = 'c'
    explanation = """
    What is |0⟩ ⊗ |1⟩ written as?
    
    Answer: C - |01⟩
    
    Tensor product notation:
    |0⟩ ⊗ |1⟩ = |01⟩ = |0⟩₁|1⟩₂
    
    In vector form:
    |0⟩ ⊗ |1⟩ = [1] ⊗ [0] = [0]
                [0]   [1]   [1]
                            [0]
                            [0]
    
    This represents:
    - First qubit in state |0⟩
    - Second qubit in state |1⟩
    
    General notation equivalences:
    |ψ⟩ ⊗ |φ⟩ = |ψφ⟩ = |ψ⟩|φ⟩
    
    Examples:
    |0⟩ ⊗ |0⟩ = |00⟩
    |1⟩ ⊗ |0⟩ = |10⟩
    |+⟩ ⊗ |−⟩ = |+−⟩
    """
    return correct_answer, explanation


def verify_ex11_1():
    """Exercise 11.1: Bell states"""
    correct_answer = 'b'
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    explanation = f"""
    Which is a Bell state?
    
    Answer: B - (|00⟩ + |11⟩)/√2
    
    The four Bell states:
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    
    Properties:
    - Maximally entangled
    - Orthonormal basis for 2-qubit space
    - Cannot be written as |ψ⟩ ⊗ |φ⟩
    
    |Φ⁺⟩ in vector form:
    {bell_state}
    
    Creating |Φ⁺⟩:
    1. Start with |00⟩
    2. Apply H to first qubit: (|0⟩ + |1⟩)|0⟩/√2
    3. Apply CNOT: (|00⟩ + |11⟩)/√2
    """
    return correct_answer, explanation

def verify_ex11_2():
    """Exercise 11.2: Entangled state separability"""
    correct_answer = 0  # No, cannot be written as tensor product
    explanation = """
    Can an entangled state be written as a tensor product?
    
    Answer: NO (0)
    
    Definition of entanglement:
    A state |ψ⟩ is entangled if it CANNOT be written as |ψ₁⟩ ⊗ |ψ₂⟩
    
    Example - Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
    
    Suppose |Φ⁺⟩ = (α|0⟩ + β|1⟩) ⊗ (γ|0⟩ + δ|1⟩)
    
    Expanding:
    = αγ|00⟩ + αδ|01⟩ + βγ|10⟩ + βδ|11⟩
    
    For this to equal (|00⟩ + |11⟩)/√2:
    - αγ = 1/√2
    - αδ = 0
    - βγ = 0
    - βδ = 1/√2
    
    From αδ = 0: either α = 0 or δ = 0
    From βγ = 0: either β = 0 or γ = 0
    
    But then αγ = 0 or βδ = 0, contradiction!
    
    Therefore, |Φ⁺⟩ cannot be factorized.
    """
    return correct_answer, explanation

def verify_ex11_3():
    """Exercise 11.3: Spooky action at a distance"""
    correct_answer = 'b'
    explanation = """
    What did Einstein call "spooky action at a distance"?
    
    Answer: B - Quantum entanglement
    
    Einstein's concern (EPR paradox, 1935):
    - Two entangled particles separated by large distance
    - Measuring one instantly affects the other
    - Seems to violate locality (no faster-than-light signaling)
    
    Example with |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
    1. Alice and Bob share this state
    2. Alice measures her qubit → gets 0 or 1 (random, 50/50)
    3. Bob's qubit instantly becomes correlated
    4. If Alice got 0, Bob will definitely get 0
    5. If Alice got 1, Bob will definitely get 1
    
    Resolution:
    - Correlations are real and instantaneous
    - But no information is transmitted faster than light
    - Cannot use entanglement for communication
    - Bell's theorem (1964) proved quantum mechanics is correct
    
    This is now used for:
    - Quantum teleportation
    - Quantum cryptography
    - Quantum computing
    """
    return correct_answer, explanation


def verify_ex12_1():
    """Exercise 12.1: Bloch sphere poles"""
    correct_answer = 'a'
    explanation = """
    Where is |0⟩ on the Bloch sphere?
    
    Answer: A - North pole
    
    Bloch sphere mapping:
    - North pole (θ=0): |0⟩
    - South pole (θ=π): |1⟩
    - Equator (θ=π/2): superposition states
    
    Specific points:
    - (θ=0, φ=any): |0⟩
    - (θ=π, φ=any): |1⟩
    - (θ=π/2, φ=0): |+⟩ = (|0⟩+|1⟩)/√2
    - (θ=π/2, φ=π): |−⟩ = (|0⟩−|1⟩)/√2
    - (θ=π/2, φ=π/2): |+i⟩ = (|0⟩+i|1⟩)/√2
    - (θ=π/2, φ=3π/2): |−i⟩ = (|0⟩−i|1⟩)/√2
    
    General state:
    |ψ(θ,φ)⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    """
    return correct_answer, explanation

def verify_ex12_2():
    """Exercise 12.2: Quantum gates as rotations"""
    correct_answer = 'b'
    explanation = """
    What do quantum gates represent on the Bloch sphere?
    
    Answer: B - Rotations
    
    Common gates as rotations:
    
    X gate: π rotation around x-axis
    - |0⟩ → |1⟩ (north to south)
    - |1⟩ → |0⟩ (south to north)
    
    Y gate: π rotation around y-axis
    - |0⟩ → i|1⟩
    - |1⟩ → -i|0⟩
    
    Z gate: π rotation around z-axis
    - |+⟩ → |−⟩
    - |−⟩ → |+⟩
    
    H gate: π rotation around axis (x+z)/√2
    - |0⟩ → |+⟩
    - |1⟩ → |−⟩
    
    General rotation:
    Rₙ(θ) = exp(-iθn·σ/2)
    
    where n is the rotation axis and σ = (X,Y,Z)
    
    All single-qubit unitaries are rotations!
    """
    return correct_answer, explanation

def verify_ex12_3():
    """Exercise 12.3: Mixed states on Bloch sphere"""
    correct_answer = 0  # No, not on surface
    explanation = """
    Can mixed states be on the surface of the Bloch sphere?
    
    Answer: NO (0)
    
    Bloch sphere regions:
    - Surface: Pure states (||r|| = 1)
    - Interior: Mixed states (||r|| < 1)
    - Center: Completely mixed state (||r|| = 0)
    
    Pure state:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    Bloch vector: r = (sin θ cos φ, sin θ sin φ, cos θ)
    Length: ||r|| = 1 (on surface)
    
    Mixed state (density matrix):
    ρ = Σ pᵢ|ψᵢ⟩⟨ψᵢ|
    Bloch vector: r = Tr(ρσ)
    Length: ||r|| < 1 (inside sphere)
    
    Example - completely mixed state:
    ρ = I/2 = [0.5  0  ]
              [0    0.5]
    Bloch vector: r = (0, 0, 0) (center of sphere)
    
    The Bloch sphere beautifully visualizes purity:
    - Distance from center = purity
    - Surface = pure
    - Center = maximally mixed
    """
    return correct_answer, explanation


def verify_ex13_1():
    """Exercise 13.1: Shor's algorithm"""
    correct_answer = 'b'
    explanation = """
    Which algorithm provides exponential speedup for factoring?
    
    Answer: B - Shor's algorithm
    
    Shor's Algorithm (1994):
    - Factors N-bit numbers in O((log N)³) time
    - Classical best: O(exp((log N)^(1/3))) time
    - Exponential speedup!
    
    Impact:
    - Breaks RSA encryption (based on factoring)
    - Motivates post-quantum cryptography
    - Demonstrates quantum advantage
    
    Key steps:
    1. Reduce factoring to period finding
    2. Use Quantum Fourier Transform (QFT)
    3. Measure to find period
    4. Classical post-processing
    
    Example:
    - Factor 15 = 3 × 5
    - Classical: try divisors 2,3,4,...
    - Quantum: find period of f(x) = aˣ mod 15
    
    Other quantum algorithms:
    - Grover: √N speedup for search
    - Deutsch-Jozsa: exponential speedup for oracle problems
    - VQE: quantum chemistry simulations
    - QAOA: optimization problems
    """
    return correct_answer, explanation

def verify_ex13_2():
    """Exercise 13.2: Quantum supremacy"""
    correct_answer = 'b'
    explanation = """
    What is quantum supremacy (quantum advantage)?
    
    Answer: B - When a quantum computer solves a problem that classical 
    computers cannot solve in reasonable time
    
    Definition:
    Quantum supremacy is achieved when a quantum device performs a 
    computation that would be impractical for any classical computer.
    
    Milestones:
    - 2019: Google's Sycamore (53 qubits)
      - Random circuit sampling
      - 200 seconds vs 10,000 years (claimed)
    
    - 2020: USTC's Jiuzhang (photonic)
      - Gaussian boson sampling
      - 200 seconds vs 2.5 billion years (claimed)
    
    Important notes:
    - Not about practical applications (yet)
    - Demonstrates quantum advantage exists
    - Motivates further development
    
    Path to practical quantum advantage:
    1. Quantum supremacy (achieved)
    2. Quantum advantage for useful problems (in progress)
    3. Fault-tolerant quantum computing (future)
    """
    return correct_answer, explanation

def verify_ex13_3():
    """Exercise 13.3: Quantum computing challenges"""
    correct_answer = 'd'
    explanation = """
    Which is NOT a key challenge in building quantum computers?
    
    Answer: D - Lack of theoretical algorithms
    
    We have MANY quantum algorithms:
    - Shor's algorithm (factoring)
    - Grover's algorithm (search)
    - VQE (quantum chemistry)
    - QAOA (optimization)
    - Quantum simulation algorithms
    - HHL (linear systems)
    - And many more!
    
    The REAL challenges are hardware-related:
    
    1. Decoherence:
       - Qubits lose quantum information
       - Typical coherence: microseconds to seconds
       - Need: error correction
    
    2. Error rates:
       - Gate errors: 0.1% - 1%
       - Measurement errors: 1% - 5%
       - Need: < 0.01% for fault tolerance
    
    3. Scalability:
       - Current: ~100-1000 qubits
       - Need: millions for useful applications
       - Challenges: control, cooling, connectivity
    
    4. Additional challenges:
       - Qubit connectivity
       - Calibration and control
       - Cryogenic requirements
       - Cost and complexity
    
    The theory is ahead of the hardware!
    """
    return correct_answer, explanation


def create_exercise_summary_visualization():
    """Create a visual summary of all exercises"""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Qubits Practice Exercises - Complete Overview', fontsize=20, fontweight='bold')
    
    concepts = [
        ('01. Introduction', 3, '#667eea'),
        ('02. Complex Numbers', 3, '#764ba2'),
        ('03. Vectors', 3, '#f093fb'),
        ('04. Bit vs Qubit', 3, '#4facfe'),
        ('05. Qubit from Bit', 3, '#00f2fe'),
        ('06. Mathematical View', 3, '#43e97b'),
        ('07. Physical View', 3, '#38f9d7'),
        ('08. Computational View', 3, '#fa709a'),
        ('09. Single Qubit', 3, '#fee140'),
        ('10. Multi-Qubits', 3, '#30cfd0'),
        ('11. Entangled Qubits', 3, '#a8edea'),
        ('12. Bloch Sphere', 3, '#fed6e3'),
        ('13. Summary', 3, '#667eea'),
    ]
    
    for idx, (ax, (concept, num_ex, color)) in enumerate(zip(axes.flat, concepts)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        box = FancyBboxPatch((0.1, 0.3), 0.8, 0.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        
        ax.text(0.5, 0.6, concept, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        ax.text(0.5, 0.4, f'{num_ex} Exercises', ha='center', va='center',
                fontsize=9, color='white')
        
        ex_nums = [f'{idx+1}.{i+1}' for i in range(num_ex)]
        ax.text(0.5, 0.15, ' • '.join(ex_nums), ha='center', va='center',
                fontsize=8, color='black')
    
    for idx in range(len(concepts), len(axes.flat)):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('exercise_outputs/exercise_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Created exercise summary visualization")
    plt.close()

def create_concept_coverage_chart():
    """Create a chart showing exercise coverage across concepts"""
    concepts = ['Intro', 'Complex\nNumbers', 'Vectors', 'Bit vs\nQubit', 'Qubit\nfrom Bit',
                'Math\nView', 'Physical\nView', 'Comp\nView', 'Single\nQubit', 
                'Multi-\nQubits', 'Entangled\nQubits', 'Bloch\nSphere', 'Summary']
    exercises_per_concept = [3] * 13
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 13))
    bars = ax.bar(range(13), exercises_per_concept, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Concept', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Exercises', fontsize=14, fontweight='bold')
    ax.set_title('Exercise Distribution Across All 13 Concepts', fontsize=16, fontweight='bold')
    ax.set_xticks(range(13))
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_ylim(0, 5)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    total = sum(exercises_per_concept)
    ax.text(0.5, 0.95, f'Total: {total} Exercises', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('exercise_outputs/concept_coverage.png', dpi=150, bbox_inches='tight')
    print("✓ Created concept coverage chart")
    plt.close()


def run_all_verifications():
    """Run all exercise verifications and create visualizations"""
    print("=" * 70)
    print("QUBITS PRACTICE EXERCISES - VERIFICATION AND DEMONSTRATIONS")
    print("=" * 70)
    print()
    
    print("Creating visualizations...")
    create_exercise_summary_visualization()
    create_concept_coverage_chart()
    print()
    
    print("Sample Exercise Verifications:")
    print("-" * 70)
    
    print("\n[Exercise 1.1] Fundamental difference between bit and qubit")
    answer, explanation = verify_ex1_1()
    print(f"Correct Answer: {answer}")
    print(explanation)
    
    print("\n[Exercise 2.1] Magnitude of complex number")
    answer, explanation = verify_ex2_1()
    print(f"Correct Answer: {answer}")
    print(explanation)
    
    print("\n[Exercise 3.1] Inner product")
    real, imag, explanation = verify_ex3_1()
    print(f"Correct Answer: {real} + {imag}i")
    print(explanation)
    
    print("\n[Exercise 8.1] Hadamard gate")
    answer, explanation = verify_ex8_1()
    print(f"Correct Answer: {answer}")
    print(explanation)
    
    print("\n[Exercise 11.1] Bell states")
    answer, explanation = verify_ex11_1()
    print(f"Correct Answer: {answer}")
    print(explanation)
    
    print("\n" + "=" * 70)
    print("All 39 exercises have verification functions implemented!")
    print("Visualizations saved to exercise_outputs/")
    print("=" * 70)

if __name__ == "__main__":
    run_all_verifications()
