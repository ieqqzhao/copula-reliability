"""
Two-component dependent failure model:
Weibull marginals + Clayton copula + controlled-RY quantum state preparation.

State convention:
    x_i = 0: component i is working at mission time t
    x_i = 1: component i has failed by mission time t

Install:
    pip install qiskit matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class WeibullComponent:
    eta: float   # scale parameter
    beta: float  # shape parameter

    def failure_probability(self, t: float) -> float:
        """F(t) = 1 - exp[-(t/eta)^beta]."""
        if self.eta <= 0 or self.beta <= 0:
            raise ValueError("eta and beta must be positive.")
        if t < 0:
            raise ValueError("Mission time t must be nonnegative.")
        return 1.0 - math.exp(-((t / self.eta) ** self.beta))


def clayton_copula(u: float, v: float, alpha: float) -> float:
    """
    Clayton copula:
        C(u,v; alpha) = (u^(-alpha) + v^(-alpha) - 1)^(-1/alpha)

    alpha > 0 gives positive lower-tail dependence.
    alpha = 0 is the independence limit C(u,v)=uv.
    """
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("u and v must be in [0, 1].")
    if alpha < 0:
        raise ValueError("This demonstration uses alpha >= 0.")

    if alpha == 0:
        return u * v
    if u == 0.0 or v == 0.0:
        return 0.0
    if u == 1.0:
        return v
    if v == 1.0:
        return u

    return (u ** (-alpha) + v ** (-alpha) - 1.0) ** (-1.0 / alpha)


def joint_binary_probabilities(
    f1: float, f2: float, copula_value: float
) -> dict[str, float]:
    """
    Return P(x1, x2), where 1 means failed:
        p11 = C(F1,F2)
        p10 = F1 - p11
        p01 = F2 - p11
        p00 = 1 - F1 - F2 + p11
    """
    probabilities = {
        "00": 1.0 - f1 - f2 + copula_value,
        "01": f2 - copula_value,
        "10": f1 - copula_value,
        "11": copula_value,
    }

    tolerance = 1e-12
    for state, value in probabilities.items():
        if value < -tolerance:
            raise ValueError(
                f"Invalid joint probability P({state})={value}. "
                "Check the copula and marginal probabilities."
            )
        probabilities[state] = max(0.0, value)

    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, abs_tol=1e-10):
        raise ValueError(f"Joint probabilities sum to {total}, not 1.")

    return probabilities


def conditional_rotation_angles(
    f1: float, probabilities: dict[str, float]
) -> tuple[float, float, float]:
    """
    Factor P(x1,x2)=P(x1)P(x2|x1).

    theta1      encodes P(x1=1)=F1.
    theta2_if_0 encodes P(x2=1 | x1=0).
    theta2_if_1 encodes P(x2=1 | x1=1).
    """
    eps = 1e-15

    p_x2_1_if_x1_0 = (
        probabilities["01"] / (1.0 - f1) if 1.0 - f1 > eps else 0.0
    )
    p_x2_1_if_x1_1 = (
        probabilities["11"] / f1 if f1 > eps else 0.0
    )

    def probability_to_ry_angle(p: float) -> float:
        p = min(1.0, max(0.0, p))
        return 2.0 * math.asin(math.sqrt(p))

    return (
        probability_to_ry_angle(f1),
        probability_to_ry_angle(p_x2_1_if_x1_0),
        probability_to_ry_angle(p_x2_1_if_x1_1),
    )


def build_dependent_failure_circuit(
    theta1: float, theta2_if_0: float, theta2_if_1: float
) -> QuantumCircuit:
    """
    q0 represents component 1.
    q1 represents component 2.

    Qiskit cry() uses a control-on-|1>.
    To implement control-on-|0>, sandwich the controlled rotation
    between X gates on the control qubit.
    """
    circuit = QuantumCircuit(2, name="Copula dependent failures")

    # Encode marginal P(x1=1).
    circuit.ry(theta1, 0)
    circuit.barrier()

    # Apply RY(theta2_if_0) to q1 when q0=0.
    circuit.x(0)
    circuit.cry(theta2_if_0, 0, 1)
    circuit.x(0)

    # Apply RY(theta2_if_1) to q1 when q0=1.
    circuit.cry(theta2_if_1, 0, 1)

    return circuit


def qiskit_probabilities_in_x1x2_order(
    circuit: QuantumCircuit,
) -> dict[str, float]:
    """
    Statevector.probabilities_dict() labels basis states as |q1 q0>.
    Convert labels to the model convention |x1 x2> = |q0 q1>.
    """
    raw = Statevector.from_instruction(circuit).probabilities_dict()
    converted: dict[str, float] = {}

    for qiskit_label, probability in raw.items():
        # Qiskit label is q1q0; reverse it to q0q1 = x1x2.
        x1x2 = qiskit_label[::-1]
        converted[x1x2] = float(probability)

    return dict(sorted(converted.items()))


def main() -> None:
    # ----- Model inputs -----
    mission_time = 500.0
    component_1 = WeibullComponent(eta=1000.0, beta=2.0)
    component_2 = WeibullComponent(eta=1200.0, beta=1.5)

    # Clayton alpha=0 means independence; larger positive values mean
    # stronger positive dependence for this copula family.
    clayton_alpha = 2.0

    # ----- Classical probability calculation -----
    f1 = component_1.failure_probability(mission_time)
    f2 = component_2.failure_probability(mission_time)
    c12 = clayton_copula(f1, f2, clayton_alpha)

    target = joint_binary_probabilities(f1, f2, c12)
    theta1, theta2_if_0, theta2_if_1 = conditional_rotation_angles(
        f1, target
    )

    # ----- Quantum circuit -----
    circuit = build_dependent_failure_circuit(
        theta1, theta2_if_0, theta2_if_1
    )
    quantum = qiskit_probabilities_in_x1x2_order(circuit)

    # ----- Output and verification -----
    print("Marginal failure probabilities")
    print(f"F1(t) = {f1:.8f}")
    print(f"F2(t) = {f2:.8f}")
    print(f"C(F1,F2) = P(11) = {c12:.8f}\n")

    print("Rotation angles, radians")
    print(f"theta1      = {theta1:.8f}")
    print(f"theta2|x1=0 = {theta2_if_0:.8f}")
    print(f"theta2|x1=1 = {theta2_if_1:.8f}\n")

    print("Target classical joint probabilities")
    for state, probability in target.items():
        print(f"P({state}) = {probability:.8f}")

    print("\nQuantum statevector probabilities")
    for state, probability in quantum.items():
        print(f"P({state}) = {probability:.8f}")

    max_error = max(abs(target[s] - quantum.get(s, 0.0)) for s in target)
    print(f"\nMaximum absolute error = {max_error:.3e}")

    print("\nCircuit")
    print(circuit.draw(output="text"))


if __name__ == "__main__":
    main()
