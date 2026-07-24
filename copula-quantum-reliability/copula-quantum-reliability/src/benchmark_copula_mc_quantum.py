"""
Benchmark three methods for a two-component dependent-failure probability:

1. Analytic solution
2. Classical inverse-transform Monte Carlo
3. Quantum-circuit implementation:
   a) exact statevector simulation;
   b) shot-based sampling of the quantum circuit.

Model:
    Weibull marginal lifetimes + Clayton copula dependence.

Target event:
    both components fail by mission time t, i.e. P(X1=1, X2=1).

Important:
    The statevector and shot-based quantum results below are executed on a
    classical simulator. Their wall-clock time is NOT evidence of quantum
    speedup. For algorithmic comparison, also report:
        classical Monte Carlo error = O(N^{-1/2})
        quantum amplitude estimation error = O(M^{-1})
    where N is the number of classical samples and M is the number of
    applications of the Grover/amplitude operator.

Install:
    pip install numpy pandas matplotlib qiskit

Run:
    python benchmark_copula_mc_quantum.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeibullComponent:
    eta: float
    beta: float

    def cdf(self, t: float | np.ndarray) -> float | np.ndarray:
        if self.eta <= 0 or self.beta <= 0:
            raise ValueError("eta and beta must be positive.")
        return 1.0 - np.exp(-np.power(np.asarray(t) / self.eta, self.beta))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse Weibull CDF."""
        if np.any((u <= 0.0) | (u >= 1.0)):
            raise ValueError("u must lie strictly inside (0,1).")
        return self.eta * np.power(-np.log1p(-u), 1.0 / self.beta)


def clayton_copula_cdf(u: float, v: float, alpha: float) -> float:
    if alpha < 0:
        raise ValueError("This example uses alpha >= 0.")
    if alpha == 0:
        return u * v
    if u == 0.0 or v == 0.0:
        return 0.0
    return (u ** (-alpha) + v ** (-alpha) - 1.0) ** (-1.0 / alpha)


def sample_clayton_copula(
    n: int, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Clayton-copula uniforms using the Marshall-Olkin representation.

    For alpha > 0:
        W ~ Gamma(shape=1/alpha, scale=1)
        E1,E2 ~ Exp(1)
        Ui = (1 + Ei/W)^(-1/alpha)

    For alpha = 0, return independent uniforms.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if alpha < 0:
        raise ValueError("This example uses alpha >= 0.")

    if alpha == 0:
        return rng.random(n), rng.random(n)

    w = rng.gamma(shape=1.0 / alpha, scale=1.0, size=n)
    e1 = rng.exponential(scale=1.0, size=n)
    e2 = rng.exponential(scale=1.0, size=n)

    u1 = np.power(1.0 + e1 / w, -1.0 / alpha)
    u2 = np.power(1.0 + e2 / w, -1.0 / alpha)
    return u1, u2


def analytic_joint_failure_probability(
    t: float,
    comp1: WeibullComponent,
    comp2: WeibullComponent,
    alpha: float,
) -> float:
    f1 = float(comp1.cdf(t))
    f2 = float(comp2.cdf(t))
    return clayton_copula_cdf(f1, f2, alpha)


def inverse_transform_mc(
    n: int,
    t: float,
    comp1: WeibullComponent,
    comp2: WeibullComponent,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """
    Full inverse-transform implementation:
        (U1,U2) ~ Clayton copula
        Ti = Fi^{-1}(Ui)
        estimate P(T1 <= t, T2 <= t)

    Returns:
        estimate, estimated standard error
    """
    rng = np.random.default_rng(seed)

    # Keep values away from exact 0 and 1 for numerical stability.
    eps = np.finfo(float).eps
    u1, u2 = sample_clayton_copula(n, alpha, rng)
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)

    lifetime1 = comp1.ppf(u1)
    lifetime2 = comp2.ppf(u2)

    indicator = (lifetime1 <= t) & (lifetime2 <= t)
    estimate = float(np.mean(indicator))
    std_error = math.sqrt(max(estimate * (1.0 - estimate), 0.0) / n)
    return estimate, std_error


def joint_state_probabilities(
    t: float,
    comp1: WeibullComponent,
    comp2: WeibullComponent,
    alpha: float,
) -> dict[str, float]:
    f1 = float(comp1.cdf(t))
    f2 = float(comp2.cdf(t))
    p11 = clayton_copula_cdf(f1, f2, alpha)

    probabilities = {
        "00": 1.0 - f1 - f2 + p11,
        "01": f2 - p11,
        "10": f1 - p11,
        "11": p11,
    }

    for key, value in probabilities.items():
        if value < -1e-12:
            raise ValueError(f"Invalid joint probability {key}={value}")
        probabilities[key] = max(0.0, float(value))

    return probabilities


def probability_to_ry_angle(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 2.0 * math.asin(math.sqrt(p))


def build_quantum_circuit(
    t: float,
    comp1: WeibullComponent,
    comp2: WeibullComponent,
    alpha: float,
):
    """
    Build the conditional-probability circuit:
        q0 = X1
        q1 = X2
        1 means failed.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit is not installed. Run: pip install qiskit"
        ) from exc

    probabilities = joint_state_probabilities(t, comp1, comp2, alpha)
    f1 = probabilities["10"] + probabilities["11"]

    p2_if_0 = probabilities["01"] / max(
        probabilities["00"] + probabilities["01"], 1e-15
    )
    p2_if_1 = probabilities["11"] / max(
        probabilities["10"] + probabilities["11"], 1e-15
    )

    theta1 = probability_to_ry_angle(f1)
    theta2_if_0 = probability_to_ry_angle(p2_if_0)
    theta2_if_1 = probability_to_ry_angle(p2_if_1)

    qc = QuantumCircuit(2, 2)
    qc.ry(theta1, 0)

    # Control on q0=0.
    qc.x(0)
    qc.cry(theta2_if_0, 0, 1)
    qc.x(0)

    # Control on q0=1.
    qc.cry(theta2_if_1, 0, 1)

    return qc


def quantum_statevector_exact(qc) -> float:
    """
    Exact simulator probability P(q0=1,q1=1).
    Qiskit labels basis states as |q1 q0>, so both failed is still "11".
    """
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit is not installed. Run: pip install qiskit"
        ) from exc

    probabilities = Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)
    ).probabilities_dict()
    return float(probabilities.get("11", 0.0))


def quantum_shot_sampling(qc, shots: int, seed: int) -> float:
    """
    Sample the circuit with Qiskit's StatevectorSampler.

    This is still a classical full-statevector simulator followed by sampling.
    """
    try:
        from qiskit.primitives import StatevectorSampler
    except ImportError as exc:
        raise RuntimeError(
            "Qiskit with StatevectorSampler is required. "
            "Run: pip install --upgrade qiskit"
        ) from exc

    measured = qc.copy()
    measured.measure([0, 1], [0, 1])

    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    result = sampler.run([measured]).result()

    # Qiskit V2 result objects expose classical data by register name.
    pub_result = result[0]
    classical_register_name = measured.cregs[0].name
    bit_array = getattr(pub_result.data, classical_register_name)
    counts = bit_array.get_counts()

    return counts.get("11", 0) / shots


def timed_call(
    function: Callable, *args, repeats: int = 1, **kwargs
) -> tuple[object, float]:
    elapsed = []
    output = None

    for _ in range(repeats):
        start = time.perf_counter()
        output = function(*args, **kwargs)
        elapsed.append(time.perf_counter() - start)

    return output, float(np.median(elapsed))


def classical_samples_for_error(
    p: float, epsilon: float, confidence_z: float = 1.96
) -> int:
    """
    Normal-approximation sample size for absolute half-width epsilon.
    Uses known/analytic p only for planning the benchmark.
    """
    return math.ceil(
        (confidence_z**2) * p * (1.0 - p) / (epsilon**2)
    )


def theoretical_qae_query_scale(epsilon: float) -> float:
    """
    Only an order-of-growth benchmark.

    Classical Monte Carlo uses O(1/epsilon^2) samples.
    Ideal QAE uses O(1/epsilon) Grover/operator queries.

    The returned 1/epsilon is a scale, not an exact implementation count.
    Constants depend on the selected QAE variant and confidence level.
    """
    return 1.0 / epsilon


def main() -> None:
    # ---------------- User-adjustable inputs ----------------
    mission_time = 500.0
    comp1 = WeibullComponent(eta=1000.0, beta=2.0)
    comp2 = WeibullComponent(eta=1200.0, beta=1.5)
    clayton_alpha = 2.0

    mc_sample_sizes = [1_000, 10_000, 100_000, 1_000_000]
    quantum_shots = [1_000, 10_000, 100_000]
    seed = 20260724
    timing_repeats = 5
    # --------------------------------------------------------

    rows: list[dict[str, float | int | str]] = []

    exact, exact_time = timed_call(
        analytic_joint_failure_probability,
        mission_time,
        comp1,
        comp2,
        clayton_alpha,
        repeats=timing_repeats,
    )

    rows.append(
        {
            "method": "Analytic Copula formula",
            "workload": 1,
            "estimate": exact,
            "abs_error": 0.0,
            "std_error": 0.0,
            "median_time_s": exact_time,
        }
    )

    # Classical inverse-transform Monte Carlo.
    for n in mc_sample_sizes:
        (estimate, std_error), elapsed = timed_call(
            inverse_transform_mc,
            n,
            mission_time,
            comp1,
            comp2,
            clayton_alpha,
            seed,
            repeats=timing_repeats,
        )
        rows.append(
            {
                "method": "Classical inverse-transform MC",
                "workload": n,
                "estimate": estimate,
                "abs_error": abs(estimate - exact),
                "std_error": std_error,
                "median_time_s": elapsed,
            }
        )

    # Quantum circuit construction and exact statevector simulation.
    try:
        qc, build_time = timed_call(
            build_quantum_circuit,
            mission_time,
            comp1,
            comp2,
            clayton_alpha,
            repeats=timing_repeats,
        )
        q_exact, q_exact_time = timed_call(
            quantum_statevector_exact,
            qc,
            repeats=timing_repeats,
        )

        rows.append(
            {
                "method": "Quantum statevector simulator",
                "workload": 2,
                "estimate": q_exact,
                "abs_error": abs(q_exact - exact),
                "std_error": 0.0,
                "median_time_s": build_time + q_exact_time,
            }
        )

        for shots in quantum_shots:
            estimate, elapsed = timed_call(
                quantum_shot_sampling,
                qc,
                shots,
                seed,
                repeats=timing_repeats,
            )
            std_error = math.sqrt(
                max(estimate * (1.0 - estimate), 0.0) / shots
            )
            rows.append(
                {
                    "method": "Quantum circuit shot sampling (simulator)",
                    "workload": shots,
                    "estimate": estimate,
                    "abs_error": abs(estimate - exact),
                    "std_error": std_error,
                    "median_time_s": elapsed,
                }
            )

        print("\nQuantum circuit:\n")
        print(qc.draw(output="text"))

    except RuntimeError as exc:
        print(f"\nQuantum benchmark skipped: {exc}\n")

    results = pd.DataFrame(rows)
    results.to_csv("copula_mc_quantum_benchmark.csv", index=False)

    print("\nBenchmark results")
    print(results.to_string(index=False))

    print("\nComplexity comparison for target absolute error epsilon")
    complexity_rows = []
    for epsilon in [0.10, 0.05, 0.02, 0.01, 0.005]:
        complexity_rows.append(
            {
                "epsilon": epsilon,
                "classical_MC_samples_95pct": classical_samples_for_error(
                    exact, epsilon
                ),
                "ideal_QAE_query_scale_1_over_epsilon":
                    theoretical_qae_query_scale(epsilon),
            }
        )

    complexity = pd.DataFrame(complexity_rows)
    complexity.to_csv("copula_complexity_comparison.csv", index=False)
    print(complexity.to_string(index=False))

    # Runtime plot.
    try:
        import matplotlib.pyplot as plt

        plot_data = results[results["workload"] > 1].copy()
        fig, ax = plt.subplots()
        for method, group in plot_data.groupby("method"):
            ax.plot(
                group["workload"],
                group["median_time_s"],
                marker="o",
                label=method,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Samples / shots / state dimension marker")
        ax.set_ylabel("Median wall-clock time (s)")
        ax.set_title("Runtime comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig("copula_runtime_comparison.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots()
        error_data = results[
            results["method"].str.contains("MC|shot", regex=True)
        ].copy()

        for method, group in error_data.groupby("method"):
            ax.plot(
                group["workload"],
                group["abs_error"],
                marker="o",
                label=method,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Samples / shots")
        ax.set_ylabel("Absolute error")
        ax.set_title("Empirical error comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig("copula_error_comparison.png", dpi=200)
        plt.close(fig)

        print(
            "\nSaved: copula_mc_quantum_benchmark.csv, "
            "copula_complexity_comparison.csv, "
            "copula_runtime_comparison.png, "
            "copula_error_comparison.png"
        )
    except ImportError:
        print("\nmatplotlib not installed; plots were skipped.")


if __name__ == "__main__":
    main()
