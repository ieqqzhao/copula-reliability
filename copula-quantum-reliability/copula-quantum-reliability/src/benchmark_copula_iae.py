"""Compare P(11) estimators at a reproducible finite-shot setting."""
import argparse
import json
import math
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from scipy.stats import binomtest

from benchmark_copula_mc_quantum import (
    WeibullComponent,
    analytic_joint_failure_probability,
    build_quantum_circuit,
    inverse_transform_mc,
    quantum_shot_sampling,
    quantum_statevector_exact,
)


def make_problem(qc):
    """Reuse the existing gate sequence without its unused classical bits."""
    if qc.num_qubits != 2 or qc.parameters:
        raise ValueError("Expected a bound two-qubit circuit.")
    a = QuantumCircuit(2, name="A")
    a.global_phase = qc.global_phase
    for item in qc.data:
        if item.operation.name not in {"ry", "cry", "x", "barrier"}:
            raise ValueError("Expected an unmeasured RY/CRY preparation.")
        a.append(item.operation, [qc.find_bit(q).index for q in item.qubits])
    return EstimationProblem(state_preparation=a, objective_qubits=[0, 1])


def run_iae(qc, epsilon=0.02, alpha=0.05, shots=256, seed=17):
    if shots < 1:
        raise ValueError("Shots per round must be positive.")
    start = perf_counter()
    # Pass a Generator so repeated rounds advance one reproducible RNG stream.
    sampler = StatevectorSampler(default_shots=shots, seed=np.random.default_rng(seed))
    algorithm = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, confint_method="beta", sampler=sampler
    )
    result = algorithm.estimate(make_problem(qc))
    seconds = perf_counter() - start
    powers = [int(k) for k in result.powers[1:]]  # skip initial sentinel
    total_shots = shots * len(powers)
    q_queries = shots * sum(powers)
    if q_queries != result.num_oracle_queries:
        raise RuntimeError("Unexpected IQAE query count.")
    return {
        "estimate": float(result.estimation),
        "ci": list(map(float, result.confidence_interval)),
        "shots": total_shots,
        "q_queries": q_queries,
        "a_and_adjoint_queries": total_shots + 2 * q_queries,
        "powers": powers,
        "seconds": seconds,
    }


def default_model():
    return 500.0, WeibullComponent(1000, 2), WeibullComponent(1200, 1.5), 2.0


def benchmark(epsilon, alpha, shots, seed):
    model = default_model()
    start = perf_counter()
    truth = analytic_joint_failure_probability(*model)
    analytic = {"estimate": truth, "seconds": perf_counter() - start}
    start = perf_counter()
    qc = build_quantum_circuit(*model)
    prep_seconds = perf_counter() - start
    start = perf_counter()
    sv = quantum_statevector_exact(qc)
    exact_sv = {"estimate": sv, "seconds": perf_counter() - start}
    if not math.isclose(sv, truth, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Encoded probability differs from analytic reference.")
    n = math.ceil(math.log(2 / alpha) / (2 * epsilon**2))
    rng = np.random.SeedSequence(seed).spawn(2)
    mc_seed, shot_seed = (int(s.generate_state(1)[0]) for s in rng)

    def sampled(estimate, elapsed, **counts):
        ci = binomtest(round(estimate * n), n).proportion_ci(
            confidence_level=1 - alpha, method="exact"
        )
        return {
            "estimate": estimate, "seconds": elapsed, "ci": list(ci), **counts
        }

    start = perf_counter()
    mc, _ = inverse_transform_mc(n, *model, mc_seed)
    classical = sampled(mc, perf_counter() - start, samples=n)
    start = perf_counter()
    estimate = quantum_shot_sampling(qc, n, shot_seed)
    ordinary = sampled(
        estimate, perf_counter() - start, shots=n, q_queries=0,
        a_and_adjoint_queries=n
    )
    methods = {
        "analytic": analytic,
        "statevector": exact_sv,
        "classical_mc": classical,
        "ordinary_quantum_sampling": ordinary,
        "iqae": run_iae(qc, epsilon, alpha, shots, seed),
    }
    for row in methods.values():
        row["abs_error"] = abs(row["estimate"] - truth)
    return {
        "backend": "classical StatevectorSampler simulation",
        "epsilon_target": epsilon,
        "confidence": 1 - alpha,
        "seed": seed,
        "iqae_shots_per_round": shots,
        "circuit_build_seconds": prep_seconds,
        "methods": methods,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not (0 < args.epsilon <= 0.5 and 0 < args.alpha < 1
            and args.shots > 0 and args.seed >= 0):
        parser.error("Require 0 < epsilon <= 0.5, 0 < alpha < 1, positive shots and nonnegative seed.")
    print(json.dumps(benchmark(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
