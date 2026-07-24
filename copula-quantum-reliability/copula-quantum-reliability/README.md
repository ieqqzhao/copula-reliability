# Copula-Based Dependent Failures in Quantum Reliability Analysis

This repository demonstrates how to model two statistically dependent component failures using Weibull marginal lifetime distributions, a Clayton copula, and controlled quantum rotation gates in Qiskit.

## Contents

- `src/copula_dependent_failure_qiskit.py`: prepares a two-qubit joint failure state using one `RY` gate and conditional `CRY` gates.
- `src/benchmark_copula_mc_quantum.py`: compares the analytic copula result, inverse-transform Monte Carlo, exact statevector simulation, and shot-based quantum-circuit sampling.
- `docs/copula_dependency_quantum.md`: mathematical derivation and circuit interpretation.
- `docs/method_comparison.md`: fair runtime and complexity comparison guidance.

## Model

For component lifetimes `T1` and `T2`, the Weibull marginal CDFs are

```math
F_i(t)=1-\exp\left[-\left(\frac{t}{\eta_i}\right)^{\beta_i}\right].
```

The Clayton copula is

```math
C(u,v;\alpha)=\left(u^{-\alpha}+v^{-\alpha}-1\right)^{-1/\alpha}.
```

At mission time `t`, the joint failure probability is

```math
P(T_1\le t,T_2\le t)=C(F_1(t),F_2(t);\alpha).
```

The resulting binary joint distribution is encoded as

```math
|\psi\rangle=\sum_{x_1,x_2\in\{0,1\}}\sqrt{P(x_1,x_2)}|x_1x_2\rangle.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the dependent-failure circuit

```bash
python src/copula_dependent_failure_qiskit.py
```

## Run the benchmark

```bash
python src/benchmark_copula_mc_quantum.py
```

The benchmark generates CSV result tables and runtime/error plots in the current directory.

## Interpretation of timing results

Statevector and shot-based Qiskit executions in this project run on a classical simulator. Their wall-clock times do not demonstrate quantum speedup. A fair methodological comparison distinguishes:

- analytic runtime;
- classical inverse-transform Monte Carlo runtime and error;
- quantum simulator runtime;
- ideal quantum amplitude estimation query complexity.

Classical Monte Carlo has error order `O(N^{-1/2})`, requiring `O(1/epsilon^2)` samples for target error `epsilon`. Ideal quantum amplitude estimation has query complexity `O(1/epsilon)`, but this potential advantage requires an actual amplitude-estimation implementation and must include state-preparation and oracle costs.
