# Reproduce the finite-shot IQAE benchmark

This entry point runs Qiskit Algorithms 0.4.0 IterativeAmplitudeEstimation for
P(T1 <= t, T2 <= t), reusing the existing two-qubit RY/CRY preparation.
The unused classical register is removed; both objective qubits must be 1.
The default phase oracle and the good-state predicate therefore both mark 11.
The original benchmark and its theoretical complexity helper remain available.

## Run

From the repository root:

```bash
cd copula-quantum-reliability/copula-quantum-reliability
python -m pip install -r requirements.txt
python -m pytest tests/test_copula_iae.py -q
python src/benchmark_copula_iae.py --epsilon 0.02 --alpha 0.05 --shots 256 --seed 17
```

The CLI writes JSON to standard output. The alpha option is the significance
level (0.05 means 95% confidence); the Clayton parameter is fixed separately
at 2.0 in default_model(). The shots option is the number of shots per IQAE
round, not the total budget. All executions use a classical simulator.

## Validated environment and regression checks

Executed on 2026-11-16 on Windows with Python 3.12.10, Qiskit 2.4.1,
Qiskit Algorithms 0.4.0, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.3,
matplotlib 3.11.2 and pytest 8.4.2.
The dependency installation command found all requirements already satisfied.
For the same direct dependency versions, install:

```bash
python -m pip install qiskit==2.4.1 qiskit-algorithms==0.4.0 numpy==2.4.4 scipy==1.17.1 pandas==3.0.3 matplotlib==3.11.2 pytest==8.4.2
```

These are the observed direct versions, not a complete transitive lockfile.
The broad lower bounds in requirements.txt do not guarantee future-version
compatibility. Only the environment above was executed for this result.

All 6 regression cases passed (38.13 seconds in the recorded run).
They cover all four joint probabilities and bit ordering, inverse preparation,
independence and deterministic endpoints, the sin^2((2k+1) theta) amplification
identity, fixed-seed reproducibility, nonzero Grover powers and query accounting,
and rejection of measured state preparation. A NumPy Generator advances the
random stream between adaptive IQAE rounds.

The run emitted 238 upstream deprecation warnings relating to GroverOperator
and MCX APIs. They did not cause failures; migration to Qiskit 3 is untested.

## Recorded experiment

Model: t=500, component 1 (eta=1000, beta=2), component 2
(eta=1200, beta=1.5), Clayton parameter=2.
Precision target=0.02, confidence=95%, seed=17, IQAE shots per round=256.
The unrounded output is in [iqae_run_2026-11-16.json](results/iqae_run_2026-11-16.json).

| Method | Estimate | 95% interval | Absolute error | Measured seconds |
| --- | ---: | --- | ---: | ---: |
| Analytic | 0.1634756261 | N/A | 0 | 0.0000636000 |
| Exact statevector | 0.1634756261 | N/A | 5.55e-17 | 0.0026811999 |
| Classical inverse-transform MC | 0.1595836947 | [0.1491240259, 0.1704762749] | 0.0038919314 | 0.0020017000 |
| Ordinary quantum sampling | 0.1530789245 | [0.1428009869, 0.1637981458] | 0.0103967016 | 0.0235729000 |
| IQAE | 0.1629328459 | [0.1599734353, 0.1658922566] | 0.0005427802 | 0.0555601000 |

MC and ordinary sampling each use 4,612 samples/shots, chosen by the Hoeffding
bound without using the analytic answer. Their reported intervals are
fixed-sample Clopper-Pearson intervals. IQAE uses the algorithm's beta-based
adaptive interval; these are different statistical procedures.

IQAE executed powers [0, 1, 10] with 256 shots each:

- Total shots: 768.
- Q queries: sum(shots_j * k_j) = 2,816, matching num_oracle_queries.
- A plus A-dagger calls: sum(shots_j * (2*k_j+1)) = 6,400.
- Final interval half-width: approximately 0.00295941.

The initial sentinel in result.powers is excluded; the executed k=0 round is
included. Q contains one A and one A-dagger, and each shot also needs an initial A.
Q queries, state-preparation calls, shots and classical samples are distinct costs.

Circuit construction took 0.0003290001 seconds and is reported separately.
MC and ordinary-sampling timers exclude the subsequent binomial interval
calculation; the IQAE timer includes sampler/problem setup and adaptive interval
updates. These are single-run timings excluding imports, not equal-scope runtime
benchmarks or stable performance estimates.

This is a correctness experiment on a classical StatevectorSampler. The
preparation angles already use the analytic probability. No hardware experiment,
multi-seed coverage study, or asymptotic scaling experiment was performed.
The smaller IQAE shot count alone does not establish a speedup: its 6,400
A/A-dagger calls exceed the ordinary sampler's 4,612 A calls in this run.
The classical method here samples lifetimes, not repairable-system sequential
Monte Carlo trajectories.
