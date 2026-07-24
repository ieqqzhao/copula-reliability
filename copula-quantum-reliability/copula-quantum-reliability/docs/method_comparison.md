# Comparison of analytic, Monte Carlo, and quantum approaches

## Target quantity

All methods estimate

```math
P_{11}(t)=P(T_1\le t,T_2\le t).
```

## Analytic solution

For known Weibull marginals and an explicit bivariate copula,

```math
P_{11}(t)=C(F_1(t),F_2(t);\alpha).
```

For the two-component example, this calculation is effectively constant time and should be used as the benchmark truth.

## Inverse-transform Monte Carlo

Generate dependent uniforms from the copula, calculate

```math
T_i=F_i^{-1}(U_i),
```

and average the indicator

```math
\mathbf{1}\{T_1\le t,T_2\le t\}.
```

Its standard error is approximately

```math
\sqrt{\frac{P(1-P)}{N}},
```

so the error is `O(N^{-1/2})` and the sample requirement for absolute error `epsilon` is `O(epsilon^{-2})`.

## Quantum circuit with ordinary measurement

The controlled-rotation circuit prepares the required joint distribution. Repeated ordinary measurements still have sampling error `O(S^{-1/2})`, where `S` is the number of shots. Ordinary quantum sampling alone therefore does not provide the amplitude-estimation quadratic advantage.

## Quantum amplitude estimation

Ideal amplitude estimation estimates a target-state probability using `O(epsilon^{-1})` applications of the amplitude/Grover operator. This is a query-complexity improvement over classical Monte Carlo, but a complete comparison must include:

- state-preparation cost;
- copula dependence-loading cost;
- reliability oracle cost;
- circuit depth and two-qubit gate count;
- error-mitigation and measurement overhead;
- hardware execution and queue time when relevant.

## Reporting recommendations

Report both empirical and algorithmic quantities:

1. analytic wall-clock time;
2. Monte Carlo wall-clock time, MAE, RMSE, and confidence interval;
3. quantum simulator wall-clock time, clearly labelled as classical simulation;
4. qubits, circuit depth, two-qubit gates, shots, and oracle calls;
5. theoretical `O(epsilon^{-2})` versus `O(epsilon^{-1})` scaling;
6. repeated experiments with independent random seeds.

For the current two-component example, the expected wall-clock ordering is generally analytic first, vectorized Monte Carlo second, and Qiskit simulation last. The example validates the encoding mechanism rather than demonstrating practical quantum speedup.
