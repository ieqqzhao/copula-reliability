# Copula dependence and controlled quantum gates

## 1. Failure-state convention

For component `i`, define

```math
X_i(t)=\mathbf{1}\{T_i\le t\},
```

where `0` means operating and `1` means failed by mission time `t`.

## 2. Joint probabilities from a copula

Let

```math
u=F_1(t),\qquad v=F_2(t).
```

By Sklar's theorem,

```math
P(X_1=1,X_2=1)=C(u,v;\alpha)=P_{11}.
```

The remaining probabilities are

```math
P_{10}=u-P_{11},
```

```math
P_{01}=v-P_{11},
```

```math
P_{00}=1-u-v+P_{11}.
```

## 3. Conditional-probability factorization

Factor the joint distribution as

```math
P(X_1,X_2)=P(X_1)P(X_2\mid X_1).
```

The first rotation angle is

```math
\theta_1=2\arcsin\sqrt{P(X_1=1)}.
```

The conditional probabilities for component 2 are

```math
p_{2\mid0}=\frac{P_{01}}{P_{00}+P_{01}},
```

```math
p_{2\mid1}=\frac{P_{11}}{P_{10}+P_{11}}.
```

Their controlled-rotation angles are

```math
\theta_{2\mid0}=2\arcsin\sqrt{p_{2\mid0}},
```

```math
\theta_{2\mid1}=2\arcsin\sqrt{p_{2\mid1}}.
```

The controlled gates do not represent the copula directly. They implement the conditional probability structure implied by the copula.

## 4. Circuit structure

```text
q0: |0> -- RY(theta1) -- o -------- @ --------
                         |          |
q1: |0> --------------- RY(θ2|0) -- RY(θ2|1)
```

Qiskit's `cry` gate is controlled on `|1>`. Control on `|0>` is implemented by applying `X`, then `CRY`, then `X` to the control qubit.

## 5. Extension to inverse-transform sampling

A more complete lifetime-generation model first prepares dependent uniforms

```math
(U_1,U_2)\sim C,
```

then computes

```math
T_i=F_i^{-1}(U_i).
```

For Weibull marginals,

```math
F_i^{-1}(u)=\eta_i[-\ln(1-u)]^{1/\beta_i}.
```

This route requires multi-qubit registers, correlated state preparation, reversible arithmetic or threshold comparators, and uncomputation. It is more faithful to inverse-transform sampling but substantially more expensive than direct binary-state encoding at a fixed mission time.
