import math
import sys
from pathlib import Path

import pytest
from qiskit.quantum_info import Statevector
from qiskit_algorithms import IterativeAmplitudeEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_copula_iae import default_model, make_problem, run_iae
from benchmark_copula_mc_quantum import (
    build_quantum_circuit,
    joint_state_probabilities,
)


@pytest.mark.parametrize("time,clayton", [(0, 2), (500, 0), (500, 2), (1e6, 2)])
def test_joint_distribution_and_grover_rotation(time, clayton):
    _, c1, c2, _ = default_model()
    model = time, c1, c2, clayton
    p = joint_state_probabilities(*model)
    problem = make_problem(build_quantum_circuit(*model))
    a = problem.state_preparation
    state = Statevector.from_instruction(a)
    assert list(state.probabilities()) == pytest.approx(
        [p[s] for s in ("00", "10", "01", "11")], abs=1e-12
    )
    assert list(state.evolve(a.inverse()).data) == pytest.approx([1, 0, 0, 0])
    assert [s for s in p if problem.is_good_state(s)] == ["11"]
    algorithm = IterativeAmplitudeEstimation(0.02, 0.05)
    theta = math.asin(math.sqrt(p["11"]))
    for k in (1, 2, 5):
        circuit = algorithm.construct_circuit(problem, k)
        assert Statevector.from_instruction(circuit).probabilities()[3] == pytest.approx(
            math.sin((2 * k + 1) * theta)**2, abs=1e-10
        )


def test_actual_iae_amplifies_and_is_reproducible():
    model = default_model()
    qc = build_quantum_circuit(*model)
    first = run_iae(qc)
    second = run_iae(qc)
    assert {k: v for k, v in first.items() if k != "seconds"} == {
        k: v for k, v in second.items() if k != "seconds"
    }
    lo, hi = first["ci"]
    assert 0 <= lo <= first["estimate"] <= hi <= 1
    assert hi - lo <= 0.04 + 1e-12
    assert first["q_queries"] == 256 * sum(first["powers"]) > 0
    assert first["shots"] == 256 * len(first["powers"])
    assert first["a_and_adjoint_queries"] == first["shots"] + 2 * first["q_queries"]
    assert abs(first["estimate"] - joint_state_probabilities(*model)["11"]) < 0.05


def test_measured_circuit_cannot_be_used_as_preparation():
    qc = build_quantum_circuit(*default_model())
    qc.measure([0, 1], [0, 1])
    with pytest.raises(ValueError):
        make_problem(qc)
