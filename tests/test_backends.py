import numpy as np
import pytest

from multivar_horner import HornerMultivarPolynomial, helpers_fcts_numba
from tests.helpers import proto_test_case


@pytest.mark.parametrize("backend", ["c", "recipe"])
def test_backend_selection_and_values(backend, monkeypatch, tmp_path):
    from multivar_horner.classes import horner_poly

    monkeypatch.setattr(horner_poly, "PATH2CACHE", tmp_path)
    if backend == "recipe":
        monkeypatch.setenv("PATH", str(tmp_path))
    poly = HornerMultivarPolynomial(
        [5.0, 1.0, 2.0, 3.0],
        [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]],
        rectify_input=True,
    )
    assert poly.use_c_eval == (backend == "c")
    for x in (np.array([-2.0, 3.0, 1.0]), np.array([1 + 2j, -3j, 2 - 1j])):
        expected = 5 + x[0] ** 3 * x[1] + 2 * x[0] ** 2 * x[2] + 3 * x[0] * x[1] * x[2]
        actual = poly.eval_complex(x) if np.iscomplexobj(x) else poly(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    if helpers_fcts_numba.using_numba:
        assert helpers_fcts_numba.eval_recipe.nopython_signatures


def test_reference_helper_propagates_failures():
    with pytest.raises(AssertionError):
        proto_test_case([(([], [], [0.0]), 1.0)], lambda _: 2.0)
