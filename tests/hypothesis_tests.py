from typing import Tuple

import hypothesis as h
import numpy as np
import numpy.testing
from hypothesis import strategies as s

from multivar_horner import HornerMultivarPolynomial, MultivarPolynomial
from multivar_horner.global_settings import COMPLEX_DTYPE, FLOAT_DTYPE
from multivar_horner.helper_fcts import rectify_construction_parameters, rectify_query_point
from tests.helpers import all_possible_exponents

REQUIRED_REL_PRECISION = 1e-4
REQUIRED_ABS_PRECISION = 5

# for small and very large values numerical instabilities might occur!
# still both signs accepted!
FLOAT_MAX_VAL = 1e50
FLOAT_MIN_VAL = 1e-50
MAX_DIM = 3
MAX_DEG = 3

dimensions = s.integers(min_value=1, max_value=MAX_DIM)
poly_degrees = s.integers(min_value=0, max_value=MAX_DEG)
unsigned_float_sampling = s.floats(
    min_value=FLOAT_MIN_VAL, max_value=FLOAT_MAX_VAL, allow_infinity=False, allow_nan=False
)


@s.composite
def float_sampling(draw) -> float:
    val = draw(unsigned_float_sampling)
    sign = draw(s.booleans())
    if sign:
        val = -val
    return val


@s.composite
def query_point_sampling_complex(draw) -> np.complex:
    real_part = draw(float_sampling())
    imaginary_part = draw(float_sampling())
    query_point = np.complex(real_part, imaginary_part)
    return query_point


@s.composite
def poly_param_sampling(draw) -> Tuple[np.ndarray, np.ndarray]:
    # NOTE: obtain internal events via conversion of incoming events (more realistic)
    # rather than sampling internal event types directly
    dimension = draw(dimensions)
    poly_degree = draw(poly_degrees)
    possible_exponents = all_possible_exponents(dimension, poly_degree)
    np.random.shuffle(possible_exponents)
    nr_monomials_max = len(possible_exponents)
    # there must be at least 1 entry ('active' monomial)
    nr_monomials = draw(s.integers(min_value=1, max_value=nr_monomials_max))
    exponent_matrix = possible_exponents[:nr_monomials]

    coefficients = draw(s.lists(float_sampling(), min_size=nr_monomials, max_size=nr_monomials))

    coefficients, exponent_matrix = rectify_construction_parameters(coefficients, exponent_matrix)
    return coefficients, exponent_matrix


@s.composite
def poly_sampling(draw) -> Tuple[MultivarPolynomial, HornerMultivarPolynomial]:
    coefficients, exponents = draw(poly_param_sampling())
    poly = MultivarPolynomial(coefficients, exponents, rectify_input=False, compute_representation=True, verbose=True)
    poly_h = HornerMultivarPolynomial(
        coefficients, exponents, rectify_input=False, compute_representation=True, verbose=True
    )
    return poly, poly_h


@s.composite
def example_case_sampling(draw) -> Tuple[MultivarPolynomial, HornerMultivarPolynomial, np.ndarray]:
    poly, poly_h = draw(poly_sampling())
    nr_dims = poly.dim
    x = draw(s.lists(float_sampling(), min_size=nr_dims, max_size=nr_dims))
    x = rectify_query_point(x)
    return poly, poly_h, x


@s.composite
def example_case_sampling_complex(draw) -> Tuple[MultivarPolynomial, HornerMultivarPolynomial, np.ndarray]:
    poly, poly_h = draw(poly_sampling())
    nr_dims = poly.dim
    x = draw(s.lists(query_point_sampling_complex(), min_size=nr_dims, max_size=nr_dims))
    x = rectify_query_point(x, dtype=COMPLEX_DTYPE)
    return poly, poly_h, x


@h.settings(
    deadline=None,  # varying execution times
)
@h.given(params=example_case_sampling())
def test_c_eval(params):
    poly, poly_h, x = params

    res = poly(x)
    res_h = poly_h._eval_c(x)
    coefficients = poly.coefficients
    exponents = poly.exponents
    all_close(res, res_h, coefficients, exponents, x)


@h.settings(
    deadline=None,  # varying execution times
)
@h.given(params=example_case_sampling())
def test_recipe_eval(params):
    poly, poly_h, x = params

    res = poly(x)
    res_h = poly_h._eval_recipe(x, dtype=FLOAT_DTYPE)
    coefficients = poly.coefficients
    exponents = poly.exponents
    all_close(res, res_h, coefficients, exponents, x)


@h.settings(
    deadline=None,  # varying execution times
)
@h.given(params=example_case_sampling_complex())
def test_complex_eval(params):
    poly, poly_h, x = params

    res = poly.eval_complex(x)
    res_h = poly_h.eval_complex(x)
    coefficients = poly.coefficients
    exponents = poly.exponents
    all_close(res, res_h, coefficients, exponents, x)


def all_close(res1, res2, coefficients, exponents, x):
    err_msg = f"{x=}\n{coefficients=}\n{exponents=}"
    if abs(res1) < 1e-5:
        np.testing.assert_almost_equal(res1, res2, decimal=REQUIRED_ABS_PRECISION, err_msg=err_msg)
    else:
        np.testing.assert_allclose(res1, res2, rtol=REQUIRED_REL_PRECISION, err_msg=err_msg)
