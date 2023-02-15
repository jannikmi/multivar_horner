from typing import Tuple

import hypothesis as h
import numpy as np
import numpy.testing
from hypothesis import strategies as s

from multivar_horner import HornerMultivarPolynomial, MultivarPolynomial
from multivar_horner.global_settings import FLOAT_DTYPE
from multivar_horner.helper_fcts import rectify_construction_parameters, rectify_query_point
from tests.helpers import all_possible_exponents

REQUIRED_REL_PRECISION = 1e-5
FLOAT_MAX_VAL = 1e50  # for high values numerical instabilities might occur!
FLOAT_MIN_VAL = -FLOAT_MAX_VAL
MAX_DIM = 1
MAX_DEG = 3

dimensions = s.integers(min_value=1, max_value=MAX_DIM)
poly_degrees = s.integers(min_value=0, max_value=MAX_DEG)
query_point_sampling = s.floats(
    min_value=FLOAT_MIN_VAL,
    max_value=FLOAT_MAX_VAL,
    allow_infinity=False,
    allow_nan=False,
)
coefficient_sampling = s.floats(min_value=FLOAT_MIN_VAL, max_value=FLOAT_MAX_VAL, allow_nan=False, allow_infinity=False)


@s.composite
def poly_param_sampling(draw):
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

    coefficients = draw(s.lists(coefficient_sampling, min_size=nr_monomials, max_size=nr_monomials))

    coefficients, exponent_matrix = rectify_construction_parameters(coefficients, exponent_matrix)
    return coefficients, exponent_matrix


@s.composite
def poly_sampling(draw) -> Tuple[MultivarPolynomial, HornerMultivarPolynomial]:
    coefficients, exponents = draw(poly_param_sampling())
    poly = MultivarPolynomial(coefficients, exponents, rectify_input=False, compute_representation=False, verbose=True)
    poly_h = HornerMultivarPolynomial(
        coefficients, exponents, rectify_input=False, compute_representation=False, verbose=True
    )
    return poly, poly_h


@s.composite
def example_case_sampling(draw):
    poly, poly_h = draw(poly_sampling())
    nr_dims = poly.dim
    x = draw(s.lists(query_point_sampling, min_size=nr_dims, max_size=nr_dims))
    x = rectify_query_point(x)
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
    np.testing.assert_allclose(res, res_h, rtol=REQUIRED_REL_PRECISION, err_msg=f"{x=}\n{coefficients=}\n{exponents=}")


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
    np.testing.assert_allclose(res, res_h, rtol=REQUIRED_REL_PRECISION, err_msg=f"{x=}\n{coefficients=}\n{exponents=}")


# TODO complex tests
