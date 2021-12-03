# -*- coding:utf-8 -*-

import numpy as np

from .global_settings import FLOAT_DTYPE, UINT_DTYPE


def rectify_coefficients(coefficients):
    rectified_coefficients = np.atleast_2d(np.asarray(coefficients, dtype=FLOAT_DTYPE)).reshape(-1, 1)
    return rectified_coefficients


def rectify_construction_parameters(coefficients, exponents):
    """
    convert the input into numpy arrays valid as input of AbstractPolynomial
    :param coefficients: possibly a python list of coefficients to be converted
    :param exponents: possibly a nested python list of exponents to be converted
     ATTENTION: when converting to unsigned integer, negative integers become large!
    :return: the input converted into appropriate numpy data types
    """
    rectified_coefficients = rectify_coefficients(coefficients)
    exponents = np.atleast_2d(exponents)
    if np.any(exponents < 0):
        raise ValueError(
            "negative exponents are not allowed!"
            f"the conversion to {UINT_DTYPE} would turn negative values into very large values"
        )
    rectified_exponents = exponents.astype(UINT_DTYPE)
    return rectified_coefficients, rectified_exponents


def rectify_query_point(x):
    """
    convert the input into numpy ndarray valid as input to AbstractPolynomial.eval()
    :param x: possibly a python list to be converted
    :return: the input converted into appropriate numpy data type
    """
    rectified_x = np.atleast_1d(np.asarray(x, dtype=FLOAT_DTYPE))
    return rectified_x


def validate_coefficients(coefficients) -> None:
    if not isinstance(coefficients, np.ndarray):
        raise TypeError("coefficients must be given as numpy.ndarray")
    if coefficients.dtype.type is not FLOAT_DTYPE:
        raise TypeError(f"coefficients must have dtype {FLOAT_DTYPE} but have dtype {coefficients.dtype.type}")
    if len(coefficients.shape) != 2 or coefficients.shape[1] != 1:
        raise ValueError("coefficients must be given as a [M, 1] ndarray")
    if coefficients.shape[0] == 0:
        raise ValueError("there must be at least one coefficient")

    # NOTE: "0 coefficients" are allowed because coefficients should be changeable!


def validate_construction_parameters(coefficients: np.ndarray, exponents: np.ndarray) -> None:
    validate_coefficients(coefficients)

    if not isinstance(exponents, np.ndarray):
        raise TypeError("exponents must be given as numpy.ndarray")
    if exponents.dtype.type is not UINT_DTYPE:
        # NOTE: this also ensures non negative exponents
        # value check is being done in rectification fct
        raise TypeError(f"exponents must have dtype {UINT_DTYPE}")
    if len(exponents.shape) != 2:
        raise ValueError("exponents must be given as a [M, N] ndarray (a list of exponent vectors)")
    if np.any(exponents < 0):
        raise ValueError("exponents must not be negative")
    if exponents.shape != np.unique(exponents, axis=0).shape:
        raise ValueError("there must not be duplicate exponent vectors")
    if coefficients.shape[0] != exponents.shape[0]:
        raise ValueError("there must be as many exponent vectors as coefficients")


def validate_query_point(x: np.ndarray, dim: int) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError("the query point x  must be given as numpy.ndarray")
    if x.dtype.type is not FLOAT_DTYPE:
        raise TypeError(f"the query point x must have dtype {FLOAT_DTYPE}")
    if len(x.shape) != 1:
        raise ValueError("the query point x must be given as a ndarray of shape [N]")
    if x.shape[0] != dim:
        raise ValueError(f"the query point x does not have the required dimensionality {dim}")
