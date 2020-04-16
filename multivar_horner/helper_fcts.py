# -*- coding:utf-8 -*-

import numpy as np


def rectify_coefficients(coefficients):
    rectified_coefficients = np.atleast_1d(np.array(coefficients, dtype=np.float64)).reshape(-1, 1)
    return rectified_coefficients


def rectify(coefficients, exponents):
    """
    convert the input into numpy arrays valid as input to MultivarPolynomial
    raise an error if the given input is incompatible
    :param coefficients: possibly a python list of coefficients to be converted
    :param exponents: possibly a nested python list of exponents to be converted
    :return: the input converted into appropriate numpy data types
    """
    rectified_coefficients = rectify_coefficients(coefficients)

    rectified_exponents = np.atleast_2d(np.array(exponents, dtype=np.int))
    # exponents must not be negative!
    # ATTENTION: when converting to unsigned integer, negative integers become large!
    assert not np.any(rectified_exponents < 0)
    rectified_exponents = rectified_exponents.astype(np.uint32)

    # ignore the entries with 0.0 coefficients
    if np.any(rectified_coefficients == 0.0):
        non_zero_coeff_rows = np.where(rectified_coefficients != 0.0)[0]
        rectified_coefficients = rectified_coefficients[non_zero_coeff_rows, :]
        rectified_exponents = rectified_exponents[non_zero_coeff_rows, :]

    return rectified_coefficients, rectified_exponents


def validate_coefficients(coefficients) -> None:

    assert type(coefficients) is np.ndarray, 'coefficients must be given as numpy ndarray'
    assert len(coefficients.shape) == 2 and coefficients.shape[1] == 1, \
        'coefficients must be given as a [n, 1] ndarray'

    assert coefficients.shape[0] > 0, 'there must be at least one coefficient'

    # assert not np.any(coefficients == 0.0), 'there must not be any coefficients with 0.0'
    # allowed since coefficients should be changeable


def validate(coefficients, exponents) -> None:
    """
    raise an error when the given input parameters of a polynomial are not valid
    """

    validate_coefficients(coefficients)

    assert type(exponents) is np.ndarray, 'exponents must be given as numpy ndarray'
    assert len(exponents.shape) == 2, 'exponents must be 2 dimensional (a list of exponent vectors)'
    assert not np.any(exponents < 0), 'exponents must not be negative'

    if exponents.shape != np.unique(exponents, axis=0).shape:
        raise ValueError('there must not be duplicate exponent vectors')

    assert coefficients.shape[0] == exponents.shape[0], 'there must be as many exponent vectors as coefficients'
