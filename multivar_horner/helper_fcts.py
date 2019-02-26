import itertools

import numpy as np


# https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
# a generator yielding all prime numbers in ascending order
def erat2():
    D = {}
    yield 2
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        p = D.pop(q, None)
        if p is None:
            D[q * q] = q
            yield q
        else:
            x = p + q
            while x in D or not (x & 1):
                x += p
            D[x] = p


def get_prime_array(length):
    return np.array(list(itertools.islice(erat2(), length)), dtype=np.uint32)


def get_goedel_id_of(prime_idx, exponent, prime_array):
    '''
    NOTE: factor IDs of monomials (product of scala IDs) potentially grow very large,
    especially with high dimensionality -> overflow?!
    python automatically uses the reqired long data type for ints,  no special attention required?!
    TODO use different method different from goedel id approach

    :param prime_idx:
    :param exponent:
    :param prime_array: the unique ID of any scalar monomial x_i^n
    :return:
    '''
    return int(prime_array[prime_idx] ** exponent)


def rectify(coefficients, exponents):
    """
    convert the input into numpy arrays valid as input to MultivarPolynomial
    raise an error if the given input is incompatible
    :param coefficients: possibly a python list of coefficients to be converted
    :param exponents: possibly a nested python list of exponents to be converted
    :return: the input converted into appropriate numpy data types
    """
    rectified_coefficients = np.atleast_1d(np.array(coefficients, dtype=np.float64)).reshape(-1, 1)

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


def validate(coefficients, exponents):
    """
    raises an error when the given input is not valid
    :param coefficients: a numpy array column vector of doubles representing the coefficients of the monomials
    :param exponents: a numpy array matrix of unsigned integers representing the exponents of the monomials
    :return:
    """
    # coefficients must be given as a column vector (2D)
    assert coefficients.shape[1] == 1 and len(coefficients.shape) == 2
    # exponents must be 2D (matrix) = a list of exponent vectors
    assert len(exponents.shape) == 2
    # there must be at least one entry
    assert coefficients.shape[0] > 0
    # exponents must not be negative
    assert not np.any(exponents < 0)
    # there must not be duplicate exponent vectors
    assert exponents.shape == np.unique(exponents, axis=0).shape
    # there must not be any coefficients with 0.0
    assert not np.any(coefficients == 0.0)
    # must have the same amount of entries
    assert coefficients.shape[0] == exponents.shape[0]
