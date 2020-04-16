import random

import numpy as np

from multivar_horner.global_settings import UINT_DTYPE

TEST_RESULTS_PICKLE = 'test_results.pickle'


def get_rnd_poly_properties(all_exponents, max_abs_coeff=1.0, integer_coeffs=False):
    # every exponent can take the values in the range [0; max_degree]
    max_nr_exponent_vects = all_exponents.shape[0]

    # decide how many entries the polynomial should have
    # desired for meaningful speed test results:z
    # every possible polynomial should appear with equal probability
    # <=> every possible monomial should appear with equal probability
    row_idxs = np.random.randint(0, 2, size=max_nr_exponent_vects, dtype=np.bool)  # 0 or 1

    if np.sum(row_idxs) == 0:  # there must be at least 1 entry ('active' monomial)
        row_idxs[random.randint(0, max_nr_exponent_vects - 1)] = 1

    exponents = all_exponents[row_idxs, :]
    # NOTE: the polynomial might not actually have the maximal max_degree (<- monomial not present)!

    nr_monomials = exponents.shape[0]
    coefficients = (np.random.rand(nr_monomials, 1) - 0.5) * (2 * max_abs_coeff)  # [ -max_abs_coeff; max_abs_coeff]
    if integer_coeffs:
        coefficients = np.round(coefficients)

    return coefficients, exponents


def all_possible_exponents(dim, deg):
    def cntr2exp_vect(cntr):
        exp_vect = np.empty((dim), dtype=UINT_DTYPE)
        for d in range(dim - 1, -1, -1):
            divisor = (deg + 1) ** d
            # cntr = quotient*divisor + remainder
            quotient, remainder = divmod(cntr, divisor)
            exp_vect[d] = quotient
            cntr = remainder
        return exp_vect

    max_nr_exponent_vects = (deg + 1) ** dim
    all_possible = np.empty((max_nr_exponent_vects, dim), dtype=UINT_DTYPE)
    for i in range(max_nr_exponent_vects):
        # print(i, cntr2exp_vect(i))
        all_possible[i] = cntr2exp_vect(i)

    # there must not be duplicate exponent vectors
    assert all_possible.shape == np.unique(all_possible, axis=0).shape

    return all_possible


def rnd_settings_list(length, dim, degree, *args, **kwargs):
    all_exponent_vect = all_possible_exponents(dim, degree)
    settings_list = [get_rnd_poly_properties(all_exponent_vect, *args, **kwargs) for i in range(length)]

    # # random settings should have approx. half the amount of maximal entries on average
    # num_monomial_entries = 0
    # for settings in settings_list:
    #     num_monomial_entries += settings[0].shape[0]
    #
    # avg_monomial_entries = num_monomial_entries / length
    # max_monomial_entries = int((max_degree + 1) ** dim)
    # print(avg_monomial_entries, max_monomial_entries)
    return settings_list


def rnd_input_list(length, dim, max_abs_val):
    return [(np.random.rand(dim) - 0.5) * (2 * max_abs_val) for i in range(length)]


def vectorize(obj):
    return lambda x: np.array([obj(np.atleast_1d(x_i)) for x_i in x])

#
# vectorised version of naive_eval() in multivar_horner.helpers_fcts_numba
def naive_eval_reference(X, exponents, coefficients):
    return np.dot(np.array([[(x ** ex).prod() for ex in exponents] for x in X]), coefficients)
