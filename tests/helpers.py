import itertools
import random

import numpy as np

from multivar_horner.global_settings import FLOAT_DTYPE, UINT_DTYPE


def proto_test_case(data, fct):
    all_good = True
    for input, expected_output in data:
        print("\n")
        actual_output = fct(input)
        print(f"p({input[2]}) == {expected_output}")
        if actual_output != expected_output:
            print(f"ERROR: p(x) == {actual_output}")
            all_good = False
        else:
            print("OK.")

    assert all_good


def get_rnd_poly_properties(all_exponents, degree, max_abs_coeff=1.0, integer_coeffs=False, enforce_degree=False):
    """
    for meaningful test results uniformly distributed random test polynomials are required
    -> every possible polynomial should appear with equal probability
    <=> every 'sparsity' AND every possible monomial have equal probability

    sparsity: the ratio of 'active' monomials out of all possible monomials

    there must be at least 1 entry ('active' monomial)

    :param all_exponents:
    :param degree: the total degree the monomial must (at least) have
    :param max_abs_coeff:
    :param integer_coeffs:
    :return: evenly distributed settings of a polynomial with the given possible exponents
    """

    max_degree_monomial_idxs = all_exponents.sum(axis=1) >= degree
    assert np.any(max_degree_monomial_idxs)

    # every exponent can take the values in the range [0; maximal_degree]
    max_nr_exponent_vects = all_exponents.shape[0]

    # pick a random sparsity: controlled by the uniformly distributed probability a monomial is present
    activation_thres = random.random()
    row_activation = np.random.rand(max_nr_exponent_vects)
    row_idxs = row_activation > activation_thres

    if enforce_degree:
        # the settings must correspond to a polynomial of at least the requested total degree
        # -> at least one monomial of this total degree (or higher) must be active
        while not np.any(row_idxs & max_degree_monomial_idxs):
            row_activation = np.random.rand(max_nr_exponent_vects)
            row_idxs = row_activation > activation_thres
    else:
        if np.sum(row_idxs) == 0:  # there must be at least 1 entry ('active' monomial)
            row_idxs[random.randint(0, max_nr_exponent_vects - 1)] = 1

    exponents = all_exponents[row_idxs, :]
    nr_monomials = exponents.shape[0]
    coefficients = (np.random.rand(nr_monomials, 1) - 0.5) * (2 * max_abs_coeff)  # [ -max_abs_coeff; max_abs_coeff]
    coefficients = coefficients.astype(FLOAT_DTYPE)
    if integer_coeffs:
        coefficients = np.round(coefficients)

    return coefficients, exponents


def all_possible_exponents(dim, deg):
    """
    generate a fully occupied exponent matrix for a polynomial of lp_degree = infinity
    -> the given degree is the total degree of the polynomials
    :param dim:
    :param deg:
    :return:
    """

    # def cntr2exp_vect(cntr):
    #     exp_vect = np.empty(dim, dtype=UINT_DTYPE)
    #     for d in range(dim - 1, -1, -1):
    #         divisor = (deg + 1) ** d
    #         # cntr = quotient*divisor + remainder
    #         quotient, remainder = divmod(cntr, divisor)
    #         exp_vect[d] = quotient
    #         cntr = remainder
    #     return exp_vect

    max_nr_exponent_vects = (deg + 1) ** dim
    all_possible = np.empty((max_nr_exponent_vects, dim), dtype=UINT_DTYPE)

    # for i in range(max_nr_exponent_vects):
    #     all_possible[i] = cntr2exp_vect(i)

    for i, exponents in enumerate(itertools.product(range(deg + 1), repeat=dim)):
        all_possible[i] = np.asarray(exponents)

    # there must not be duplicate exponent vectors
    assert all_possible.shape == np.unique(all_possible, axis=0).shape

    return all_possible


def rnd_settings_list(length, dim, degree, *args, **kwargs):
    all_exponent_vect = all_possible_exponents(dim, degree)
    settings_list = [get_rnd_poly_properties(all_exponent_vect, degree, *args, **kwargs) for i in range(length)]

    # # random settings should have approx. half the amount of maximal entries on average
    # num_monomial_entries = 0
    # for settings in settings_list:
    #     num_monomial_entries += settings[0].shape[0]
    #
    # avg_monomial_entries = num_monomial_entries / length
    # max_monomial_entries = int((maximal_degree + 1) ** dim)
    # print(avg_monomial_entries, max_monomial_entries)
    return settings_list


def rnd_input_list(length, dim, max_abs_val):
    return [(np.random.rand(dim) - 0.5) * (2 * max_abs_val) for i in range(length)]


def vectorize(obj):
    return lambda x: np.array([obj(np.atleast_1d(x_i)) for x_i in x])


# vectorised version of naive_eval() in multivar_horner.helpers_fcts_numba
def naive_eval_reference(X, exponents, coefficients):
    return np.dot(np.array([[(x**ex).prod() for ex in exponents] for x in X]), coefficients)
