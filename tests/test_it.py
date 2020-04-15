# -*- coding:utf-8 -*-

# NOTE: if this raises SIGSEGV, update your Numba dependency

# TODO compare difference in computed values of other methods (=numerical error)
# TODO test all input parameter conversions, and data rectifications
# TODO use numpy.testing.assert_allclose() or similar
# TODO test gradient
import itertools
import pickle
import sys
import unittest
from itertools import product
from math import log10

import numpy as np
import pytest

# import sys
# sys.path.insert(0, "../multivar_horner")
from multivar_horner.global_settings import UINT_DTYPE
from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial

# settings for numerical stability tests
from tests.test_helpers import rnd_settings_list, TEST_RESULTS_PICKLE, vander, vectorize

# numercial tests
DIM_RANGE = list(range(1, 8))
DEGREE_RANGE = list(range(1, 8))
MAX_DEGREE_NUMERICAL_TEST = 10
NR_TEST_POLYNOMIALS = 5
NR_COEFF_CHANGES = 20
MAX_COEFF_MAGNITUDE = 1e0
MAX_NUMERICAL_ERROR = 10 ** (log10(MAX_COEFF_MAGNITUDE) - 10)  # n orders of magnitudes less than the coefficients

INVALID_INPUT_DATA = [
    # calling with x of another dimension
    (([1.0, 2.0, 3.0],
      [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
      [-2.0, 3.0]),
     29.0),

    (([1.0, 2.0, 3.0],
      [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
      [-2.0, 3.0, 1.0, 4.0]),
     29.0),

    # negative exponents are not allowed
    (([1.0, 2.0, 3.0],
      [[3, -1, 0], [2, 0, 1], [1, 1, 1]],
      [-2.0, 3.0, 1.0, 4.0]),
     29.0),

    # duplicate exponent entries are not allowed
    (([1.0, 2.0, 3.0],
      [[3, 1, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]],
      [-2.0, 3.0, 1.0, 4.0]),
     29.0),

]

VALID_TEST_DATA = [
    #
    # p(x) =  5.0
    (([5.0],  # coefficients
      [0],  # exponents
      [0.0]),  # x
     5.0),  # p(x)

    # p(1.0) = 1.0
    (([5.0],
      [0],
      [1.0]),
     5.0),

    # p(-1.0) = -1.0
    (([5.0],
      [0],
      [-1.0]),
     5.0),

    # p(33.5) =33.5
    (([5.0],
      [0],
      [33.5]),
     5.0),

    # p(x) =  1.0* x_1^1
    # p(0.0) = 0.0
    (([1.0],  # coefficients
      [1],  # exponents
      [0.0]),  # x
     0.0),  # p(x)

    # p(1.0) = 1.0
    (([1.0],
      [1],
      [1.0]),
     1.0),

    # p(-1.0) = -1.0
    (([1.0],
      [1],
      [-1.0]),
     -1.0),

    # p(33.5) =33.5
    (([1.0],
      [1],
      [33.5]),
     33.5),

    # p(x) =  1.0* x_1^1 + 1.0 * x_2^1
    # TODO verify the length of the value arrat
    # TODO actually no computations needed for coefficients of 1.0
    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [0.0, 0.0]),
     0.0),

    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [1.0, 0.0]),
     1.0),

    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [-1.0, 0.0]),
     -1.0),

    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [-1.0, 1.0]),
     0.0),

    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [-1.0, -2.0]),
     -3.0),

    (([1.0, 1.0],
      [[1, 0], [0, 1]],
      [33.5, 0.0]),
     33.5),

    # p(x) =  5.0 +  1.0* x_1^1
    (([5.0, 1.0],
      [[0, 0], [1, 0]],
      [0.0, 0.0]),
     5.0),

    (([5.0, 1.0],
      [[0, 0], [1, 0]],
      [1.0, 0.0]),
     6.0),

    (([5.0, 1.0],
      [[0, 0], [1, 0]],
      [-1.0, 0.0]),
     4.0),

    (([5.0, 1.0],
      [[0, 0], [1, 0]],
      [33.5, 0.0]),
     38.5),

    # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2
    (([5.0, 2.0, 1.0],
      [[0, 0], [1, 0], [2, 0]],
      [0.0, 0.0]),
     5.0),

    (([5.0, 2.0, 1.0],
      [[0, 0], [1, 0], [2, 0]],
      [1.0, 0.0]),
     8.0),  # p(x) =  5.0 + 2.0 + 1.0

    (([5.0, 2.0, 1.0],
      [[0, 0], [1, 0], [2, 0]],
      [-1.0, 0.0]),
     4.0),  # p(x) =  5.0 - 2.0 + 1.0

    (([5.0, 2.0, 1.0],
      [[0, 0], [1, 0], [2, 0]],
      [2.0, 0.0]),
     13.0),  # p(x) =  5.0 + 2.0* 2.0^1 + 1.0* 2.0^2

    # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2 + 2.0* x_1^2 *x_2^1
    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [0.0, 0.0]),
     5.0),

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [1.0, 0.0]),
     8.0),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *0^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [1.0, 1.0]),
     10.0),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *1^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [-1.0, 0.0]),
     4.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *0^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [-1.0, 1.0]),
     6.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *1^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [-1.0, 2.0]),
     8.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *2^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [-1.0, 3.0]),
     10.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *3^1

    (([5.0, 2.0, 1.0, 2.0],
      [[0, 0], [1, 0], [2, 0], [2, 1]],
      [-2.0, 3.0]),
     # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
     29.0),

    # as in paper: "Greedy Algorithms for Optimizing Multivariate Horner Schemes"
    # [20] p(x) = 1.0 x_1^3 x_2^1 + 1.0 x_1^2 x_3^1 + 1.0 x_1^2 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 1.0 ] ] + x_1^2 x_3^1 [ 1.0 ]
    (([1.0, 1.0, 1.0],
      [[3, 1, 0], [2, 0, 1], [2, 1, 1]],
      [1.0, 1.0, 1.0]),
     3.0),

    # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
    (([1.0, 2.0, 3.0],
      [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
      [-2.0, 3.0, 1.0]),
     -34.0),

    # [27] p(x) = 1.0 x_3^1 + 2.0 x_1^3 x_2^3 + 3.0 x_1^2 x_2^3 x_3^1 + 4.0 x_1^1 x_2^5 x_3^1
    (([1.0, 2.0, 3.0, 4.0],
      [[0, 0, 1], [3, 3, 0], [2, 3, 1], [1, 5, 1]],
      [-2.0, 3.0, 1.0]),
     -2051.0),
]

COEFF_CHANGE_DATA = [

    # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
    (([1.0, 1.0, 1.0],  # coeffs1
      [[3, 1, 0], [2, 0, 1], [2, 1, 1]],
      [1.0, 1.0, 1.0],
      [1.0, 2.0, 3.0],  # coeffs2
      ),
     6.0),
]


def proto_test_case(data, fct):
    all_good = True
    for input, expected_output in data:
        print('\n')
        actual_output = fct(input)
        print(f'p({input[2]}) == {expected_output}')
        if actual_output != expected_output:
            print(f'ERROR: p(x) == {actual_output}')
            all_good = False
        else:
            print('OK.')

    assert all_good


# TODO dim

def evaluate_numerical_error(dim, max_degree):
    # basic idea: evaluating a polynomial at x = all 1 should give the sum of coefficients
    # -> any deviation is the numerical error
    results = []
    x = np.ones(dim, dtype=np.float)
    max_error = 0.0
    ctr_total = 0
    ctr_total_max = NR_TEST_POLYNOMIALS * NR_COEFF_CHANGES

    print(f'evaluating numerical error: dim: {dim}, max. degree: {max_degree} ...')
    for poly_ctr, (coefficients, exponents) in enumerate(rnd_settings_list(NR_TEST_POLYNOMIALS, dim, max_degree,
                                                                           max_abs_coeff=MAX_COEFF_MAGNITUDE,
                                                                           integer_coeffs=False)):
        # debug: validate_input=True
        nr_monomials = exponents.shape[0]
        # find factorisation (expensive)
        poly_horner = HornerMultivarPolynomial(coefficients, exponents, validate_input=True)
        ctr_total += 1

        for coeff_ctr in range(NR_COEFF_CHANGES):
            # simply change coefficients of the found factorisation (cheap)
            coefficients = (np.random.rand(nr_monomials, 1) - 0.5) * (2 * MAX_COEFF_MAGNITUDE)
            # is testing for in_place=True at the same time
            poly_horner.change_coefficients(coefficients, validate_input=True, in_place=True)
            p_x_horner = poly_horner.eval(x)

            poly = MultivarPolynomial(coefficients, exponents)
            p_x_expected = np.sum(coefficients)
            p_x = poly.eval(x)

            result = (poly, poly_horner, p_x_expected, p_x, p_x_horner)
            results.append(result)
            abs_numerical_error = abs(p_x_horner - p_x_expected)
            max_error = max(max_error, abs_numerical_error)
            sys.stdout.write(f'\r(poly #{poly_ctr} coeff #{coeff_ctr}, {ctr_total / ctr_total_max:.1%})'
                             f' max numerical error: {max_error:.2e}')
            sys.stdout.flush()
            if max_error > MAX_NUMERICAL_ERROR:
                # # DEBUG:
                # with open('coefficients.pickle', 'wb') as f:
                #     pickle.dump(coefficients, f)
                # with open('exponents.pickle', 'wb') as f:
                #     pickle.dump(exponents, f)
                raise AssertionError(f'numerical error {max_error:.2e} exceeded limit of {MAX_NUMERICAL_ERROR:.2e} ')

    print('\n... done.\n')
    return results


# def id2exponent_vect(prime_list, monomial_id):
#     # find the exponent vector corresponding to a monomial id
#     # = prime decomposition
#     exponent_vect = np.zeros(prime_list.shape, dtype=UINT_DTYPE)
#     current_id = monomial_id
#     for i, prime in enumerate(prime_list):
#         while 1:
#             quotient, remainder = divmod(current_id, prime)
#             if remainder == 0:
#                 exponent_vect[i] += 1
#                 current_id = quotient
#             else:
#                 break
#
#         if current_id == 0:
#             break
#
#     if current_id != 0:
#         raise ValueError('no factorisation found')
#
#     return exponent_vect
#
#
# def _sparse_range_generator(max_value, density):
#     for i in range(max_value):
#         if random.random() < density:
#             yield i


class MainTest(unittest.TestCase):

    def test_invalid_input_detection(self):

        print('\n\nTEST INVALID INPUT DETECTION...')
        for inp, expected_output in INVALID_INPUT_DATA:
            coeff, exp, x = inp
            with pytest.raises(AssertionError):
                MultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True, compute_representation=True)

            with pytest.raises(AssertionError):
                HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True,
                                         compute_representation=True, find_optimal=False)

            with pytest.raises(AssertionError):
                HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True,
                                         compute_representation=True, find_optimal=True)

        print('OK.')

    def test_eval(self):
        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            res1 = poly.eval(x, validate_input=True)
            print(poly)

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False)
            res2 = horner_poly.eval(x, validate_input=True)
            print(horner_poly)
            # print('x=',x.tolist())
            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, find_optimal=True)
            res3 = horner_poly_opt.eval(x, validate_input=True)
            print(horner_poly_opt)
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print(f'x = {x}')
                print(f'results differ:\n{res1} (canonical)\n{res2} (horner)\n{res3} (horner optimal)\n')
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return res1

        print('TEST EVALUATION...')
        proto_test_case(VALID_TEST_DATA, cmp_value_fct)
        print('OK.')

    def test_change_coefficients(self):

        print('TEST CHANGING COEFFICIENTS...')

        # Test if coefficients can actually be changed, representation should change accordingly

        def change_coeffs_fct(inp):
            print('\n')
            # changing the coefficients to the same coefficients should not alter the evaluation results
            # (reuse test data)

            coeffs1, exp, x, coeffs2 = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            print(poly)
            poly = poly.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                            compute_representation=True, )
            print(poly)
            res1 = poly.eval(x, validate_input=True)

            print('\n')
            horner_poly = HornerMultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, keep_tree=True, find_optimal=False)
            print(horner_poly)
            horner_poly = horner_poly.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                                          compute_representation=True, )
            res2 = horner_poly.eval(x, validate_input=True)
            print(horner_poly)
            # print('x=',x.tolist())

            print('\n')
            horner_poly_opt = HornerMultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, keep_tree=True, find_optimal=True)
            print(horner_poly_opt)
            horner_poly_opt = horner_poly_opt.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                                                  compute_representation=True, )
            res3 = horner_poly_opt.eval(x, validate_input=True)
            print(horner_poly_opt)
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print(f'x = {x}')
                print(f'results differ:\n{res1} (canonical)\n{res2} (horner)\n{res3} (horner optimal)\n')

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return res1

        # this also tests if the factorisation tree, can still be accessed after changing the coefficients
        # representation would otherwise be empty
        # keep_tree has to be True
        proto_test_case(COEFF_CHANGE_DATA, change_coeffs_fct)

        def cmp_value_changed_coeffs_fct(inp):
            # changing the coefficients to the same coefficients should not alter the evaluation results
            # (reuse test data)
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            poly = poly.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                            compute_representation=True, )
            print(poly)
            res1 = poly.eval(x, validate_input=True)

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False, keep_tree=True)
            horner_poly = horner_poly.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                                          compute_representation=True, )
            print(horner_poly)
            res2 = horner_poly.eval(x, validate_input=True)
            # print('x=',x.tolist())

            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, keep_tree=True, find_optimal=True)
            horner_poly_opt = horner_poly_opt.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                                                  compute_representation=True, )
            print(horner_poly_opt)
            res3 = horner_poly_opt.eval(x, validate_input=True)
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print(f'x = {x}')
                print(f'results differ:\n{res1} (canonical)\n{res2} (horner)\n{res3} (horner optimal)\n')
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return res1

        proto_test_case(VALID_TEST_DATA, cmp_value_changed_coeffs_fct)
        print('OK.')

    # TODO understand
    def test_eval_compare_numpy_1d(self):
        print('TEST COMPARISON TO NUMPY 1D POLYNOMIAL EVALUATION...')
        x = -10 + np.random.rand(100) * 20
        for ncoeff in range(1, 8):
            exponents = np.array(range(ncoeff), dtype=UINT_DTYPE).reshape(ncoeff, 1)
            coeffs = np.random.rand(len(exponents))
            ##p_np = lambda x: np.polyval(coeffs[::-1], x)
            p_np = np.polynomial.Polynomial(coeffs)
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents))
                assert np.allclose(p_np(x), p_mv(x))

        print('OK.\n')

    def test_eval_compare_vandermonde_nd(self):
        print('TEST COMPARISON TO VANDERMONDE POLYNOMIAL EVALUATION...')
        for ndim, deg in itertools.product(range(1, 5), range(1, 5)):
            exponents = np.array(list(itertools.product(range(deg), repeat=ndim)),
                                 dtype=UINT_DTYPE)
            coeffs = np.random.rand(exponents.shape[0])
            X = np.random.rand(100, ndim)
            V = vander(X, exponents)
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents))
                assert np.allclose(np.dot(V, coeffs), p_mv(X))

        print('OK.\n')

    def test_api(self):
        print('\nTESTING API...')
        ndim = 3
        deg = 3
        keys = ['rectify_input', 'validate_input']
        vals = [True, False]
        exponents = np.array(list(itertools.product(range(deg), repeat=ndim)),
                             dtype=UINT_DTYPE)
        coeffs = np.random.rand(exponents.shape[0])
        X = np.random.rand(10, ndim)
        V = vander(X, exponents)
        ref = np.dot(V, coeffs)
        for kwds in [dict(zip(keys, tf)) for tf in
                     itertools.product(*(vals,) * len(keys))]:
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents, **kwds))
                assert np.allclose(ref, p_mv(X))

        print('OK.\n')

        #
    # def test_numerical_stability(self):
    #
    #     print('evaluating the numerical error:')
    #     results = []
    #     for dim, max_degree in product(DIM_RANGE, DEGREE_RANGE):
    #         results += evaluate_numerical_error(dim, max_degree)  # do not append list as entry
    #
    #     with open(TEST_RESULTS_PICKLE, 'wb') as f:
    #         print(f'exporting numerical test results in {TEST_RESULTS_PICKLE}')
    #         pickle.dump(results, f)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
