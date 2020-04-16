# -*- coding:utf-8 -*-

# NOTE: if this raises SIGSEGV, update your Numba dependency

# TODO compare difference in computed values of other methods (=numerical error)
# TODO test all input parameter conversions, and data rectifications
# TODO test derivative correctness
# TODO test: addresses of factors must never be target addresses
#  (value would be overwritten, but might need to be reused)
#   might not be relevant in the future any more with other computation procedures reusing addresses


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
from tests.test_helpers import rnd_settings_list, TEST_RESULTS_PICKLE, vectorize, naive_eval_reference

MAX_DIMENSION = 4
DIM_RANGE = list(range(1, MAX_DIMENSION))
MAX_DEGREE = 4
DEGREE_RANGE = list(range(1, MAX_DEGREE))
NR_TEST_POLYNOMIALS = 5  # repetitions
MAX_COEFF_MAGNITUDE = 1e0
MAX_INP_MAGNITUDE = MAX_COEFF_MAGNITUDE  # max magnitude of evaluation points x

# numercial tests
MAX_DEGREE_NUMERICAL_TEST = 10
NR_COEFF_CHANGES = 20

# numerical error
# n orders of magnitudes less than the coefficients
# maximally the machine precision
MAX_ERR_EXPONENT = max(-15, (int(log10(MAX_COEFF_MAGNITUDE)) - 10))
MAX_NUMERICAL_ERROR = 10 ** MAX_ERR_EXPONENT

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


class MainTest(unittest.TestCase):

    # TODO split up tests into cross class test cases and single class test cases,
    #  then use inheritance to change test class
    def test_construction_basic(self):
        """
        test the basic construction API functionalities
        :return:
        """
        print('\nTESTING COSNTRUCTION API...')
        coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)

        polynomial1 = MultivarPolynomial(coefficients, exponents, compute_representation=False)
        polynomial2 = MultivarPolynomial(coefficients, exponents, compute_representation=True)
        # both must have a string representation
        # [#ops=27] p(x)
        # [#ops=27] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
        assert len(str(polynomial1)) < len(str(polynomial2))
        assert str(polynomial1) == polynomial1.representation
        assert polynomial1.num_ops == 27

        return_str_repr = polynomial1.compute_string_representation(coeff_fmt_str='{:1.1e}',
                                                                    factor_fmt_str='(x{dim} ** {exp})')
        # the representation should get updated
        assert return_str_repr == polynomial1.representation

        polynomial1 = HornerMultivarPolynomial(coefficients, exponents, compute_representation=False)
        polynomial2 = HornerMultivarPolynomial(coefficients, exponents, compute_representation=True)
        # [#ops=10]
        # [#ops=10] p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0
        assert len(str(polynomial1)) < len(str(polynomial2))
        assert str(polynomial1) == polynomial1.representation
        assert polynomial1.num_ops == 10

        return_str_repr = polynomial1.compute_string_representation(coeff_fmt_str='{:1.1e}',
                                                                    factor_fmt_str='(x{dim} ** {exp})')
        # the representation should get updated
        assert return_str_repr == polynomial1.representation

        # converting the input to the required numpy data structures
        coefficients = [5.0, 1.0, 2.0, 3.0]
        exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
        horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, rectify_input=True, validate_input=True,
                                                     compute_representation=True, keep_tree=True)

        # search for an optimal factorisation
        horner_polynomial_optimal = HornerMultivarPolynomial(coefficients, exponents, find_optimal=True,
                                                             compute_representation=True, rectify_input=True,
                                                             validate_input=True)
        assert horner_polynomial_optimal.num_ops <= horner_polynomial.num_ops

        # partial derivative:
        deriv_2 = horner_polynomial.get_partial_derivative(2, compute_representation=True)

        # NOTE: partial derivatives themselves will be instances of the same parent class
        assert deriv_2.__class__ is horner_polynomial.__class__

        grad = horner_polynomial.get_gradient(compute_representation=True)
        # partial derivative for every dimension
        assert len(grad) == horner_polynomial.dim
        print('OK.\n')

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

    def test_eval_cases(self):
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

        print('\nTESTING EVALUATION CASES...')
        proto_test_case(VALID_TEST_DATA, cmp_value_fct)
        print('OK.\n')

    def test_eval_compare_vandermonde_nd(self):
        print('TEST COMPARISON TO VANDERMONDE POLYNOMIAL EVALUATION...')
        for dim, deg in itertools.product(DIM_RANGE, DEGREE_RANGE):
            exponents = np.array(list(itertools.product(range(deg), repeat=dim)), dtype=UINT_DTYPE)
            coeffs = np.random.rand(exponents.shape[0])
            X = np.random.rand(NR_TEST_POLYNOMIALS, dim)
            p_ref = naive_eval_reference(X, exponents, coeffs)
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents))
                assert np.allclose(p_ref, p_mv(X))

        print('OK.\n')

    def test_eval_api(self):
        print('\nTESTING EVALUATION API...')
        keys = ['rectify_input', 'validate_input']
        vals = [True, False]
        exponents = np.array(list(itertools.product(DEGREE_RANGE, repeat=MAX_DIMENSION)), dtype=UINT_DTYPE)
        coeffs = np.random.rand(exponents.shape[0])
        X = np.random.rand(NR_TEST_POLYNOMIALS, MAX_DIMENSION)
        p_ref = naive_eval_reference(X, exponents, coeffs)
        for kwargs in [dict(zip(keys, tf)) for tf in itertools.product(*[vals] * len(keys))]:
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents, **kwargs))
                assert np.allclose(p_ref, p_mv(X))

        print('OK.\n')

    def test_eval_compare_numpy_1d(self):
        print('TEST COMPARISON TO NUMPY 1D POLYNOMIAL EVALUATION...')
        x = MAX_INP_MAGNITUDE * ((np.random.rand(NR_TEST_POLYNOMIALS) * 2) - 1)
        for ncoeff in range(1, 8):
            exponents = np.arange(ncoeff, dtype=UINT_DTYPE).reshape(ncoeff, 1)
            coeffs = np.random.rand(ncoeff)
            ##p_np = lambda x: np.polyval(coeffs[::-1], x)
            p_np = np.polynomial.Polynomial(coeffs)
            for cls in [MultivarPolynomial, HornerMultivarPolynomial]:
                p_mv = vectorize(cls(coeffs[:, None], exponents))
                assert np.allclose(p_np(x), p_mv(x))

        print('OK.\n')

    def test_change_coefficients(self):

        print('\nTEST CHANGING COEFFICIENTS...')

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
        print('OK.\n')

    def test_numerical_stability(self):

        print('\nevaluating the numerical error:')
        results = []
        for dim, max_degree in product(DIM_RANGE, DEGREE_RANGE):
            results += evaluate_numerical_error(dim, max_degree)  # do not append list as entry

        with open(TEST_RESULTS_PICKLE, 'wb') as f:
            print(f'exporting numerical test results in {TEST_RESULTS_PICKLE}')
            pickle.dump(results, f)

        print('done.\n')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
