# -*- coding:utf-8 -*-

# NOTE: if this raises SIGSEGV, update your Numba dependency

# TODO compare difference in computed values of other methods (=numerical error)
# TODO test all input parameter conversions, and data rectifications
# TODO test derivative correctness
# TODO test factorisation for 1D polynomials! should always find optimum = unique 1D Horner factorisation
# TODO test: addresses of factors must never be target addresses
#  (value would be overwritten, but might need to be reused)
#   might not be relevant in the future any more with other computation procedures reusing addresses


import itertools
import unittest

import numpy as np
import pytest

from multivar_horner.global_settings import FLOAT_DTYPE, UINT_DTYPE
from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial
# settings for numerical stability tests
from tests.test_helpers import naive_eval_reference, proto_test_case, vectorize
from tests.test_settings import (
    COEFF_CHANGE_DATA, INPUT_DATA_INVALID_TYPES_CONSTRUCTION, INPUT_DATA_INVALID_TYPES_QUERY,
    INPUT_DATA_INVALID_VALUES_CONSTRUCTION, INPUT_DATA_INVALID_VALUES_QUERY,
    MAX_INP_MAGNITUDE, NR_TEST_POLYNOMIALS, VALID_TEST_DATA,
)


class MainTest(unittest.TestCase):

    # TODO split up tests into cross class test cases and single class test cases,
    #  then use inheritance to change test class
    def test_construction_basic(self):
        """
        test the basic construction API functionalities
        :return:
        """
        print('\nTESTING COSNTRUCTION API...')
        coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE)
        exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE)

        polynomial1 = MultivarPolynomial(coefficients, exponents, compute_representation=False)
        polynomial2 = MultivarPolynomial(coefficients, exponents, compute_representation=True)
        # both must have a string representation
        # [#ops=10] p(x)
        # [#ops=10] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
        assert len(str(polynomial1)) < len(str(polynomial2))
        assert str(polynomial1) == polynomial1.representation
        assert polynomial1.num_ops == 10

        return_str_repr = polynomial1.compute_string_representation(coeff_fmt_str='{:1.1e}',
                                                                    factor_fmt_str='(x{dim} ** {exp})')
        # the representation should get updated
        assert return_str_repr == polynomial1.representation

        polynomial1 = HornerMultivarPolynomial(coefficients, exponents, compute_representation=False)
        polynomial2 = HornerMultivarPolynomial(coefficients, exponents, compute_representation=True)
        r1 = str(polynomial1)  # [#ops=7] p(x)
        r2 = str(polynomial2)  # [#ops=7] p(x) = x_1 (x_1 (x_1 (1.0 x_2) + 2.0 x_3) + 3.0 x_2 x_3) + 5.0
        assert len(r1) < len(r2)
        assert r1 == polynomial1.representation
        assert r2 == polynomial2.representation
        assert polynomial1.num_ops == polynomial2.num_ops
        assert polynomial2.num_ops == 7

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

        def construction_should_raise(data, expected_error):
            for inp, expected_output in data:
                coeff, exp, x = inp
                with pytest.raises(expected_error):
                    MultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True, )
                with pytest.raises(expected_error):
                    HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True, find_optimal=False)
                with pytest.raises(expected_error):
                    HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True, find_optimal=True)

        construction_should_raise(INPUT_DATA_INVALID_TYPES_CONSTRUCTION, TypeError)
        construction_should_raise(INPUT_DATA_INVALID_VALUES_CONSTRUCTION, ValueError)

        # input rectification with negative exponents should raise a ValueError:
        coeff = [3.2]
        exp = [[0, 3, -4]]
        with pytest.raises(ValueError):
            HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=False, find_optimal=True)
        with pytest.raises(ValueError):
            MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=False)
        with pytest.raises(ValueError):
            HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=False, find_optimal=False)

        def query_should_raise(data, expected_error):
            for inp, expected_output in data:
                coeff, exp, x = inp
                # NOTE: construction must not raise an error:
                p = MultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True)
                with pytest.raises(expected_error):
                    p(x, rectify_input=False, validate_input=True)
                p = HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True)
                with pytest.raises(expected_error):
                    p(x, rectify_input=False, validate_input=True)
                p = HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True, find_optimal=True)
                with pytest.raises(expected_error):
                    p(x, rectify_input=False, validate_input=True)

        query_should_raise(INPUT_DATA_INVALID_TYPES_QUERY, TypeError)
        query_should_raise(INPUT_DATA_INVALID_VALUES_QUERY, ValueError)
        print('OK.')

    def test_eval_cases(self):
        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            res1 = poly.eval(x, rectify_input=True, validate_input=False)
            print('MultivarPolynomial', poly)

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False)
            res2 = poly.eval(x, rectify_input=True, validate_input=False)

            print('HornerMultivarPolynomial', horner_poly)
            # print('x=',x.tolist())
            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, find_optimal=True)
            res3 = poly.eval(x, rectify_input=True, validate_input=False)

            print('HornerMultivarPolynomial (optimal)', horner_poly_opt)
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
        degree_range = range(1, 4)
        dim_range = range(1, 4)
        for dim, deg in itertools.product(dim_range, degree_range):
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
        max_dim = 3
        degree_range = range(1, 4)
        exponents = np.array(list(itertools.product(degree_range, repeat=max_dim)), dtype=UINT_DTYPE)
        coeffs = np.random.rand(exponents.shape[0])
        X = np.random.rand(NR_TEST_POLYNOMIALS, max_dim)
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
            # p_np = lambda x: np.polyval(coeffs[::-1], x)
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
            # x = np.array(x).T
            print(x)
            poly = MultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            print(poly)
            poly = poly.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                            compute_representation=True, )
            print(poly)
            res1 = poly.eval(x, rectify_input=True, validate_input=True)

            print('\n')
            horner_poly = HornerMultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, keep_tree=True, find_optimal=False)
            print(horner_poly)
            horner_poly = horner_poly.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                                          compute_representation=True, )
            res2 = horner_poly.eval(x, rectify_input=True, validate_input=True)
            print(horner_poly)
            # print('x=',x.tolist())

            print('\n')
            horner_poly_opt = HornerMultivarPolynomial(coeffs1, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, keep_tree=True, find_optimal=True)
            print(horner_poly_opt)
            horner_poly_opt = horner_poly_opt.change_coefficients(coeffs2, rectify_input=True, validate_input=True,
                                                                  compute_representation=True, )
            res3 = horner_poly_opt.eval(x, rectify_input=True, validate_input=True)
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
            # x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                      compute_representation=True)
            poly = poly.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                            compute_representation=True, )
            print(poly)
            res1 = poly.eval(x, rectify_input=True, validate_input=True)

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False, keep_tree=True)
            horner_poly = horner_poly.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                                          compute_representation=True, )
            print(horner_poly)
            res2 = horner_poly.eval(x, rectify_input=True, validate_input=True)
            # print('x=',x.tolist())

            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, keep_tree=True, find_optimal=True)
            horner_poly_opt = horner_poly_opt.change_coefficients(coeff, rectify_input=True, validate_input=True,
                                                                  compute_representation=True, )
            print(horner_poly_opt)
            res3 = horner_poly_opt.eval(x, rectify_input=True, validate_input=True)
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print(f'x = {x}')
                print(f'results differ:\n{res1} (canonical)\n{res2} (horner)\n{res3} (horner optimal)\n')
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return res1

        proto_test_case(VALID_TEST_DATA, cmp_value_changed_coeffs_fct)
        print('OK.\n')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
