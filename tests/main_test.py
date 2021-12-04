# -*- coding:utf-8 -*-

# NOTE: if this raises SIGSEGV, update your Numba dependency

# TODO compare difference in computed values of other methods (=numerical error)
# TODO test derivative correctness

import itertools
import unittest

import numpy as np
import pytest

from multivar_horner import HornerMultivarPolynomial, HornerMultivarPolynomialOpt, MultivarPolynomial
from multivar_horner.global_settings import FLOAT_DTYPE, UINT_DTYPE
from tests.helpers import naive_eval_reference, proto_test_case, vectorize
from tests.settings import (
    COEFF_CHANGE_DATA,
    INPUT_DATA_INVALID_TYPES_CONSTRUCTION,
    INPUT_DATA_INVALID_TYPES_QUERY,
    INPUT_DATA_INVALID_VALUES_CONSTRUCTION,
    INPUT_DATA_INVALID_VALUES_QUERY,
    MAX_INP_MAGNITUDE,
    NR_TEST_POLYNOMIALS,
    VALID_TEST_DATA,
)

DEFAULT_CONSTR_KWARGS = {
    "rectify_input": True,
    "compute_representation": True,
    "verbose": True,
    "store_c_instr": True,
    "store_numpy_recipe": True,
}


# TODO add cross class tests
class PerClassTestRegular(unittest.TestCase):
    # test cases testing a single polynomial class
    # inheritance used to change test class
    class2test = MultivarPolynomial

    def test_construction_basic(self):
        """
        test the basic construction API functionalities
        :return:
        """
        print("\nTESTING CONSTRUCTION API...")
        cls = self.class2test

        coefficients = [5.0, 1.0, 2.0, 3.0]
        exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
        coeffs_np = np.array(coefficients, dtype=FLOAT_DTYPE).reshape(4, 1)
        exp_np = np.array(exponents, dtype=UINT_DTYPE)

        p1 = cls(coeffs_np, exp_np, compute_representation=False)
        p2 = cls(coeffs_np, exp_np, compute_representation=True)
        # both must have a string representation
        # [#ops=10] p(x)
        # [#ops=10] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
        assert len(str(p1)) < len(str(p2))
        assert str(p1) == p1.representation

        return_str_repr = p1.compute_string_representation(coeff_fmt_str="{:1.1e}", factor_fmt_str="(x{dim} ** {exp})")
        # the representation should get updated
        assert return_str_repr == p1.representation

        # converting the input to the required numpy data structures automatically:
        p3 = cls(
            coefficients,
            exponents,
            rectify_input=True,
            compute_representation=True,
            keep_tree=True,
        )
        # the data structures should be stored as the desired numpy array
        np.array_equal(p3.coefficients, coeffs_np)
        np.array_equal(p3.exponents, exp_np)

        # partial derivative:
        # TODO own full test
        p3_deriv = p3.get_partial_derivative(2, compute_representation=True)
        # NOTE: partial derivatives themselves will be instances of the same parent class
        assert p3_deriv.__class__ is p3.__class__

        grad = p3.get_gradient(compute_representation=True)
        # TODO own full test
        # partial derivative for every dimension
        assert len(grad) == p3.dim
        print("OK.\n")

    def test_eval_api(self):
        print("\nTESTING EVALUATION API...")
        keys = ["rectify_input", "validate_input"]
        vals = [True, False]
        max_dim = 3
        degree_range = range(1, 4)
        exponents = np.array(list(itertools.product(degree_range, repeat=max_dim)), dtype=UINT_DTYPE)
        coeffs = np.random.rand(exponents.shape[0])
        X = np.random.rand(NR_TEST_POLYNOMIALS, max_dim)
        p_ref = naive_eval_reference(X, exponents, coeffs)
        for kwargs in [dict(zip(keys, tf)) for tf in itertools.product(*[vals] * len(keys))]:
            p_mv = vectorize(self.class2test(coeffs[:, None], exponents, **kwargs))
            assert np.allclose(p_ref, p_mv(X))

        print("OK.\n")

    def test_eval_compare_numpy_1d(self):
        print("TEST COMPARISON TO NUMPY 1D POLYNOMIAL EVALUATION...")
        x = MAX_INP_MAGNITUDE * ((np.random.rand(NR_TEST_POLYNOMIALS) * 2) - 1)
        for ncoeff in range(1, 8):
            exponents = np.arange(ncoeff, dtype=UINT_DTYPE).reshape(ncoeff, 1)
            coeffs = np.random.rand(ncoeff)
            # p_np = lambda x: np.polyval(coeffs[::-1], x)
            p_np = np.polynomial.Polynomial(coeffs)
            p_mv = vectorize(self.class2test(coeffs[:, None], exponents))
            results_multivar_horner = p_mv(x)
            results_numpy = p_np(x)
            assert np.allclose(results_numpy, results_multivar_horner)

        print("OK.\n")

    def test_eval_compare_vandermonde_nd(self):
        print("TEST COMPARISON TO VANDERMONDE POLYNOMIAL EVALUATION...")
        degree_range = range(1, 4)
        dim_range = range(1, 4)
        for dim, deg in itertools.product(dim_range, degree_range):
            exponents = np.array(list(itertools.product(range(deg), repeat=dim)), dtype=UINT_DTYPE)
            coeffs = np.random.rand(exponents.shape[0])
            X = np.random.rand(NR_TEST_POLYNOMIALS, dim)
            p_ref = naive_eval_reference(X, exponents, coeffs)
            p_mv = vectorize(self.class2test(coeffs[:, None], exponents))
            assert np.allclose(p_ref, p_mv(X))

        print("OK.\n")

    def test_eval_cases(self):
        print("\nTESTING STANDARD EVALUATION CASES...")
        cls = self.class2test

        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = cls(coeff, exp, **DEFAULT_CONSTR_KWARGS)
            print(cls, poly)
            res = poly(x, rectify_input=True)
            return res

        proto_test_case(VALID_TEST_DATA, cmp_value_fct)
        print("OK.\n")

    def test_invalid_input_detection(self):
        print("\n\nTEST INVALID INPUT DETECTION...")
        cls = self.class2test

        def construction_test(coeff, exp, expected_error, rectify: bool = False):
            with pytest.raises(expected_error):
                cls(coeff, exp, rectify_input=rectify)

        def construction_should_raise(data, expected_error):
            for inp, _ in data:
                coeff, exp, x = inp
                construction_test(coeff, exp, expected_error)

        construction_should_raise(INPUT_DATA_INVALID_TYPES_CONSTRUCTION, TypeError)
        construction_should_raise(INPUT_DATA_INVALID_VALUES_CONSTRUCTION, ValueError)

        # input rectification with negative exponents should raise a ValueError:
        coeff, exp = [3.2], [[0, 3, -4]]
        construction_test(coeff, exp, ValueError, rectify=True)

        def query_should_raise(data, expected_error):
            for inp, _ in data:
                coeff, exp, x = inp

                # NOTE: construction must not raise an error:
                p = cls(coeff, exp, rectify_input=False)
                with pytest.raises(expected_error):
                    p(x, rectify_input=False)

        query_should_raise(INPUT_DATA_INVALID_TYPES_QUERY, TypeError)
        query_should_raise(INPUT_DATA_INVALID_VALUES_QUERY, ValueError)
        print("OK.")

    def test_change_coefficients(self):
        print("\nTEST CHANGING COEFFICIENTS...")
        # Test if coefficients can actually be changed, representation should change accordingly
        cls = self.class2test

        def change_coeff_same_values(inp):
            # changing the coefficients while using the same coeff values should not alter the evaluation results
            # (reuse test data)
            coeff, exp, x = inp
            poly = cls(coeff, exp, rectify_input=True, compute_representation=True)
            poly = poly.change_coefficients(
                coeff,
                rectify_input=True,
                compute_representation=True,
            )
            print(poly)
            res = poly.eval(x, rectify_input=True)
            return res

        proto_test_case(VALID_TEST_DATA, change_coeff_same_values)

        def change_coeffs_fct(inp):
            print("\n")
            coeffs1, exp, x, coeffs2 = inp
            print(x)
            poly = cls(coeffs1, exp, **DEFAULT_CONSTR_KWARGS)
            print(poly)
            poly = poly.change_coefficients(
                coeffs2,
                rectify_input=True,
                compute_representation=True,
            )
            print(poly)
            res = poly(x, rectify_input=True)
            return res

        # this also tests if the factorisation tree, can still be accessed after changing the coefficients
        # representation would otherwise be empty
        # keep_tree has to be True
        proto_test_case(COEFF_CHANGE_DATA, change_coeffs_fct)
        print("OK.\n")


class PerClassTestHorner(PerClassTestRegular):
    class2test = HornerMultivarPolynomial

    def test_c_eval(self):
        print("\nTESTING C INSTRUCTION EVALUATION CASES (HORNER)...")
        cls = self.class2test

        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = cls(coeff, exp, **DEFAULT_CONSTR_KWARGS)
            print(cls, poly)
            res = poly._eval_c(x)
            return res

        print("\nTESTING EVALUATION CASES...")
        proto_test_case(VALID_TEST_DATA, cmp_value_fct)
        print("OK.\n")

    def test_recipe_eval(self):
        print("\nTESTING RECIPE EVALUATION CASES (HORNER)...")
        cls = self.class2test

        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = cls(coeff, exp, **DEFAULT_CONSTR_KWARGS)
            print(cls, poly)
            res = poly._eval_recipe(x)
            return res

        proto_test_case(VALID_TEST_DATA, cmp_value_fct)
        print("OK.\n")


class PerClassTestOptimal(PerClassTestRegular):
    class2test = HornerMultivarPolynomialOpt


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(PerClassTestRegular)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
