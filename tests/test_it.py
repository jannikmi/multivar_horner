import unittest

import numpy as np
import pytest

from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial


# TODO compare difference in computed values (=numerical error)
# TODO test consecutive evaluations of the same polynomial
# TODO 0 coefficients should be ignored
# TODO test all conversions, and data rectifications


# TODO debug wrong reprensentation


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
    ((
         [1.0, 1.0, 1.0],  # coeffs1
         [[3, 1, 0], [2, 0, 1], [2, 1, 1]],
         [1.0, 1.0, 1.0],
         [1.0, 2.0, 3.0],  # coeffs2
     ),
     6.0),
]


class MainTest(unittest.TestCase):

    def test_invalid_input_detection(self):

        print('\n\nTEST INVALID INPUT DETECTION')
        for inp, expected_output in INVALID_INPUT_DATA:
            coeff, exp, x = inp
            with pytest.raises(AssertionError):
                poly = MultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True,
                                          compute_representation=True)

            with pytest.raises(AssertionError):
                horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True,
                                                       compute_representation=True, find_optimal=False)

            with pytest.raises(AssertionError):
                horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=False, validate_input=True,
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
                print('results differ:', res1, res2, res3)
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return poly.eval(x, validate_input=True)

        print('TEST EVALUATION')
        proto_test_case(VALID_TEST_DATA, cmp_value_fct)

    def test_change_coefficients(self):

        print('TEST CHANGING COEFFICIENTS')

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
                print('results differ:', res1, res2, res3)
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return poly.eval(x, validate_input=True)

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
                print('results differ:', res1, res2, res3)
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return poly.eval(x, validate_input=True)

        proto_test_case(VALID_TEST_DATA, cmp_value_changed_coeffs_fct)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
