import unittest

import numpy as np
import pytest

from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial


def proto_test_case(data, fct):
    all_good = True
    for input, expected_output in data:
        # print(input, expected_output, fct(input))
        actual_output = fct(input)
        if actual_output != expected_output:
            print('input: {} expected: {} got: {}'.format(input, expected_output, actual_output))
            all_good = False

    assert all_good


# TODO compare difference in computed values (=numerical error)
# TODO test consecutive evaluations of the same polynomial!

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

    def test_eval(self):
        def cmp_value_fct(inp):
            print()
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True)
            res1 = poly.eval(x, validate_input=True)
            print(str(poly))

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False)
            res2 = horner_poly.eval(x, validate_input=True)
            print(str(horner_poly))
            # print('x=',x.tolist())

            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, find_optimal=True)
            res3 = horner_poly_opt.eval(x, validate_input=True)
            print(str(horner_poly_opt))
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print('results differ:', res1, res2, res3)
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return poly.eval(x, validate_input=True)

        invalid_test_data = [
            # calling with x of another dimension
            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            # negative exponents are not allowed
            (([1.0, 2.0, 3.0],
              [[3, -1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            # duplicate exponent entries are not allowed
            # negative exponents are not allowed
            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

        ]

        for inp, expected_output in invalid_test_data:
            with pytest.raises(AssertionError):
                cmp_value_fct(inp)

        valid_test_data = [
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

            # p(x) =  1.0* x_1^1 + 0.0* * x_2^1
            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [0.0, 0.0]),
             0.0),

            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [1.0, 0.0]),
             1.0),

            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [-1.0, 0.0]),
             -1.0),

            (([1.0, 0.0],
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
            # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
            # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
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

        proto_test_case(valid_test_data, cmp_value_fct)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
