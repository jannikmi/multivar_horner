import unittest
from math import sqrt
import numpy as np

import pytest

from multivar_horner import MultivarPolynomial


def proto_test_case(data, fct):
    for input, expected_output in data:
        # print(input, expected_output, fct(input))
        actual_output = fct(input)
        if actual_output != expected_output:
            print('input: {} expected: {} got: {}'.format(input, expected_output, actual_output))
        assert actual_output == expected_output


class MainTest(unittest.TestCase):

    def test_eval(self):
        def cmp_value_fct(inp):
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp)
            return poly.eval(x)

        # negative exponents are not allowed
        # TODO
        # with pytest.raises(AssertionError):
        #     environment.find_shortest_path(start_coordinates, goal_coordinates)

        test_data = [
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

            # p(x) =  1.0* x_1^1 + 0.0* * x_1^1
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
        ]

        proto_test_case(test_data, cmp_value_fct)

    # TODO test gradient




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
