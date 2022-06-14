from math import log10
from os.path import abspath, join, pardir

import numpy as np
from numpy import array as a

from multivar_horner.global_settings import FLOAT_DTYPE, UINT_DTYPE

EXPORT_RESOLUTION = 300  # dpi
EXPORT_SIZE_X = 19.0  # inch
EXPORT_SIZE_Y = 11.0  # inch
SHOW_PLOTS = False
PLOTTING_DIR = abspath(join(__file__, pardir, "plots"))

NR_TEST_POLYNOMIALS = 5  # repetitions
MAX_COEFF_MAGNITUDE = 1e0
MAX_INP_MAGNITUDE = MAX_COEFF_MAGNITUDE  # max magnitude of evaluation points x
MAX_DIMENSION = 7
DIM_RANGE = list(range(1, MAX_DIMENSION + 1))
MAX_DEGREE = 7
DEGREE_RANGE = list(range(1, MAX_DEGREE + 1))

# speed tests
NR_SAMPLES_SPEED_TEST = 100
SPEED_RUN_PICKLE = "speed_results.pickle"

# numercial tests
NR_COEFF_CHANGES = 100  # controlling the noise for averaging the numerical error of a single polynomial run
TEST_RESULTS_PICKLE = "test_results.pickle"
DTYPE_HIGH_PREC = np.float128

# max allowed numerical error
# n orders of magnitudes less than the coefficients
# maximally the machine precision
MAX_ERR_EXPONENT = max(-15, (int(log10(MAX_COEFF_MAGNITUDE)) - 10))
MAX_NUMERICAL_ERROR = 10**MAX_ERR_EXPONENT

# TEST CASES:

# invalid:
# rectify_input=False, numpy arrays needed
# should raise TypeError
INPUT_DATA_INVALID_TYPES_CONSTRUCTION = [
    # not numpy.ndarray
    (
        (
            ([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]]),
            ([[3, 1, 0], [2, 0, 1], [1, 1, 1]]),
            a([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
    (
        (
            None,
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
    ((a([[1.0], [2.0], [3.0]]), None, a([-2.0, 3.0, 1.0])), 29.0),
    # incorrect dtype
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=int),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=float),
            a([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
]

INPUT_DATA_INVALID_TYPES_QUERY = [
    # not numpy.ndarray
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            ([-2.0, 3.0, 1.0]),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            None,
        ),
        29.0,
    ),
    # incorrect dtype
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=int),
        ),  # wrong
        29.0,
    ),
]

# should raise ValueError
INPUT_DATA_INVALID_VALUES_CONSTRUCTION = [
    # wrong shapes
    (
        (
            a([[1.0, 2.0, 3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    (
        (
            a([1.0, 2.0, 3.0], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    # different amount of coefficients and exponents
    (
        (
            a(
                [
                    [1.0],
                    [2.0],
                ],
                dtype=FLOAT_DTYPE,
            ),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0], [0.5]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    (
        (
            a(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                ],
                dtype=FLOAT_DTYPE,
            ),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1], [1, 1, 3]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    # duplicate exponent entries are not allowed
    (
        (
            a([[1.0], [2.0], [3.0]], dtype=FLOAT_DTYPE),
            a([[3, 1, 0], [2, 0, 1], [2, 0, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
    # no coefficients
    (
        (
            a([], dtype=FLOAT_DTYPE),
            a([], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0], dtype=FLOAT_DTYPE),
        ),
        29.0,
    ),
]

# should raise ValueError
INPUT_DATA_INVALID_VALUES_QUERY = [
    # query point x has wrong dimension
    (
        (
            a([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0]),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([-2.0, 3.0, 1.0, 0.1]),
        ),
        29.0,
    ),
    # wrong shape
    (
        (
            a([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([[-2.0, 3.0, 1.0]]),
        ),
        29.0,
    ),
    (
        (
            a([[1.0], [2.0], [3.0]]),
            a([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=UINT_DTYPE),
            a([[-2.0], [3.0], [1.0]]),
        ),
        29.0,
    ),
]

# rectify_input=True, numpy arrays not needed
VALID_TEST_DATA = [
    #
    # p(x) =  5.0
    (([5.0], [0], [0.0]), 5.0),  # coefficients  # exponents  # x  # p(x)
    # p(1.0) = 1.0
    (([5.0], [0], [1.0]), 5.0),
    # p(-1.0) = -1.0
    (([5.0], [0], [-1.0]), 5.0),
    # p(33.5) =33.5
    (([5.0], [0], [33.5]), 5.0),
    # p(x) =  1.0* x_1^1
    # p(0.0) = 0.0
    (([1.0], [1], [0.0]), 0.0),  # coefficients  # exponents  # x  # p(x)
    # p(1.0) = 1.0
    (([1.0], [1], [1.0]), 1.0),
    # p(-1.0) = -1.0
    (([1.0], [1], [-1.0]), -1.0),
    # p(33.5) =33.5
    (([1.0], [1], [33.5]), 33.5),
    # p(x) =  1.0* x_1^1 + 1.0 * x_2^1
    (([1.0, 1.0], [[1, 0], [0, 1]], [0.0, 0.0]), 0.0),
    (([1.0, 1.0], [[1, 0], [0, 1]], [1.0, 0.0]), 1.0),
    (([1.0, 1.0], [[1, 0], [0, 1]], [-1.0, 0.0]), -1.0),
    (([1.0, 1.0], [[1, 0], [0, 1]], [-1.0, 1.0]), 0.0),
    (([1.0, 1.0], [[1, 0], [0, 1]], [-1.0, -2.0]), -3.0),
    (([1.0, 1.0], [[1, 0], [0, 1]], [33.5, 0.0]), 33.5),
    # p(x) =  5.0 +  1.0* x_1^1
    (([5.0, 1.0], [[0, 0], [1, 0]], [0.0, 0.0]), 5.0),
    (([5.0, 1.0], [[0, 0], [1, 0]], [1.0, 0.0]), 6.0),
    (([5.0, 1.0], [[0, 0], [1, 0]], [-1.0, 0.0]), 4.0),
    (([5.0, 1.0], [[0, 0], [1, 0]], [33.5, 0.0]), 38.5),
    # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2
    (([5.0, 2.0, 1.0], [[0, 0], [1, 0], [2, 0]], [0.0, 0.0]), 5.0),
    (
        ([5.0, 2.0, 1.0], [[0, 0], [1, 0], [2, 0]], [1.0, 0.0]),
        8.0,
    ),  # p(x) =  5.0 + 2.0 + 1.0
    (
        ([5.0, 2.0, 1.0], [[0, 0], [1, 0], [2, 0]], [-1.0, 0.0]),
        4.0,
    ),  # p(x) =  5.0 - 2.0 + 1.0
    (
        ([5.0, 2.0, 1.0], [[0, 0], [1, 0], [2, 0]], [2.0, 0.0]),
        13.0,
    ),  # p(x) =  5.0 + 2.0* 2.0^1 + 1.0* 2.0^2
    # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2 + 2.0* x_1^2 *x_2^1
    (([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [0.0, 0.0]), 5.0),
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [1.0, 0.0]),
        8.0,
    ),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *0^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [1.0, 1.0]),
        10.0,
    ),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *1^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [-1.0, 0.0]),
        4.0,
    ),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *0^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [-1.0, 1.0]),
        6.0,
    ),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *1^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [-1.0, 2.0]),
        8.0,
    ),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *2^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [-1.0, 3.0]),
        10.0,
    ),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *3^1
    (
        ([5.0, 2.0, 1.0, 2.0], [[0, 0], [1, 0], [2, 0], [2, 1]], [-2.0, 3.0]),
        # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
        29.0,
    ),
    # as in paper: "Greedy Algorithms for Optimizing Multivariate Horner Schemes"
    # [20] p(x) = 1.0 x_1^3 x_2^1 + 1.0 x_1^2 x_3^1 + 1.0 x_1^2 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 1.0 ] ] + x_1^2 x_3^1 [ 1.0 ]
    (([1.0, 1.0, 1.0], [[3, 1, 0], [2, 0, 1], [2, 1, 1]], [1.0, 1.0, 1.0]), 3.0),
    # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
    (([1.0, 2.0, 3.0], [[3, 1, 0], [2, 0, 1], [1, 1, 1]], [-2.0, 3.0, 1.0]), -34.0),
    # [27] p(x) = 1.0 x_3^1 + 2.0 x_1^3 x_2^3 + 3.0 x_1^2 x_2^3 x_3^1 + 4.0 x_1^1 x_2^5 x_3^1
    (
        (
            [1.0, 2.0, 3.0, 4.0],
            [[0, 0, 1], [3, 3, 0], [2, 3, 1], [1, 5, 1]],
            [-2.0, 3.0, 1.0],
        ),
        -2051.0,
    ),
]
COEFF_CHANGE_DATA = [
    # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
    (
        (
            [1.0, 1.0, 1.0],  # coeffs1
            [[3, 1, 0], [2, 0, 1], [2, 1, 1]],
            [1.0, 1.0, 1.0],  # coeffs1
            [1.0, 2.0, 3.0],  # coeffs2
        ),
        6.0,
    ),
]
