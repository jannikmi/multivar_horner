import numpy as np
from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

# input parameters defining the polynomial
#   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
#   with...
#       dimension N = 4
#       amount of monomials M = 4
#       max_degree D = 3
# IMPORTANT: the data types and shapes are required by the precompiled helpers in helper_fcts_numba.py
coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # column numpy vector = (M,1)-matrix
exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)  # numpy (M,N)-matrix

# represent the polynomial in the regular (naive) form
polynomial = MultivarPolynomial(coefficients, exponents)

# visualising the used polynomial representation
# NOTE: the number in square brackets indicates the required number of operations
#   to evaluate the polynomial (ADD, MUL, POW).
#   it is not equal to the visually identifiable number of operations (due to the internally used algorithms)
print(polynomial)  # [27] p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1

# define the query point
x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy row vector (1,N)

p1_x = polynomial.eval(x)
print(p1_x)  # -29.0

# represent the polynomial in the factorised (near to minimal) form
horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)
print(horner_polynomial)  # [15] p(x) = 5.0 + x_2^1 [ x_1^1 x_3^1 [ 3.0 ] + x_1^3 [ 1.0 ] ] + x_1^2 x_3^1 [ 2.0 ]

p2_x = horner_polynomial.eval(x)
print(p2_x)  # -29.0

# export the factorised polynomial
path = 'file_name.picke'
horner_polynomial.export_pickle(path=path)

from multivar_horner.multivar_horner import load_pickle

# import a polynomial
horner_polynomial = load_pickle(path)
