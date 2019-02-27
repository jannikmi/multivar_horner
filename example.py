import numpy as np
from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

# input parameters defining the polynomial
#   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
#   with...
#       dimension N = 3
#       amount of monomials M = 4
#       max_degree D = 3
# IMPORTANT: the data types and shapes are required by the precompiled functions in helper_fcts_numba.py
coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # column numpy vector = (M,1)-matrix
exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)  # numpy (M,N)-matrix

# represent the polynomial in the regular, naive form without any factorisation (simply stores the matrices)
polynomial = MultivarPolynomial(coefficients, exponents)

# visualising the used polynomial representation
print(polynomial)
# [#ops=27] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
# NOTE: the number in square brackets indicates the number of operations required
#   to evaluate the polynomial (ADD, MUL, POW).
# NOTE: in the case of unfactorised polynomials many unnecessary operations are being done
# (internally uses numpy matrix operations)

# define a query point and evaluate the polynomial
x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy row vector (1,N)
p_x = polynomial.eval(x)
print(p_x)  # -29.0

# represent the polynomial in factorised form:
# uses the heuristic proposed in [1]: iteratively factors out the factor with the highest usage
# pass compute_representation=True in order to compile a string representation of the factorisation
# pass keep_tree=True when the factorisation tree should be kept after the factorisation process
horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, compute_representation=True)
print(horner_polynomial)  # [#ops=10] p(x) = x_1 (x_1 (x_1 (c x_2) + c x_3) + c x_2 x_3) + c

# pass rectify_input=True to automatically try to convert the input
#   to the required numpy data structure with the right data type and shape
# pass validate_input=True to check if input data is valid (e.g. only non negative exponents)
# NOTE: the default for both options is false (increased speed)
coefficients = [5.0, 1.0, 2.0, 3.0]  # must not be a column vector, but dimensions must still fit
exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, rectify_input=True, validate_input=True)

p_x = horner_polynomial.eval(x)
print(p_x)  # -29.0

# export the factorised polynomial
path = 'file_name.pickle'
horner_polynomial.export_pickle(path=path)

from multivar_horner.multivar_horner import load_pickle

# import a polynomial
horner_polynomial = load_pickle(path)
p_x = horner_polynomial.eval(x)
print(p_x)  # -29.0

# BETA feature:
# pass find_optimal=True to start an adapted A* search through all possible factorisations
# theoretically guaranteed to find the optimal solution
# NOTE: time and memory consumption is MUCH higher!
horner_polynomial_optimal = HornerMultivarPolynomial(coefficients, exponents, find_optimal=True,
                                                     compute_representation=True)
print(horner_polynomial_optimal)  # [#ops=10] p(x) = x_3 (x_1 (c x_1 + c x_2)) + c + c x_1^3 x_2

# BETA: untested features
# derivative and gradient of a polynomial
# NOTE: partial derivatives themselves will be instances of the same parent class
deriv_2 = horner_polynomial.get_partial_derivative(2)

grad = horner_polynomial.get_gradient()
