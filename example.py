import numpy as np

from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

coefficients = np.array([[1.0], [2.0], [3.0]], dtype=np.float)
exponents = np.array([[3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint)

# regular polynomial representation
polynomial = MultivarPolynomial(coefficients, exponents)
print(polynomial)

x = np.array([-2.0, 3.0, 1.0], dtype=np.float)

p_x = polynomial.eval(x)
print(p_x)

# factorised polynomial representation
polynomial = HornerMultivarPolynomial(coefficients, exponents)
print(polynomial)

p_x = polynomial.eval(x)
print(p_x)
