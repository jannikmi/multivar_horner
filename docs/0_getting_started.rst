

===============
Getting started
===============



Installation
------------
Installation with pip:

::

    pip install multivar_horner


Dependencies
------------


(``python3.6+``),
``numpy``,
``numba``



Basics
------



.. code-block:: python

    import numpy as np
    from multivar_horner.multivar_horner import HornerMultivarPolynomial

    # input parameters defining the polynomial
    #   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    #   with...
    #       dimension N = 3
    #       amount of monomials M = 4
    #       max_degree D = 3
    # NOTE: these data types and shapes are required by the Numba jit compiled functions
    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # numpy (M,1) ndarray
    exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)  # numpy (M,N) ndarray

    # represent the polynomial in factorised form:
    # [#ops=10] p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)

     # define a query point
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy (1,N) ndarray
    # evaluate the polynomial
    p_x = horner_polynomial(x) # -29.0


The available functionalities of this package are explained :ref:`HERE <usage>`.
The API documentation can be fround :ref:`HERE <api>`.


.. TODO API link