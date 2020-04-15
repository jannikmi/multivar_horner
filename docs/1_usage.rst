.. _usage:

=====
Usage
=====


.. TODO link


.. note::

    For a quick overview check out ``example.py``


.. note::

    For a more detailed documentation of all the features please confer to the API documentation
    and the comments in the code.



Initialisation
--------------


Create a representation of a multivariate polynomial in Horner factorisation:

.. code-block:: python

    import numpy as np

    # input parameters defining the polynomial
    #   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    #   with...
    #       dimension N = 3
    #       amount of monomials M = 4
    #       max_degree D = 3
    # NOTE: these data types and shapes are required by the Numba jit compiled functions
    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # numpy (M,1) ndarray
    exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)  # numpy (M,N) ndarray

    from multivar_horner.multivar_horner import HornerMultivarPolynomial

    # Horner factorisation:
    # [#ops=10] p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)




pass keep_tree=True during construction of a Horner factorised polynomial,
when its factorisation tree should be kept after the factorisation process


.. code-block:: python

    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, keep_tree=True)



pass rectify_input=True to automatically try converting the input to the required numpy data structures
pass validate_input=True to check if input data is valid (e.g. only non negative exponents)

.. note::

    the default for both options is false for increased speed


.. code-block:: python


    coefficients = [5.0, 1.0, 2.0, 3.0]
    exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, rectify_input=True, validate_input=True)



represent the polynomial in the canonical form (without any factorisation):

.. code-block:: python

    from multivar_horner.multivar_horner import MultivarPolynomial
    polynomial = MultivarPolynomial(coefficients, exponents)




in order to compile a string representation of the factorised polynomial pass compute_representation=True


.. code-block:: python

    print(polynomial) # [#ops=27] p(x)

    polynomial = MultivarPolynomial(coefficients, exponents, compute_representation=True)
    print(polynomial)
    # [#ops=27] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # NOTE: the number in square brackets indicates the number of operations required
    #   to evaluate the polynomial (ADD, MUL, POW).
    # NOTE: in the case of unfactorised polynomials many unnecessary operations are being done
    # (internally uses numpy matrix operations)



the string representation can be computed after construction as well.
the formatting of the string representation can be changed with the parameters `coeff_fmt_str` and `factor_fmt_str`:


.. note::

    for HornerMultivarPolynomial: keep_tree=True is required at construction time


.. code-block:: python

    polynomial.compute_string_representation(coeff_fmt_str='{:1.1e}', factor_fmt_str='(x{dim} ** {exp})')
    print(polynomial)
    # [#ops=27] p(x) = 5.0e+00 (x1 ** 0) (x2 ** 0) (x3 ** 0) + 1.0e+00 (x1 ** 3) (x2 ** 1) (x3 ** 0)
    #                   + 2.0e+00 (x1 ** 2) (x2 ** 0) (x3 ** 1) + 3.0e+00 (x1 ** 1) (x2 ** 1) (x3 ** 1)





.. TODO eval by class call explain
func = MultivarPolynomial(..)
y = func(x)


to change the coefficients of a polynomial:
in order to access the polynomial string representation with the updated coefficients pass compute_representation=True
with in_place=False a new polygon object is being generated

.. note::

    the string representation of a polynomial in Horner factorisation depends on the factorisation tree.
    the polynomial object must hence have keep_tree=True


.. code-block:: python

    new_coefficients = [7.0, 2.0, 0.5, 0.75]  # must not be a ndarray, but the length must still fit
    new_polynomial = horner_polynomial.change_coefficients(new_coefficients, rectify_input=True, validate_input=True,
                                                           compute_representation=True, in_place=False)





new_coefficients = [7.0, 2.0, 0.5, 0.75]
new_polynomial = horner_polynomial.change_coefficients(new_coefficients, rectify_input=True, validate_input=True,
                                                       compute_representation=True, in_place=False)




evaluating a polynomial
-----------------------

in order to evaluate a polynomial at a point x:



.. code-block:: python
    # define a query point and evaluate the polynomial
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy (1,N) ndarray
    p_x = polynomial(x) # -29.0


or


.. code-block:: python

    p_x = polynomial.eval(x)  # -29.0




computing the derivative of a polynomial
----------------------------------------



