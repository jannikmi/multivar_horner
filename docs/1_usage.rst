.. _usage:

=====
Usage
=====

.. note::

   Also check out the :ref:`API documentation <api>` or the `code <https://github.com/MrMinimal64/multivar_horner>`__.


Let's look at the example multivariate polynomial:

:math:`p(x) = 5 + 1 x_1^3 x_2^1 + 2 x_1^2 x_3^1 + 3 x_1^1 x_2^1 x_3^1`


Which can also be written as:

:math:`p(x) = 5 x_1^0 x_2^0 x_3^0 + 1 x_1^3 x_2^1 x_3^0 + 2 x_1^2 x_2^0 x_3^1 + 3 x_1^1 x_2^1 x_3^1`

A polynomial is a sum of monomials.
Our example polynomial has :math:`M = 4` monomials and dimensionality :math:`N = 3`.

The coefficients of our example polynomial are: 5.0, 1.0, 2.0, 3.0

The exponent vectors of the corresponding monomials are:

* [0, 0, 0]
* [3, 1, 0]
* [2, 0, 1]
* [1, 1, 1]

To represent polynomials this package requires the coefficients and the exponent vectors as input.


.. code-block:: python

    import numpy as np

    coefficients = np.array(
        [[5.0], [1.0], [2.0], [3.0]], dtype=np.float64
    )  # numpy (M,1) ndarray
    exponents = np.array(
        [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32
    )  # numpy (M,N) ndarray


.. note::

    by default the Numba jit compiled functions require these data types and shapes



.. _horner_usage:

Horner factorisation
-----------------------------------------------


to create a representation of the multivariate polynomial :math:`p` in Horner factorisation:

.. code-block:: python

    from multivar_horner.multivar_horner import HornerMultivarPolynomial

    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)


the found factorisation is :math:`p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0`.


pass ``rectify_input=True`` to automatically try converting the input to the required ``numpy`` data structures

.. note::

    the default for both options is ``False`` for increased speed

.. note::

    the dtypes are fixed due to the just in time compiled ``Numba`` functions


.. code-block:: python


    coefficients = [5.0, 1.0, 2.0, 3.0]
    exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
    horner_polynomial = HornerMultivarPolynomial(
        coefficients, exponents, rectify_input=True
    )



pass ``keep_tree=True`` during construction of a Horner factorised polynomial,
when its factorisation tree should be kept after the factorisation process
(e.g. to be able to compute string representations of the polynomials later on)


.. code-block:: python

    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, keep_tree=True)



.. _canonical_usage:

canonical form
--------------

if ...

* the Horner factorisation takes too long
* the polynomial is going to be evaluated only a few times
* fast polynomial evaluation is not required or
* the numerical stability of the evaluation is not important

it is possible to represent the polynomial without any factorisation (refered to as 'canonical form' or 'normal form'):

.. note::

    in the case of unfactorised polynomials many unnecessary operations are being done
    (internally numpy matrix operations are being used)


.. code-block:: python

    from multivar_horner.multivar_horner import MultivarPolynomial

    polynomial = MultivarPolynomial(coefficients, exponents)




string representation
---------------------


in order to compile a string representation of a polynomial pass ``compute_representation=True`` during construction

.. note::

    the number in square brackets indicates the number of multiplications required
    to evaluate the polynomial.

.. note::

    exponentiations are counted as exponent - 1 operations, e.g. x^3 <-> 2 operations

.. code-block:: python

    polynomial = MultivarPolynomial(coefficients, exponents)
    print(polynomial)  # [#ops=10] p(x)


    polynomial = MultivarPolynomial(coefficients, exponents, compute_representation=True)
    print(polynomial)
    # [#ops=10] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1

    horner_polynomial = HornerMultivarPolynomial(
        coefficients, exponents, compute_representation=True
    )
    print(horner_polynomial.representation)
    # [#ops=7] p(x) = x_1 (x_1 (x_1 (1.0 x_2) + 2.0 x_3) + 3.0 x_2 x_3) + 5.0


the formatting of the string representation can be changed with the parameters ``coeff_fmt_str`` and ``factor_fmt_str``:

.. code-block:: python

    polynomial = MultivarPolynomial(
        coefficients,
        exponents,
        compute_representation=True,
        coeff_fmt_str="{:1.1e}",
        factor_fmt_str="(x{dim} ** {exp})",
    )


the string representation can be computed after construction as well.


.. note::

    for ``HornerMultivarPolynomial``: ``keep_tree=True`` is required at construction time


.. code-block:: python

    polynomial.compute_string_representation(
        coeff_fmt_str="{:1.1e}", factor_fmt_str="(x{dim} ** {exp})"
    )
    print(polynomial)
    # [#ops=10] p(x) = 5.0e+00 (x1 ** 0) (x2 ** 0) (x3 ** 0) + 1.0e+00 (x1 ** 3) (x2 ** 1) (x3 ** 0)
    #                   + 2.0e+00 (x1 ** 2) (x2 ** 0) (x3 ** 1) + 3.0e+00 (x1 ** 1) (x2 ** 1) (x3 ** 1)



change the coefficients of a polynomial
---------------------------------------

in order to access the polynomial string representation with the updated coefficients pass ``compute_representation=True``
with ``in_place=False`` a new polygon object is being generated


.. note::

    the string representation of a polynomial in Horner factorisation depends on the factorisation tree.
    the polynomial object must hence have keep_tree=True


.. code-block:: python

    new_coefficients = [
        7.0,
        2.0,
        0.5,
        0.75,
    ]  # must not be a ndarray, but the length must still fit
    new_polynomial = horner_polynomial.change_coefficients(
        new_coefficients,
        rectify_input=True,
        compute_representation=True,
        in_place=False,
    )



.. _optimal_usage:

optimal Horner factorisations
-----------------------------


pass ``find_optimal=True`` during construction of a Horner factorised polynomial
to start an adapted A* search through all possible factorisations.

See :ref:`this chapter <optimal>` for further information.


.. note::

    BETA: untested feature


.. note::

    time and memory consumption is MUCH higher!

.. code-block:: python

    horner_polynomial_optimal = HornerMultivarPolynomial(
        coefficients,
        exponents,
        find_optimal=True,
        compute_representation=True,
        rectify_input=True,
    )




caching polynomials
-------------------


export

.. code-block:: python

    path = "file_name.pickle"
    polynomial.export_pickle(path=path)


import

.. code-block:: python

    from multivar_horner.multivar_horner import load_pickle

    horner_polynomial = load_pickle(path)




evaluating a polynomial
-----------------------

in order to evaluate a polynomial at a point ``x``:


.. code-block:: python

    # define a query point and evaluate the polynomial
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy ndarray with shape [N]
    p_x = polynomial(x)  # -29.0


or


.. code-block:: python

    p_x = polynomial.eval(x)  # -29.0


or

.. code-block:: python

    x = [-2.0, 3.0, 1.0]
    p_x = polynomial.eval(x, rectify_input=True)  # -29.0


As during construction of a polynomial instance, pass ``rectify_input=True`` to automatically try converting the input to the required ``numpy`` data structure.


.. note::

    the default for both options is ``False`` for increased speed

.. note::

    the dtypes are fixed due to the just in time compiled ``Numba`` functions


computing the partial derivative of a polynomial
------------------------------------------------


.. note::

    BETA: untested feature


.. note::

    partial derivatives will be instances of the same parent class



.. note::

    all given additional arguments will be passed to the constructor of the derivative polynomial


.. note::

    dimension counting starts with 1 -> the first dimension is #1!


.. code-block:: python

    deriv_2 = polynomial.get_partial_derivative(2, compute_representation=True)
    # p(x) = x_1 (x_1^2 (1.0) + 3.0 x_3)




computing the gradient of a polynomial
------------------------------------------------

.. note::

    BETA: untested feature



.. note::

    all given additional arguments will be passed to the constructor of the derivative polynomials



.. code-block:: python

    grad = polynomial.get_gradient(compute_representation=True)
    # grad = [
    #     p(x) = x_1 (x_1 (3.0 x_2) + 4.0 x_3) + 3.0 x_2 x_3,
    #     p(x) = x_1 (x_1^2 (1.0) + 3.0 x_3),
    #     p(x) = x_1 (x_1 (2.0) + 3.0 x_2)
    # ]
