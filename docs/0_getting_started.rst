===============
Getting started
===============



Installation
------------
Installation with pip:

::

    pip install multivar_horner



Basics
------

Let's consider this example multivariate polynomial:

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

This code shows how to compute the Horner factorisation of our example polynomial :math:`p`
and evaluating :math:`p` at a point :math:`x`:

.. code-block:: python

    import numpy as np
    from multivar_horner.multivar_horner import HornerMultivarPolynomial

    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # shape: (M,1)
    exponents = np.array([
            [0, 0, 0],
            [3, 1, 0],
            [2, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint32)  # shape: (M,N)
    p = HornerMultivarPolynomial(coefficients, exponents)

    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # shape: (1,N)
    p_x = p(x) # -29.0



.. note::

    with the default settings the input is required to have these data types and shapes


With the class ``HornerMultivarPolynomial`` a polynomial can be represented in :ref:`Horner factorisation <horner_usage>`.

With the class ``MultivarPolynomial`` a polynomial can be represented in :ref:`canonical form <canonical_usage>`.


All available features of this package are explained :ref:`HERE <usage>`.

The API documentation can be found :ref:`HERE <api>`.

