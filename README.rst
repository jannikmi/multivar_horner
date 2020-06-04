===============
multivar_horner
===============



.. image:: https://travis-ci.org/MrMinimal64/multivar_horner.svg?branch=master
    :target: https://travis-ci.org/MrMinimal64/multivar_horner

.. image:: https://img.shields.io/pypi/wheel/multivar-horner.svg
    :target: https://pypi.python.org/pypi/multivar-horner

.. image:: https://pepy.tech/badge/multivar-horner
    :alt: Total PyPI downloads
    :target: https://pepy.tech/project/multivar-horner

.. image:: https://img.shields.io/pypi/v/multivar_horner.svg
    :alt: latest version on PyPI
    :target: https://pypi.python.org/pypi/multivar-horner


``multivar_horner`` is a python package implementing a multivariate
`Horner scheme ("Horner's method", "Horner's rule") <https://en.wikipedia.org/wiki/Horner%27s_method>`__
for efficiently evaluating multivariate polynomials.


Quick Guide:

::


    pip install multivar_horner


.. code-block:: python

    import numpy as np
    from multivar_horner.multivar_horner import HornerMultivarPolynomial

    # input parameters defining the polynomial
    #   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)


    # [#ops=10] p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)
    p_x = horner_polynomial(x)


For more refer to the `documentation <https://multivar_horner.readthedocs.io/en/latest/>`__.


Also see:
`GitHub <https://github.com/MrMinimal64/multivar_horner>`__,
`PyPI <https://pypi.python.org/pypi/multivar_horner/>`__

