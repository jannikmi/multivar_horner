===============
multivar_horner
===============


.. image:: https://travis-ci.org/jannikmi/multivar_horner.svg?branch=master
    :alt: CI status
    :target: https://travis-ci.org/jannikmi/multivar_horner

.. image:: https://readthedocs.org/projects/multivar_horner/badge/?version=latest
    :alt: documentation status
    :target: https://multivar_horner.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/wheel/multivar-horner.svg
    :target: https://pypi.python.org/pypi/multivar-horner

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. image:: https://pepy.tech/badge/multivar-horner
    :alt: Total PyPI downloads
    :target: https://pepy.tech/project/multivar-horner

.. image:: https://img.shields.io/pypi/v/multivar_horner.svg
    :alt: latest version on PyPI
    :target: https://pypi.python.org/pypi/multivar-horner

.. image:: https://joss.theoj.org/papers/0b514c6894780f3cc81ed88c141631d4/status.svg
    :alt: JOSS status
    :target: https://joss.theoj.org/papers/0b514c6894780f3cc81ed88c141631d4

.. image:: https://zenodo.org/badge/155578190.svg
   :target: https://zenodo.org/badge/latestdoi/155578190

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


``multivar_horner`` is a python package implementing a multivariate
`Horner scheme ("Horner's method", "Horner's rule") <https://en.wikipedia.org/wiki/Horner%27s_method>`__
for efficiently evaluating multivariate polynomials.


Quick Guide:

::


    pip install multivar_horner


For efficiency this package is compiling the instructions required for polynomial evaluation to C by default.
If you don't have a C compiler (``gcc`` or ``cc``) installed you also need to install numba for using an alternative method:

::


    pip install multivar_horner[numba]


Simple example:

.. code-block:: python

    import numpy as np
    from multivar_horner import HornerMultivarPolynomial

    # input parameters defining the polynomial
    #   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)

    # [#ops=7] p(x) = x_1 (x_1 (x_1 (1.0 x_2) + 2.0 x_3) + 3.0 x_2 x_3) + 5.0
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents)
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)
    p_x = horner_polynomial(x)


For more refer to the `documentation <https://multivar_horner.readthedocs.io/en/latest/>`__.


Also see:
`GitHub <https://github.com/jannikmi/multivar_horner>`__,
`PyPI <https://pypi.python.org/pypi/multivar_horner/>`__,
`arXiv paper <https://arxiv.org/abs/2007.13152>`__
