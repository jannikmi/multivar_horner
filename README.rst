===============
multivar_horner
===============



.. image:: https://travis-ci.org/MrMinimal64/multivar_horner.svg?branch=master
    :target: https://travis-ci.org/MrMinimal64/multivar_horner


.. image:: https://img.shields.io/pypi/wheel/multivar_horner.svg
    :target: https://pypi.python.org/pypi/multivar_horner


.. image:: https://img.shields.io/pypi/v/multivar_horner.svg
    :target: https://pypi.python.org/pypi/multivar_horner


A python package implementing a multivariate `horner scheme ("Horner's method", "Horner's rule") <https://en.wikipedia.org/wiki/Horner%27s_method>`__  for efficiently evaluating multivariate polynomials.

A polynomial is factorised according to a greedy heuristic similar to the one described in [1], with some additional computational tweaks.
This factorisation is being stored as a "Horner Tree". When the polynomial is fully factorized, a computational "recipe" for evaluating the polynomial is being compiled.
This "recipe" (stored internally as numpy arrays) allows the evaluation without the additional overhead of traversing the tree (= recursive function calls) and with functions precompiled by ``numba``.

**Advantage:** It is a near to minimal representation (in terms of storage and computational requirements) of a multivariate polynomial.

NOTE: an algorithm for finding the optimal factorisation of any multivariate polynomial is not known (to the best of my knowledge).

**Disadvantage:** Extra initial memory and computational effort is required in order to find the factorisation (cf. speed test results below).


Also see:
`GitHub <https://github.com/MrMinimal64/multivar_horner>`__,
`PyPI <https://pypi.python.org/pypi/multivar_horner/>`__


Dependencies
============

(``python3``),
``numpy``,
``Numba``


Installation
============


Installation with pip:

::

    pip install multivar_horner



Usage
=====

Check code in ``example.py``:


.. code-block:: python

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
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64) # numpy row vector (1,N)

    p_x = polynomial.eval(x)
    print(p_x)  # -29.0

    # represent the polynomial in the factorised (near to minimal) form
    # factors out the monomials! with the highest usage
    polynomial = HornerMultivarPolynomial(coefficients, exponents)
    print(polynomial)  # [15] p(x) = 5.0 + x_2^1 [ x_1^1 x_3^1 [ 3.0 ] + x_1^3 [ 1.0 ] ] + x_1^2 x_3^1 [ 2.0 ]

    p_x = polynomial.eval(x)
    print(p_x) # -29.0

    # in order to always just factor out the variable with the highest usage:
    # [17] p(x) = 5.0 + x_1^1 [ x_1^1 [ x_1^1 [ x_2^1 [ 1.0 ] ] + x_3^1 [ 2.0 ] ] + x_2^1 [ x_3^1 [ 3.0 ] ] ]
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, only_scalar_factors=True)



    # export the factorised polynomial
    path = 'file_name.picke'
    horner_polynomial.export_pickle(path=path)

    from multivar_horner.multivar_horner import load_pickle

    # import a polynomial
    horner_polynomial = load_pickle(path)



Speed Test Results
==================


::

    Speed test:
    testing 200 evenly distributed random polynomials

         parameters   |  setup time (/s)                        |  eval time (/s)                      |  # operations                        | lucrative after
    dim | max_deg | naive      | horner     | delta         | naive      | horner     | delta      | naive      | horner     | delta      |    # evals
    ================================================================================================================================================================
    1   | 1       | 0.007341   | 0.07008    | 8.5 x more    | 0.006645   | 0.0008059  | 7.2 x less | 3          | 2          | 0.5 x less | 11
    1   | 2       | 0.007411   | 0.1092     | 14 x more     | 0.00576    | 0.0007883  | 6.3 x less | 5          | 4          | 0.2 x less | 20
    1   | 3       | 0.009317   | 0.1233     | 12 x more     | 0.005666   | 0.0007961  | 6.1 x less | 6          | 6          | 0.0 x more | 23
    1   | 4       | 0.007432   | 0.1448     | 18 x more     | 0.005361   | 0.0007025  | 6.6 x less | 8          | 7          | 0.1 x less | 29
    1   | 5       | 0.006413   | 0.1635     | 24 x more     | 0.005284   | 0.00076    | 6.0 x less | 10         | 9          | 0.1 x less | 35

    2   | 1       | 0.008512   | 0.1188     | 13 x more     | 0.007274   | 0.0007133  | 9.2 x less | 12         | 6          | 1.0 x less | 17
    2   | 2       | 0.00733    | 0.2345     | 31 x more     | 0.005458   | 0.0008188  | 5.7 x less | 24         | 13         | 0.8 x less | 49
    2   | 3       | 0.007316   | 0.3743     | 50 x more     | 0.005742   | 0.001501   | 2.8 x less | 41         | 22         | 0.9 x less | 87
    2   | 4       | 0.006453   | 0.5611     | 86 x more     | 0.004923   | 0.00174    | 1.8 x less | 65         | 34         | 0.9 x less | 174
    2   | 5       | 0.00855    | 0.8063     | 93 x more     | 0.006074   | 0.0007677  | 6.9 x less | 96         | 49         | 1.0 x less | 150

    3   | 1       | 0.007197   | 0.2007     | 27 x more     | 0.004981   | 0.000889   | 4.6 x less | 32         | 11         | 1.9 x less | 47
    3   | 2       | 0.006693   | 0.6062     | 90 x more     | 0.005246   | 0.0007333  | 6.2 x less | 96         | 35         | 1.7 x less | 133
    3   | 3       | 0.006891   | 1.4688     | 212 x more    | 0.005721   | 0.001267   | 3.5 x less | 234        | 81         | 1.9 x less | 328
    3   | 4       | 0.007264   | 2.725      | 374 x more    | 0.006098   | 0.000823   | 6.4 x less | 456        | 151        | 2.0 x less | 515
    3   | 5       | 0.008042   | 4.6306     | 575 x more    | 0.00705    | 0.0009687  | 6.3 x less | 753        | 247        | 2.0 x less | 760

    4   | 1       | 0.006418   | 0.3839     | 59 x more     | 0.005019   | 0.0007817  | 5.4 x less | 80         | 22         | 2.6 x less | 89
    4   | 2       | 0.006871   | 1.6235     | 235 x more    | 0.005686   | 0.0008181  | 6.0 x less | 347        | 91         | 2.8 x less | 332
    4   | 3       | 0.007662   | 5.4757     | 714 x more    | 0.007902   | 0.0009026  | 7.8 x less | 1177       | 296        | 3.0 x less | 781
    4   | 4       | 0.00936    | 17.9225    | 1914 x more   | 0.01144    | 0.002003   | 4.7 x less | 2808       | 695        | 3.0 x less | 1899
    4   | 5       | 0.01299    | 33.6465    | 2590 x more   | 0.02014    | 0.001446   | 13 x less  | 5591       | 1369       | 3.1 x less | 1799

    5   | 1       | 0.009875   | 0.7415     | 74 x more     | 0.006086   | 0.001003   | 5.1 x less | 182        | 39         | 3.7 x less | 144
    5   | 2       | 0.00896    | 5.811      | 648 x more    | 0.008984   | 0.002073   | 3.3 x less | 1381       | 281        | 3.9 x less | 840
    5   | 3       | 0.01242    | 21.5646    | 1735 x more   | 0.01886    | 0.003408   | 4.5 x less | 5431       | 1097       | 4.0 x less | 1395
    5   | 4       | 0.02282    | 69.0324    | 3024 x more   | 0.0442     | 0.002326   | 18 x less  | 16740      | 3346       | 4.0 x less | 1648
    5   | 5       | 0.04214    | 186.1053   | 4415 x more   | 0.1008     | 0.004103   | 24 x less  | 43795      | 8584       | 4.1 x less | 1925


# TODO plots, then just link to github on the PyPI description


Contact
=======

Most certainly there is stuff I missed, things I could have optimized even further or explained more clearly, etc. I would be really glad to get some feedback on my code.

If you encounter any bugs, have suggestions etc.
do not hesitate to **open an Issue** or **add a Pull Requests** on Git.


License
=======

``multivar_horner`` is distributed under the terms of the MIT license
(see LICENSE.txt).



References
==========

[1] CEBERIO, Martine; KREINOVICH, Vladik. `Greedy Algorithms for Optimizing Multivariate Horner Schemes <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.330.7430&rep=rep1&type=pdf>`__. ACM SIGSAM Bulletin, 2004, 38. Jg., Nr. 1, S. 8-15.
