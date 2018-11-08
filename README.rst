===============
multivar_horner
===============



.. image:: https://travis-ci.org/MrMinimal64/multivar_horner.svg?branch=master
    :target: https://travis-ci.org/MrMinimal64/multivar_horner


.. image:: https://img.shields.io/pypi/wheel/multivar_horner.svg
    :target: https://pypi.python.org/pypi/multivar_horner


.. image:: https://img.shields.io/pypi/v/multivar_horner.svg
    :target: https://pypi.python.org/pypi/multivar_horner


A python package implementing a multivariate horner scheme for efficiently evaluating multivariate polynomials.

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


::

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
    polynomial = HornerMultivarPolynomial(coefficients, exponents)
    print(polynomial)  # [15] p(x) = 5.0 + x_2^1 [ x_1^1 x_3^1 [ 3.0 ] + x_1^3 [ 1.0 ] ] + x_1^2 x_3^1 [ 2.0 ]

    p_x = polynomial.eval(x)
    print(p_x) # -29.0


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
    dim | max_deg | naive      | horner     | delta         | naive      | horner     | delta      | naive      | horner     | delta      |     # evals
    ================================================================================================================================================================
    1   | 1       | 0.005892   | 0.06047    | 9.3 x more    | 0.004712   | 0.0006059  | 6.8 x less | 4          | 2          | 1.0 x less | 13
    1   | 2       | 0.005843   | 0.08022    | 13 x more     | 0.004772   | 0.0007251  | 5.6 x less | 5          | 4          | 0.2 x less | 18
    1   | 3       | 0.006712   | 0.1014     | 14 x more     | 0.004644   | 0.0006409  | 6.2 x less | 7          | 6          | 0.2 x less | 24
    1   | 4       | 0.006179   | 0.1375     | 21 x more     | 0.004477   | 0.0007467  | 5.0 x less | 8          | 7          | 0.1 x less | 35
    1   | 5       | 0.006059   | 0.1432     | 23 x more     | 0.004627   | 0.0006215  | 6.4 x less | 9          | 9          | 0.0 x more | 34

    2   | 1       | 0.007445   | 0.1129     | 14 x more     | 0.00638    | 0.0006415  | 8.9 x less | 12         | 5          | 1.4 x less | 18
    2   | 2       | 0.006407   | 0.2159     | 33 x more     | 0.004935   | 0.0006426  | 6.7 x less | 24         | 13         | 0.8 x less | 49
    2   | 3       | 0.006468   | 0.3441     | 52 x more     | 0.004673   | 0.001437   | 2.3 x less | 43         | 23         | 0.9 x less | 104
    2   | 4       | 0.006288   | 0.5189     | 82 x more     | 0.004837   | 0.0007046  | 5.9 x less | 63         | 33         | 0.9 x less | 124
    2   | 5       | 0.006339   | 0.781      | 122 x more    | 0.004598   | 0.0006951  | 5.6 x less | 95         | 48         | 1.0 x less | 198

    3   | 1       | 0.006838   | 0.1746     | 24 x more     | 0.005002   | 0.000662   | 6.6 x less | 30         | 11         | 1.7 x less | 39
    3   | 2       | 0.00725    | 0.5564     | 76 x more     | 0.005399   | 0.0007045  | 6.7 x less | 102        | 36         | 1.8 x less | 117
    3   | 3       | 0.006433   | 1.3021     | 201 x more    | 0.005007   | 0.0008305  | 5.0 x less | 229        | 79         | 1.9 x less | 310
    3   | 4       | 0.007125   | 2.5504     | 357 x more    | 0.005796   | 0.0008626  | 5.7 x less | 448        | 149        | 2.0 x less | 516
    3   | 5       | 0.007424   | 4.4592     | 600 x more    | 0.006275   | 0.000861   | 6.3 x less | 767        | 251        | 2.1 x less | 822

    4   | 1       | 0.01081    | 0.3098     | 28 x more     | 0.006394   | 0.0007032  | 8.1 x less | 71         | 20         | 2.5 x less | 53
    4   | 2       | 0.007201   | 1.5558     | 215 x more    | 0.008119   | 0.0007251  | 10 x less  | 349        | 92         | 2.8 x less | 209
    4   | 3       | 0.007502   | 5.491      | 731 x more    | 0.007367   | 0.0008973  | 7.2 x less | 1239       | 310        | 3.0 x less | 848
    4   | 4       | 0.009323   | 13.3025    | 1426 x more   | 0.01137    | 0.001002   | 10 x less  | 2882       | 713        | 3.0 x less | 1282
    4   | 5       | 0.01223    | 27.3097    | 2232 x more   | 0.01822    | 0.00138    | 12 x less  | 5790       | 1417       | 3.1 x less | 1621

    5   | 1       | 0.01036    | 0.6624     | 63 x more     | 0.006455   | 0.0007832  | 7.2 x less | 188        | 40         | 3.7 x less | 115
    5   | 2       | 0.007524   | 4.7011     | 624 x more    | 0.007443   | 0.0008885  | 7.4 x less | 1333       | 274        | 3.9 x less | 716
    5   | 3       | 0.01195    | 21.4938    | 1798 x more   | 0.01708    | 0.001224   | 13 x less  | 5825       | 1164       | 4.0 x less | 1355
    5   | 4       | 0.02044    | 67.563     | 3304 x more   | 0.04303    | 0.002114   | 19 x less  | 17079      | 3395       | 4.0 x less | 1651
    5   | 5       | 0.04125    | 169.8522   | 4116 x more   | 0.102      | 0.003819   | 26 x less  | 40348      | 7978       | 4.1 x less | 1729


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
