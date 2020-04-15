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



Dependencies
============

(``python3.6+``),
``numpy``,
``numba``


Installation
============


Installation with pip:

::

    pip install multivar_horner



Usage
=====

Check this code in ``example.py``:


.. code-block:: python

    import numpy as np
    from multivar_horner.multivar_horner import MultivarPolynomial, HornerMultivarPolynomial

    # input parameters defining the polynomial
    #   p(x) = 5.0 + 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    #   with...
    #       dimension N = 3
    #       amount of monomials M = 4
    #       max_degree D = 3
    # NOTE: these data types and shapes are required by the precompiled functions in helper_fcts_numba.py
    coefficients = np.array([[5.0], [1.0], [2.0], [3.0]], dtype=np.float64)  # column numpy vector = (M,1)-matrix
    exponents = np.array([[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]], dtype=np.uint32)  # numpy (M,N)-matrix

    # represent the polynomial in the regular, naive form without any factorisation (simply stores the matrices)
    # pass compute_representation=True in order to compile a string representation of the factorised polynomial
    polynomial = MultivarPolynomial(coefficients, exponents, compute_representation=True)

    # visualising the used polynomial representation
    print(polynomial)
    # [#ops=27] p(x) = 5.0 x_1^0 x_2^0 x_3^0 + 1.0 x_1^3 x_2^1 x_3^0 + 2.0 x_1^2 x_2^0 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
    # NOTE: the number in square brackets indicates the number of operations required
    #   to evaluate the polynomial (ADD, MUL, POW).
    # NOTE: in the case of unfactorised polynomials many unnecessary operations are being done
    # (internally uses numpy matrix operations)

    # the formatting of the string representation can be changed with the parameters `coeff_fmt_str` and `factor_fmt_str`:
    polynomial.compute_string_representation(coeff_fmt_str='{:1.1e}', factor_fmt_str='(x{dim} ** {exp})')
    print(polynomial)
    # [#ops=27] p(x) = 5.0e+00 (x1 ** 0) (x2 ** 0) (x3 ** 0) + 1.0e+00 (x1 ** 3) (x2 ** 1) (x3 ** 0)
    #                   + 2.0e+00 (x1 ** 2) (x2 ** 0) (x3 ** 1) + 3.0e+00 (x1 ** 1) (x2 ** 1) (x3 ** 1)

    # define a query point and evaluate the polynomial
    x = np.array([-2.0, 3.0, 1.0], dtype=np.float64)  # numpy row vector (1,N)
    p_x = polynomial.eval(x)
    print(p_x)  # -29.0

    # represent the polynomial in factorised form:
    # uses the heuristic proposed in [1]: iteratively factors out the factor with the highest usage
    # pass keep_tree=True when the factorisation tree should be kept after the factorisation process
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, compute_representation=True)
    print(horner_polynomial)
    # [#ops=10] p(x) = x_1^1 (x_1^1 (x_1^1 (1.0 x_2^1) + 2.0 x_3^1) + 3.0 x_2^1 x_3^1) + 5.0


    # pass rectify_input=True to automatically try converting the input to the required numpy data structures
    # pass validate_input=True to check if input data is valid (e.g. only non negative exponents)
    # NOTE: the default for both options is false (increased speed)
    coefficients = [5.0, 1.0, 2.0, 3.0]  # must not be a column vector, but dimensions must still fit
    exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
    horner_polynomial = HornerMultivarPolynomial(coefficients, exponents, rectify_input=True, validate_input=True,
                                                 compute_representation=True, keep_tree=True)

    p_x = horner_polynomial.eval(x)
    print(p_x)  # -29.0

    # the coefficients of a polynomial representation can be changed "on the fly"
    # i.e. without recomputing the factorisation
    # with in_place=False a new polygon object is being generated
    # in order to access the polynomial string representation with the updated coefficients pass compute_representation=True
    # NOTE: the string representation of the Horner factorisation depends on the factorisation tree
    #   the polynomial object must hence have keep_tree=True
    new_coefficients = [7.0, 2.0, 0.5, 0.75]  # must not be a column vector, but dimensions must still fit
    new_polynomial = horner_polynomial.change_coefficients(new_coefficients, rectify_input=True, validate_input=True,
                                                           compute_representation=True, in_place=False)
    print(new_polynomial)

    # export the factorised polynomial
    path = 'file_name.pickle'
    horner_polynomial.export_pickle(path=path)

    from multivar_horner.multivar_horner import load_pickle

    # import a polynomial
    horner_polynomial = load_pickle(path)
    print(horner_polynomial)  # [#ops=10] p(x) = x_1 (x_1 (x_1 (1.0 x_2) + 2.0 x_3) + 3.0 x_2 x_3) + 5.0
    p_x = horner_polynomial.eval(x)
    print(p_x)  # -29.0

    # BETA:
    # pass find_optimal=True to start an adapted A* search through all possible factorisations
    # theoretically guaranteed to find the optimal solution
    # NOTE: time and memory consumption is MUCH higher! cf. Readme: "Optimal Horner Factorisation"
    horner_polynomial_optimal = HornerMultivarPolynomial(coefficients, exponents, find_optimal=True,
                                                         compute_representation=True, rectify_input=True,
                                                         validate_input=True)
    print(horner_polynomial_optimal)  # [#ops=10] p(x) = x_3 (x_1 (2.0 x_1 + 3.0 x_2)) + 5.0 + 1.0 x_1^3 x_2

    # BETA: untested features
    # derivative and gradient of a polynomial
    # NOTE: partial derivatives themselves will be instances of the same parent class
    deriv_2 = horner_polynomial.get_partial_derivative(2, compute_representation=True)
    print(deriv_2)  # [#ops=5] p(x) = x_1 (x_1^2 (1.0) + 3.0 x_3)

    grad = horner_polynomial.get_gradient(compute_representation=True)
    print(grad)
    # grad = [
    #     [#ops=8] p(x) = x_1 (x_1 (3.0 x_2) + 4.0 x_3) + 3.0 x_2 x_3,
    #     [#ops=5] p(x) = x_1 (x_1^2 (1.0) + 3.0 x_3),
    #     [#ops=4] p(x) = x_1 (x_1 (2.0) + 3.0 x_2)
    # ]







Optimal Horner Factorisation
============================


When passing ``find_optimal=True`` this package allows searching for an optimal factorisation:


.. code-block:: python

    horner_polynomial_optimal = HornerMultivarPolynomial(coefficients, exponents, find_optimal=True)



**Basic idea**:

Instead of using a heuristic to choose the next factor one can allow a search over all possible (meaningful) factorisations in order to arrive at a minimal Horner factorisation.
The amount of possible factorisations however is increasing exponentially with the degree of a polynomial and its amount of monomials.
One possibility to avoid computing each factorisation is to employ a version of A*-search adapted for factorisation trees:
• Initialise a set of all meaningful possible first level Newton factorisations
• Rank all factorisation according to a lower bound (“heuristic”) of their lowest possible amount of operations
• Iteratively factorise the most promising factorisation and update the heuristic
• Stop when the most promising factorisation is fully factorised

This approach is guaranteed to yield a minimal Horner factorisation, but its performance highly depends on the heuristic in use: Irrelevant factorisations are only being ignored if the heuristic is not too optimistic in estimating the amount of operations. On the other hand the heuristic must be easy to compute, because it would otherwise be computationally cheaper to just try all different factorisations.
Even though it missing to cover exponentiations, the branch-and-bound method suggested in [2, ch. 3.1] is almost identical to this procedure.

Even with a good heuristic this method is only traceable for small polynomials because of its increased resource requirements.
Since experiments show that factorisations obtained by choosing one factorisation according to a heuristic have the same or only a slightly higher amount of included operations[2, ch. 7], the computational effort of this approach is not justifiable in most cases.
A use case however is to compute and store a minimal representation of a polynomial in advance if possible.

**NOTES:**

* currently this approach seems to actually try all possible factorisations, because the heuristic in use is too optimistic (= brute force, improvements needed)
* This requires MUCH more memory and computing time than just trying one factorisation (the number of possible factorisations is growing exponentially with the size of the polynomial!).
* in the first test runs the results seemed to be identical (in terms of #ops) with the vanilla approach of just trying one factorisation!
* one could easily adapt this approach to find all optimal Horner factorisations
* in contrast to univariate polynomials there are possibly many optimal Horner factorisations of a multivariate polynomial. Even an optimal Horner factorisation must not be the globally minimal representation (other types factorisations possible: e.g. "algebraic factorisation", "common subexpression elimination")!




Contact
=======


Tell me if and how your are using this package. This encourages me to develop and test it further.

Most certainly there is stuff I missed, things I could have optimized even further or explained more clearly, etc.
I would be really glad to get some feedback.

If you encounter any bugs, have suggestions etc.
do not hesitate to **open an Issue** or **add a Pull Requests** on Git.



References
==========

[1] M. Ceberio and V. Kreinovich, `"Greedy Algorithms for Optimizing Multivariate Horner Schemes" <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.330.7430&rep=rep1&type=pdf>`__. ACM SIGSAM Bulletin, 2004, 38. Jg., Nr. 1, S. 8-15.

[2] M. Kojima, `“Efficient evaluation of polynomials and their partial derivatives in homotopy continuation methods” <https://pdfs.semanticscholar.org/db75/5d4f4127e43c0c81884fe2b1c8c48d292ccf.pdf>`__, Journal of the Operations Research Society of Japan, vol. 51, no. 1, pp. 29–54, 2008.

[3] J. M. Peña and T. Sauer, “On the multivariate Horner scheme”, SIAM journal on numerical analysis, vol. 37, no. 4, pp. 1186–1197, 2000.

[4] J. M. Peña and T. Sauer, “On the multivariate Horner scheme II: Running error analysis”, Computing, vol. 65, no. 4, pp. 313–322, 2000.
