---
title: 'multivar_horner: A Python package for computing Horner factorisations of multivariate polynomials'
tags:
    - python
    - mathematics
    - polynomial
    - evaluation
    - multivariate
    - horner
    - factorisation
    - factorization


authors:
    - name: Jannik Michelfeit
      orcid: 0000-0002-1819-6975
      affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
    - name: Technische Universität Dresden
      index: 1
    - name: Max Planck Institute of Molecular Cell Biology and Genetics
      index: 2
date: 20 April 2020
bibliography: paper.bib


---

# Summary

Polynomials are a central concept in mathematics and find application in a wide range of fields.
(Multivariate) polynomials have different possible mathematical representations and the beneficial properties of some representations are in great demand in many applications[@LeeFactorization2013; @leiserson2010efficient; @Hecht1].


The so called Horner factorisation is such a representation with beneficial properties.
Compared to the unfactorised representation of a multivariate polynomial, in the following called "canonical form", this representation offers some important advantages.
First of all the Horner factorisation is more compact in the sense that it requires less mathematical operations in order to evaluate the polynomial (cf. \autoref{fig:num_ops_growth}).
Consequently, evaluating a multivariate polynomial in Horner factorisation is faster and numerically more stable [@pena2000multivariate; @pena2000multivariate2; @greedyHorner] (cf. \autoref{fig:num_err_growth}).
These advantages come at the cost of an initial computational effort required to find the factorisation.

The `multivar_horner` Python package implements a multivariate Horner scheme ("Horner's method", "Horner's rule") [@horner1819xxi] and thereby allows computing Horner factorisations of multivariate polynomials given in canonical form.
Representing multivariate polynomials of arbitrary degree also in canonical form, computing derivatives of polynomials and evaluating polynomials at a given point are further features of the package.
Accordingly the package presented here can be helpful always when (multivariate) polynomials have to be evaluated efficiently, the numerical error of the polynomial evaluation has to be small or a compact representation of the polynomial is required.
This holds true for many applications applying numerical analysis.
One example use case where this package is already being employed are novel response surface methods [@michelfeitresponse] based on multivariate Netwon interploation [@Hecht1].


# Functionality

In its core `multivar_horner` implements a multivariate Horner scheme with the greedy heuristic presented in [@greedyHorner].
In the following the key functionality of this package is being outlined.
For a more details on polynomials and Horner factorisations please refer to the literature, e.g. [@neumaier2001introduction].

A polynomial in canonical form is a sum of monomials.
For a univariate polynomial, which can be written as $f(x) = a_0 + a_1 x + a_2 x^2 + ... + a_d x^d$ (canonical form), the Horner factorisation is unique: $f(x) = a_0 + x ( a_1 + x( ... x (a_d) ... )$
In the multivariate case however the factorisation is ambiguous, as there are multiple possible factors to factorise with.
The key functionality of `multivar_horner` is finding a good instance among the many possible Horner factorisations of a multivariate polynomial.

Let's consider the example multivariate polynomial in canonical form $p(x) = 5 + 1 x_1^3 x_2^1 + 2 x_1^2 x_3^1 + 3 x_1^1 x_2^1 x_3^1$.
The polynomial $p$ is the sum of $5$ monomials, has dimensionality $3$ and can also be written as $p(x) = 5 x_1^0 x_2^0 x_3^0 + 1 x_1^3 x_2^1 x_3^0 + 2 x_1^2 x_2^0 x_3^1 + 3 x_1^1 x_2^1 x_3^1$.
The coefficients of the monomials are $5$, $1$, $2$ and $3$ respectively.
It is trivial but computationally expensive to represent this kind of formulation with matrices and vectors and to evaluate it in this way.
In this particular case for example a polynomial evaluation would require 27 operations.
Due to its simplicity this kind of formulation is being used for defining multivariate polynomials as input.
The following code snipped shows how to use ``multivar_horner`` for computing a Horner factorisation of $p$ and evaluating $p$ at a point $x$:

```python
from multivar_horner import HornerMultivarPolynomial
coefficients = [5.0, 1.0, 2.0, 3.0]
exponents = [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]]
p = HornerMultivarPolynomial(coefficients, exponents, rectify_input=True)
# [#ops=10] p(x) = x_1 (x_1 (x_1 (1.0 x_2) + 2.0 x_3) + 3.0 x_2 x_3) + 5.0
x = [-2.0, 3.0, 1.0]
p_x = p.eval(x, rectify_input=True) # -29.0
```

The factorisation computed by ``multivar_horner`` is $p(x) =  x_1 (x_1 (x_1 (1 x_2) + 2 x_3) + 3 x_2 x_3) + 5$ and requires 10 operations for every polynomial evaluation.

This is achieved by recursively factorising with respect to the factor most commonly used in all monomials.
When no leaves of the resulting binary "Horner Factorisation Tree" can be factorised any more, a "recipe" for evaluating the polynomial is being compiled.
This recipe encodes all operations required to evaluate the polynomial in numpy arrays [@numpy].
This enables the use of functions just in time compiled by Numba [@numba], which cause the polynomial evaluation to be computationally efficient.
The just in time compiled functions are always being used, since a pure python polynomial evaluation would to some extent outweigh the benefits of Horner factorisation representations.


# Degrees of multivariate polynomials


It is important to note that in contrast to the one dimensional case, several concepts of degree exist for polynomials in multiple dimensions.
Following the notation of [@trefethen2017multivariate] the usual notion of degree of a polynomial, the maximal degree, is the maximal sum of exponents of all monomials.
This is equal to the maximal $l_1$-norm of all exponent vectors of the monomials.
Accordingly the euclidean degree is the maximal $l_2$-norm and the maximal degree is the maximal $l_{\infty}$-norm of all exponent vectors.
Refer to [@trefethen2017multivariate] for precise definitions.

A polynomial is called fully occupied with respect to a certain degree if all possible monomials having a smaller or equal degree are present.
The occupancy of a polynomial can then be defined as the amount of existing monomials relative to the fully occupied polynomial of this degree.
A fully occupied polynomial hence has an occupancy of $1$.


![the amount of coefficients of fully occupied polynomials of different degrees in 3 dimensions.\label{fig:num_coeff_growth}](num_coeff_growth.png)


The amount of coefficients (equal to the amount of possible monomials) in multiple dimensions highly depends on the type of degree a polynomial has (cf. \autoref{fig:num_coeff_growth}).
This effect intensifies as the dimensionality grows.


# Benchmarks

![numerical error of evaluating randomly generated polynomials of varying sizes.\label{fig:num_err_growth}](../docs/_static/num_err_growth.png)


For benchmarking our method the following procedure is used:
In order to draw polynomials with uniformly random occupancy, the probability of monomials being present is picked randomly.
For a fixed maximal degree $n$ in $m$ dimensions there are $(n+1)^m$ possible exponent vectors corresponding to monomials.
Each of these monomials is being activated with the chosen probability.

For each maximal degree up to 7 and until dimensionality 7, 5 polynomials were drawn randomly.
In order to compute the numerical error, each polynomial has been evaluated at the point of all ones.
The true result in this case should always be the sum of all coefficients.
Any deviation of the evaluation value from the sum of coefficients hence is numerical error.
In order to receive more representative results, the obtained numerical error is being averaged over 100 tries with uniformly random coefficients in the range $[-1; 1]$.

![numerical error of evaluating randomly generated polynomials in canonical form relative to the Horner factorisation.\label{fig:num_err_heatmap}](../docs/_static/num_err_heatmap.png)

Note that even though the original monomials are not actually present in a Horner factorisation, the amount of coefficients however is identical to the amount of coefficients of its canonical form.
With increasing size in terms of the amount of included coefficients the numerical error of both the canonical form and the Horner factorisation found by `multivar_horner` grow exponentially (cf. \autoref{fig:num_err_growth}).
However, in comparison to the canonical form, the Horner factorisation is more numerically stable as it has also been visualised in \autoref{fig:num_err_heatmap}.

Even though the amount of operations required for evaluating the polynomials grow exponentially with their size irrespective of the representation, the rate of growth is lower for the Horner factorisation (cf. \autoref{fig:num_ops_growth}).
As a result, the Horner factorisations are computationally easier to evaluate.

![amount of operations required to evaluate randomly generated polynomials.\label{fig:num_ops_growth}](../docs/_static/num_ops_growth.png)

# Related work

The package has been created due to the recent advances in multivariate polynomial interpolation [@Hecht1; @Hecht2].
High dimensional interpolants of large degrees create the demand for evaluating multivariate polynomials computationally efficient and numerically stable.
Among others, these advances enable modeling the behaviour of (physical) systems with polynomials.
Obtaining an analytical, multidimensional and nonlinear representation of a system opens up many possibilities.
With so called "interpolation response surface methods"[@michelfeitresponse] for example a system can be analysed and optimised.

[NumPy](https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html) [@numpy] offers functionality to represent and manipulate polynomials of dimensionality up to 3.
SymPy offers the dedicated module [sympy.polys](https://docs.sympy.org/latest/modules/polys/index.html) for symbolically operating with polynomials.
[Sage](https://doc.sagemath.org/html/en/reference/polynomial_rings/index.html) covers the algebraic side of polynomials.
The Julia package [StaticPolynomials](https://github.com/JuliaAlgebra/StaticPolynomials.jl) has a similar functionality, but it does not support computing Horner factorisations.

`multivar_horner` has no functions to directly interoperate with other software packages.
The generality of the required input parameters (coefficients and exponents) however still ensures the compatibility with other approaches.
It is for example easy to manipulate a polynomial with other libraries and then compute the Horner factorisation representation of the resulting output polynomial with `multivar_horner` afterwards, by simply transferring coefficients and exponents.
Some intermediary operations to convert the parameters into the required format might be necessary.


# Further reading

The documentation of the package is hosted on [readthedocs.io](https://multivar_horner.readthedocs.io/en/latest/).
Any bugs or feature requests can be issued on [GitHub](https://github.com/MrMinimal64/multivar_horner/issues) [@github].
The [contribution guidelines](https://github.com/MrMinimal64/multivar_horner/blob/master/CONTRIBUTING.rst) can be found there as well.

The underlying basic mathematical concepts are being explained in numerical analysis text books like [@neumaier2001introduction].
The Horner scheme at the core of `multivar_horner` has been theoretically outlined in [@greedyHorner].

Instead of using a heuristic to choose the next factor, one can allow a search over all possible Horner factorisations in order to arrive at a minimal factorisation.
The amount of possible factorisations, however, is increasing exponentially with the degree and dimensionality of a polynomial (the amount of monomials).
One possibility to avoid computing each factorisation is to employ a version of A-star search [@hart1968formal] adapted for factorisation trees.
`multivar_horner` also implements this approach, which is similar to the branch-and-bound method suggested in [@kojima2008efficient, ch. 3.1].

[@carnicer1990evaluation] shows how factorisation trees can be used to evaluate multivariate polynomials and their derivatives.
In [@kuipers2013improving] Monte Carlo tree search has been used to find more performant factorisations than with greedy heuristics.
Other beneficial representations of polynomials are for example being specified in [@LeeFactorization2013] and [@leiserson2010efficient].


# Acknowledgements

Thanks to Michael Hecht (Max Planck Institute of Molecular Cell Biology and Genetics) and Steve Schmerler (Helmholtz-Zentrum Dresden-Rossendorf) for valuable input enabling this publication.
I also thank the editor David P. Sanders for his helpful feedback.

TODO Reviewer contributions

# References
