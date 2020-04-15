Changelog
=========


1.3.1 (2020)
__________________


* added __call__ method for evaluating a polynomial in a simplified notation v=p(x)
* added test for numerical stability
* added plotting features for evaluation the numerical stability
* TODO added tests comparing functionality to other approaches
* clarified docstrings
* split up requirements.txt (into basic dependencies and test dependencies)
* added class AbstractPolynomial
* TODO added sphinx documentation
* TODO updated benchmark results
* TODO removed goedel id


1.3.0 (2020-03-14)
__________________


* NEW FEATURE: changing coefficients on the fly with `poly.change_coefficients(coeffs)`
* NEW DEPENDENCY: python3.6+ (for using f'' format strings)
* the real valued coefficients are now included in the string representation of a factorised polynomial
* add contribution guidelines
* added instructions in readme, example.py
* restructured the factorisation routine (simplified, clean up)
* extended tests


1.2.0 (2019-05-19)
__________________

* support of newer numpy versions (ndarray.max() not supported)
* added plotting routine (partly taken from tests)
* added plots in readme
* included latest insights into readme


1.1.0 (2019-02-27)
__________________

* added option `find_optimal` to find an optimal factorisation with A* search, explanation in readme
* optimized heuristic factorisation (more clean approach using just binary trees)
* dropped option `univariate_factors`
* added option `compute_representation` to compute the string representation of a factorisation only when required
* added option `keep_tree` to keep the factorisation tree when required
* clarification and expansion of readme and `example.py`
* explained usage of optional parameters `rectify_input=True` and `validate_input=True`
* explained usage of functions `get_gradient()` and `get_partial_derivative(i)`
* averaged runtime in speed tests



1.0.1 (2018-11-12)
__________________


* introducing option to only factor out single variables with the highest usage with the optional parameter ``univariate_factors=True``
* compute the number of operations needed by the horner factorisation by the length of its recipe (instead of traversing the full tree)
* instead of computing the value of scalar factors with exponent 1, just copy the values from the given x vector ("copy recipe")
* compile the initial value array at construction time



1.0.0 (2018-11-08)
__________________

* first stable release


0.0.1 (2018-10-05)
------------------

* birth of this package

