Changelog
=========


TODOs

* build html docs and include with package:  "docs/_build/html/*"
* run speed and numerical tests with the new C evaluation method!
* Improve tests
* compare poly.num_ops of different factorisations. tests?
* num_ops currently will be 0 when caching is used (no factorisation will be computed)


POSSIBLE IMPROVEMENTS:

MultivarPoly:

- also make use of the concept of 'recipes' for efficiently evaluating the polynomial
    skipping the most unnecessary operations
- add option to skip this optimisation

HornerMultivarPoly:

- optimise factor evaluation (save instructions, 'factor factorisation'):
    a monomial factor consists of scalar factors and in turn some monomial factors consist of other monomial factors

-> the result of evaluating a factor can be reused for evaluating other factors containing it

-> find the optimal 'factorisation' of the factors themselves

-> set the factorisation_idxs of each factor in total_degree to link the evaluation appropriately

idea:
    choose  'Goedel IDs' as the monomial factor ids
    then the id of a monomial is the product of the ids of its scalar factors
    find the highest possible divisor among all factor ids
    (corresponds to the 'largest' factor included in the monomial)
    this leads to a minimal factorisation for evaluating the monomial values quickly

- add option to skip this optimisation to save build time

- optimise space requirement:
    after building a factorisation tree for the factors themselves,
    then use its structure to cleverly reuse storage space
    -> use compiler construction theory: minimal assembler register assignment, 'graph coloring'...

- optimise 'copy recipe': avoid copy operations for accessing values of x
    problem: inserting x into the value array causes operations as well and
        complicates address assigment and recipe compilation

-  when the polynomial does not depend on all variables, build a wrapper to maintain the same "interface"
    but internally reduce the dimensionality, this reduced the size of the numpy arrays -> speed, storage benefit

- the evaluation of subtrees is independent and could theoretically be done in parallel
    probably not worth the effort. more reasonable to just evaluate multiple polynomials in parallel

3.0.3 (2022-06-15)
__________________

* bugfix: packaging. now completely based on pyproject.toml (poetry)


3.0.2 (2022-06-14)
__________________

* bugfix: optional ``numba`` dependency. numba imports were not optional
* bugfix: create __cache__ dir if not exists
* minor documentation improvements
* bumping dependencies


3.0.1 (2021-12-04)
__________________

ATTENTION: major changes:

* introduced the default behavior of compiling the evaluation instructions in C code (C compiler required)
* the previous ``numpy+numba`` evaluation using "recipes" is the fallback option in case the C file could not be compiled
* as a consequence dropping ``numba`` as a required dependency
* added the "extra" ``numba`` to install on demand with: ``pip install multivar_horner[numba]``
* introduced custom polynomial hashing and comparison operations
* using hash to cache and reuse the instructions for evaluation (for both C and recipe instructions)
* introduced constructions argument ``store_c_instr`` (``HornerMultivarPolynomial``) to force the storage of evaluation code in C for later usage
* introduced constructions argument ``store_numpy_recipe`` (``HornerMultivarPolynomial``) to force the storage of the custom "recipe" data structure required for the evaluation using ``numpy`` and ``numba``
* introduced class ``HornerMultivarPolynomialOpt`` for optimal Horner Factorisations to separate code and simplify tests
* as a consequence dropped construction argument ``find_optimal`` of class ``HornerMultivarPolynomial``
* introduced constructions argument ``verbose`` to show the output of status print statements
* dropping official python3.6 support because ``numba`` did so (supporting Python3.7+)

internal:

* using poetry for dependency management
* using GitHub Actions for CI instead of travis


2.2.0 (2021-02-04)
__________________

ATTENTION: API changes:

* removed ``validate_input`` arguments. input will now always be validated (otherwise the numba jit compiled functions will fail with cryptic error messages)
* black code style
* pre-commit checks


2.1.1 (2020-10-01)
__________________

Post-JOSS paper review release:

* Changed the method of counting the amount of operations of the polynomial representations. Only the multiplications are being counted. Exponentiations count as (exponent-1) operations.
* the numerical tests compute the relative average error with an increased precision now


2.1.0 (2020-06-15)
__________________


ATTENTION: API changes:

* ``TypeError`` and ``ValueError`` are being raised instead of ``AssertionError`` in case of invalid input parameters with ``validate_input=True``
* added same parameters and behavior of ``rectify_input`` and ``validate_input`` in the ``.eval()`` function of polynomials


internal:

* Use ``np.asarray()`` instead of ``np.array()`` to avoid unnecessary copies
* more test cases for invalid input parameters



2.0.0 (2020-04-28)
__________________

* BUGFIX: factor evaluation optimisation caused errors in rare cases. this optimisation has been removed completely. every factor occurring in a factorisation is being evaluated independently now. this simplifies the factorisation process. the concept of "Goedel ID" (=unique encoding using prime numbers) is not required any more
* ATTENTION: changed polynomial degree class attribute names to comply with naming conventions of the scientific literature
* added __call__ method for evaluating a polynomial in a simplified notation ``v=p(x)``
* fixed dependencies to: ``numpy>=1.16``, ``numba>=0.48``
* clarified docstrings (using Google style)
* more verbose error messages during input verification
* split up ``requirements.txt`` (into basic dependencies and test dependencies)
* added sphinx documentation
* updated benchmark results

tests:

* added test for numerical stability
* added plotting features for evaluating the numerical stability
* added tests comparing functionality to 1D ``numpy`` polynomials
* added tests comparing functionality to naive polynomial evaluation
* added basic API functionality test

internal:

* added class ``AbstractPolynomial``
* added typing
* adjusted publishing routine
* testing multiple python versions
* using the specific tags of the supported python version for the build wheels
* removed ``example.py``


1.3.0 (2020-03-14)
__________________


* NEW FEATURE: changing coefficients on the fly with ``poly.change_coefficients(coeffs)``
* NEW DEPENDENCY: ``python3.6+`` (for using f'' format strings)
* the real valued coefficients are now included in the string representation of a factorised polynomial
* add contribution guidelines
* added instructions in readme, ``example.py``
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
__________________

* birth of this package
