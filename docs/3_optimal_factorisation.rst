.. _optimal:

=============================
Optimal Horner Factorisations
=============================

See :ref:`this code <optimal_usage>` for an example usage.


Instead of using a heuristic to choose the next factor one can allow a search over all possible (meaningful) factorisations in order to arrive at a minimal Horner factorisation.
The amount of possible factorisations however is increasing exponentially with the degree of a polynomial and its amount of monomials.
One possibility to avoid computing each factorisation is to employ a version of A*-search :cite:`hart1968formal` adapted for factorisation trees:
• Initialise a set of all meaningful possible first level Newton factorisations
• Rank all factorisation according to a lower bound (“heuristic”) of their lowest possible amount of operations
• Iteratively factorise the most promising factorisation and update the heuristic
• Stop when the most promising factorisation is fully factorised

This approach is guaranteed to yield a minimal Horner factorisation, but its performance highly depends on the heuristic in use: Irrelevant factorisations are only being ignored if the heuristic is not too optimistic in estimating the amount of operations. On the other hand the heuristic must be easy to compute, because it would otherwise be computationally cheaper to just try all different factorisations.
Even though it missing to cover exponentiations, the branch-and-bound method suggested in :cite:`kojima2008efficient` (ch. 3.1) is almost identical to this procedure.


Even with a good heuristic this method is only traceable for small polynomials because of its increased resource requirements.
Since experiments show that factorisations obtained by choosing one factorisation according to a heuristic have the same or only a slightly higher amount of included operations :cite:`kojima2008efficient` (ch. 7), the computational effort of this approach is not justifiable in most cases.
A use case however is to compute and store a minimal representation of a polynomial in advance if possible.

**NOTES:**

* for the small polynomial examples in the current tests, the results were identical (in terms of #ops) with the approach of just using the default heuristic = trying one factorisation (further analysis needed)!
* in some cases this approach currently is trying all possible factorisations, because the heuristic in use is too optimistic (= brute force, further analysis and improvements needed)
* this requires MUCH more computational resources than just trying one factorisation (the number of possible factorisations is growing exponentially with the size of the polynomial!).
* there are possibly many optimal Horner factorisations of a multivariate polynomial. one could easily adapt this approach to find all optimal Horner factorisations
* even an optimal Horner factorisation must not be the globally minimal representation of a polynomial. there are possibly better types of factorisations and techniques: e.g. "algebraic factorisation", "common subexpression elimination"
* there are multiple possible concepts of optimality (or minimality) of a polynomial
