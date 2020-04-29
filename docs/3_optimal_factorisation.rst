.. _optimal:

=============================
Optimal Horner Factorisations
=============================


See :ref:`this code <optimal_usage>` for an example usage.


Instead of using a heuristic to choose the next factor one can allow a search over all possible (meaningful) factorisations in order to arrive at a minimal Horner factorisation.
The amount of possible factorisations however is increasing exponentially with the degree of a polynomial and its amount of monomials.
One possibility to avoid computing each factorisation is to employ a version of A*-search:cite:`hart1968formal` adapted for factorisation trees:
• Initialise a set of all meaningful possible first level Newton factorisations
• Rank all factorisation according to a lower bound (“heuristic”) of their lowest possible amount of operations
• Iteratively factorise the most promising factorisation and update the heuristic
• Stop when the most promising factorisation is fully factorised

This approach is guaranteed to yield a minimal Horner factorisation, but its performance highly depends on the heuristic in use: Irrelevant factorisations are only being ignored if the heuristic is not too optimistic in estimating the amount of operations. On the other hand the heuristic must be easy to compute, because it would otherwise be computationally cheaper to just try all different factorisations.
Even though it missing to cover exponentiations, the branch-and-bound method suggested in :cite:`kojima2008efficient` (ch. 3.1) is almost identical to this procedure.


Even with a good heuristic this method is only traceable for small polynomials because of its increased resource requirements.
Since experiments show that factorisations obtained by choosing one factorisation according to a heuristic have the same or only a slightly higher amount of included operations:cite:`kojima2008efficient` (ch. 7), the computational effort of this approach is not justifiable in most cases.
A use case however is to compute and store a minimal representation of a polynomial in advance if possible.

**NOTES:**

* currently this approach seems to try all possible factorisations in most cases, because the heuristic in use is too optimistic (= brute force, improvements needed)
* this requires MUCH more memory and computing time than just trying one factorisation (the number of possible factorisations is growing exponentially with the size of the polynomial!).
* in the first test runs the results seemed to be identical (in terms of #ops) with the vanilla approach of just trying one factorisation!
* in contrast to univariate polynomials there are possibly many optimal Horner factorisations of a multivariate polynomial. one could easily adapt this approach to find all optimal Horner factorisations
* even an optimal Horner factorisation must not be the globally minimal representation of a polynomial. there are possibly better types of factorisations and techniques: e.g. "algebraic factorisation", "common subexpression elimination"

