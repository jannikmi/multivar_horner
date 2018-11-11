Changelog
=========

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

