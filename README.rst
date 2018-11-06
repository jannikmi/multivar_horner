
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
This factorisation is being stored in a "Horner Tree". When evaluating the polynomial each subtree is being evaluated recursively.



Also see:
`GitHub <https://github.com/MrMinimal64/multivar_horner>`__,
`PyPI <https://pypi.python.org/pypi/multivar_horner/>`__


Dependencies
============

(``python3``),
``numpy``


Installation
============


Installation with pip:

::

    pip install multivar_horner





Usage
=====

Check code in ``example.py``:

TODO


Contact
=======

Most certainly there is stuff I missed, things I could have optimized even further or explained more clearly, etc. I would be really glad to get some feedback on my code.

If you encounter any bugs, have suggestions, criticism, etc.
feel free to **open an Issue**, **add a Pull Requests** on Git or ...

contact me: *[python] {*-at-*} [michelfe] {-*dot*-} [it]*



License
=======

``multivar_horner`` is distributed under the terms of the MIT license
(see LICENSE.txt).



References
==========

[1] CEBERIO, Martine; KREINOVICH, Vladik. `Greedy Algorithms for Optimizing Multivariate Horner Schemes <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.330.7430&rep=rep1&type=pdf>`__. ACM SIGSAM Bulletin, 2004, 38. Jg., Nr. 1, S. 8-15.



 
