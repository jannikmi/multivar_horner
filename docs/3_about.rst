
=====
About
=====

A python package implementing a multivariate `horner scheme ("Horner's method", "Horner's rule") <https://en.wikipedia.org/wiki/Horner%27s_method>`__  for efficiently evaluating multivariate polynomials.

A polynomial in canonical form (or normal form) is being factorised according to the greedy heuristic described in [1] with some additional computational tweaks.
The resulting Horner factorisation requires less operations for evaluation and is being computed by growing a "Horner Factorisation Tree".
When the polynomial is fully factorized (= all leaves cannot be factorised any more), a computational "recipe" for evaluating the polynomial is being compiled.
This "recipe" (stored internally as numpy arrays) enables fast evaluation with minimal memory requirement, because of the lack of additional overhead of recursive function calls (traversing the tree) and functions precompiled by ``numba`` operating on numpy arrays.

All factors in use in the factorisation are being computed only once (-> reusing computed values) to save computations.

**Pros:**
 * near to minimal representation of a multivariate polynomial (in the sense of memory and time complexity of the evaluation)
 * less roundoff errors[3], [4]
 * lower error propagation, because of fewer operations [1, Ch. 5]


**Cons:**
 * increased initial computational requirements and memory to find and then store the factorisation


For an exact evaluation of the impact of computing Horner factorisations see the benchmarks below.

.. TODO link


It is also possible to search for an optimal Horner factorisation
.. TODO link (cf. section "Optimal Horner Factorisation")

and to represent polynomials in canonical form
.. TODO link



Also see:
`GitHub <https://github.com/MrMinimal64/multivar_horner>`__,
`PyPI <https://pypi.python.org/pypi/multivar_horner/>`__


License
-------

``multivar_horner`` is distributed under the terms of the MIT license
(see `LICENSE <https://github.com/MrMinimal64/multivar_horner/blob/master/LICENSE>`__).





.. _benchmarks:
Benchmarks
==========


The benchmarks have been performed on a 15-inch MacBook Pro from 2017 with a 4 core 2,8 GHz Intel Core i7 processor, 16 GB 2133 MHz LPDDR3 RAM and macOS 10.13 High Sierra.
The software versions in use were multivar_horner 1.3.0, Python 3.7, numpy 1.16.3 and numba 0.40.1
Both evaluation algorithms (with and without Horner factorisation) make use of Numba just in time compiled functions.


TODO new

::

     parameters   |  setup time (/s)                        |  eval time (/s)                      |  # operations                        | lucrative after
    dim | max_deg | naive      | horner     | delta         | naive      | horner     | delta      | naive      | horner     | delta      |    # evals
    ================================================================================================================================================================
    1   | 1       | 2.76e-05   | 0.0001521  | 4.5 x more    | 5.901e-06  | 9.536e-06  | 0.6 x more | 4          | 2          | 1.0 x less | -
    1   | 2       | 2.348e-05  | 0.0001816  | 6.7 x more    | 2.735e-06  | 4.787e-06  | 0.8 x more | 6          | 3          | 1.0 x less | -
    1   | 3       | 2.342e-05  | 0.0002213  | 8.4 x more    | 2.944e-06  | 4.508e-06  | 0.5 x more | 8          | 4          | 1.0 x less | -
    1   | 4       | 2.217e-05  | 0.00026    | 11 x more     | 2.901e-06  | 5.934e-06  | 1.0 x more | 9          | 6          | 0.5 x less | -
    1   | 5       | 2.335e-05  | 0.0002598  | 10 x more     | 2.953e-06  | 5.87e-06   | 1.0 x more | 10         | 6          | 0.7 x less | -
    1   | 6       | 2.245e-05  | 0.0003294  | 14 x more     | 2.903e-06  | 4.393e-06  | 0.5 x more | 12         | 8          | 0.5 x less | -
    1   | 7       | 4.895e-05  | 0.0007023  | 13 x more     | 3.306e-06  | 1.101e-05  | 2.3 x more | 14         | 9          | 0.6 x less | -
    1   | 8       | 4.522e-05  | 0.0006728  | 14 x more     | 4.798e-06  | 5.245e-06  | 0.1 x more | 16         | 11         | 0.5 x less | -
    1   | 9       | 2.338e-05  | 0.0003935  | 16 x more     | 3.365e-06  | 4.297e-06  | 0.3 x more | 16         | 11         | 0.5 x less | -
    1   | 10      | 2.2e-05    | 0.0004315  | 19 x more     | 3.388e-06  | 4.664e-06  | 0.4 x more | 18         | 12         | 0.5 x less | -

    2   | 1       | 2.438e-05  | 0.0003116  | 12 x more     | 2.607e-06  | 6.293e-06  | 1.4 x more | 12         | 4          | 2.0 x less | -
    2   | 2       | 3.303e-05  | 0.0004885  | 14 x more     | 3.785e-06  | 4.919e-06  | 0.3 x more | 22         | 9          | 1.4 x less | -
    2   | 3       | 2.478e-05  | 0.0008237  | 32 x more     | 3.917e-06  | 5.146e-06  | 0.3 x more | 44         | 18         | 1.4 x less | -
    2   | 4       | 2.398e-05  | 0.001234   | 50 x more     | 4.741e-06  | 5.711e-06  | 0.2 x more | 62         | 26         | 1.4 x less | -
    2   | 5       | 2.522e-05  | 0.001719   | 67 x more     | 5.867e-06  | 5.112e-06  | 0.1 x less | 99         | 42         | 1.4 x less | 2243
    2   | 6       | 2.614e-05  | 0.002373   | 90 x more     | 7.228e-06  | 4.907e-06  | 0.5 x less | 131        | 56         | 1.3 x less | 1011
    2   | 7       | 2.649e-05  | 0.003193   | 120 x more    | 7.689e-06  | 5.068e-06  | 0.5 x less | 147        | 64         | 1.3 x less | 1208
    2   | 8       | 2.752e-05  | 0.003799   | 137 x more    | 1.073e-05  | 4.888e-06  | 1.2 x less | 215        | 92         | 1.3 x less | 645
    2   | 9       | 2.65e-05   | 0.004327   | 162 x more    | 1.089e-05  | 4.968e-06  | 1.2 x less | 254        | 109        | 1.3 x less | 726
    2   | 10      | 2.532e-05  | 0.005326   | 209 x more    | 1.355e-05  | 6.891e-06  | 1.0 x less | 309        | 132        | 1.3 x less | 796

    3   | 1       | 2.425e-05  | 0.0005184  | 20 x more     | 5.277e-06  | 4.398e-06  | 0.2 x less | 31         | 8          | 2.9 x less | 562
    3   | 2       | 2.346e-05  | 0.001594   | 67 x more     | 4.739e-06  | 7.196e-06  | 0.5 x more | 100        | 29         | 2.4 x less | -
    3   | 3       | 3.736e-05  | 0.003531   | 94 x more     | 9.475e-06  | 4.939e-06  | 0.9 x less | 224        | 69         | 2.2 x less | 770
    3   | 4       | 2.577e-05  | 0.006183   | 239 x more    | 1.444e-05  | 5.339e-06  | 1.7 x less | 400        | 124        | 2.2 x less | 677
    3   | 5       | 2.844e-05  | 0.01116    | 392 x more    | 2.203e-05  | 6.742e-06  | 2.3 x less | 737        | 226        | 2.3 x less | 728
    3   | 6       | 3.285e-05  | 0.01924    | 584 x more    | 3.914e-05  | 7.347e-06  | 4.3 x less | 1221       | 371        | 2.3 x less | 604
    3   | 7       | 3.613e-05  | 0.02789    | 771 x more    | 5.323e-05  | 8.102e-06  | 5.6 x less | 1775       | 540        | 2.3 x less | 617
    3   | 8       | 4.087e-05  | 0.03947    | 965 x more    | 7.538e-05  | 1.119e-05  | 5.7 x less | 2593       | 783        | 2.3 x less | 614
    3   | 9       | 4.976e-05  | 0.05609    | 1126 x more   | 0.000107   | 1.13e-05   | 8.5 x less | 3658       | 1096       | 2.3 x less | 586
    3   | 10      | 5.052e-05  | 0.0657     | 1299 x more   | 0.0001265  | 1.393e-05  | 8.1 x less | 4386       | 1322       | 2.3 x less | 583

    4   | 1       | 2.669e-05  | 0.001089   | 40 x more     | 4.045e-06  | 5.127e-06  | 0.3 x more | 74         | 16         | 3.6 x less | -
    4   | 2       | 2.747e-05  | 0.005081   | 184 x more    | 1.033e-05  | 5.422e-06  | 0.9 x less | 351        | 82         | 3.3 x less | 1029
    4   | 3       | 3.301e-05  | 0.01745    | 528 x more    | 3.088e-05  | 6.58e-06   | 3.7 x less | 1234       | 290        | 3.3 x less | 717
    4   | 4       | 3.863e-05  | 0.03762    | 973 x more    | 6.802e-05  | 9.488e-06  | 6.2 x less | 2675       | 638        | 3.2 x less | 642
    4   | 5       | 5.371e-05  | 0.09394    | 1748 x more   | 0.00017    | 1.422e-05  | 11 x less  | 6805       | 1582       | 3.3 x less | 603
    4   | 6       | 7.567e-05  | 0.1587     | 2096 x more   | 0.0002747  | 2.043e-05  | 12 x less  | 11400      | 2659       | 3.3 x less | 624
    4   | 7       | 0.0001071  | 0.2442     | 2279 x more   | 0.00042    | 2.96e-05   | 13 x less  | 17481      | 4104       | 3.3 x less | 625
    4   | 8       | 0.0001522  | 0.444      | 2917 x more   | 0.0006662  | 6.613e-05  | 9.1 x less | 27329      | 6413       | 3.3 x less | 740
    4   | 9       | 0.0004976  | 0.7057     | 1417 x more   | 0.001398   | 6.732e-05  | 20 x less  | 48369      | 11207      | 3.3 x less | 530
    4   | 10      | 0.0003185  | 0.9199     | 2887 x more   | 0.001583   | 9.237e-05  | 16 x less  | 64613      | 15038      | 3.3 x less | 617


TODO

Average evaluation time per polynomial using Horner factorisation

.. image:: ./plots/eval_time.png


Average evaluation time decrease per polynomial using Horner factorisation compared to using the naive matrix representation

.. image:: ./plots/eval_time_decrease.png


Average setup time per polynomial for computing the Horner factorisation

.. image:: ./plots/setup_time.png


Average setup time increase per polynomial for computing the Horner factorisation compared to using the naive matrix representation

.. image:: ./plots/setup_time_increase.png




Acknowledgements
----------------

Thanks to:

`Steve  <https://github.com/elcorto >`__ for valuable feedback and writing tests.
