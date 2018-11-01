import numpy as np
from typing import List

import itertools


# TODO reference literature

# https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
def erat2():
    D = {}
    yield 2
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        p = D.pop(q, None)
        if p is None:
            D[q * q] = q
            yield q
        else:
            x = p + q
            while x in D or not (x & 1):
                x += p
            D[x] = p


def get_prime_array(length):
    return np.array(list(itertools.islice(erat2(), length)), dtype=np.uint)


def get_goedel_id_of(prime_id, exponent, prime_array):
    # return the unique ID of any monomial
    return prime_array[prime_id] ** exponent


class AbstractFactor(object):
    def eval(self, x):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


class ScalarFactor(AbstractFactor):
    # monomial with just one variable: x_i^n
    __slots__ = ['dimension', 'exponent', 'monomial_id', 'monomial_id2val']

    def __init__(self, factor_dimension, factor_exponent, monomial_id2val):
        self.dimension = factor_dimension
        self.exponent = factor_exponent
        self.monomial_id.append(get_goedel_id_of(factor_dimension, factor_exponent))
        self.monomial_id2val = monomial_id2val

    def eval(self, x):
        try:
            return self.monomial_id2val[self.monomial_id]
        except KeyError:
            # the value of this monomial has not been computed yet
            # compute the value of any monomial only once
            val = x[self.dimension] ** self.exponent
            self.monomial_id2val[self.monomial_id] = val
        return val

    def __str__(self):
        # variable numbering starts with 1: x_1, x_2, ...
        return 'x_{}^{}'.format(self.dimension + 1, self.exponent)

    def __repr__(self):
        return self.__str__()


class MonomialFactor(AbstractFactor):
    # monomial: x_i^j * x_k^l * ...
    __slots__ = ['factors', 'monomial_id', 'monomial_id2val']

    def __init__(self, exponent_vector, monomial_id2val):
        self.monomial_id2val = monomial_id2val
        self.factors = []
        self.monomial_id = 1
        for dimension, exponent in enumerate(exponent_vector):
            if exponent > 0:
                scalar_factor = ScalarFactor(dimension, exponent, monomial_id2val)
                self.factors.append(scalar_factor)
                self.monomial_id *= scalar_factor.monomial_id

    def eval(self, x):
        try:
            return self.monomial_id2val[self.monomial_id]
        except KeyError:
            pass

        # the value of this monomial has not been computed yet
        # compute the value of any monomial only once
        factor_current = self.factors[0]
        monomial_id_current = factor_current.monomial_id
        # try to look up the value of the current monomial
        value_current = factor_current.eval(x)

        # TODO for big monomials: instead of starting with scalar monomial, start with full monomial and factorize
        # until one finds a monomial with computed value
        for factor_current in self.factors[1:0]:
            monomial_id_current *= factor_current.monomial_id
            # try to look up the value of the current monomials
            try:
                value_current = self.monomial_id2val[self.monomial_id]
            except KeyError:
                # value has to be computed first
                value_current *= factor_current.eval(x)
                self.monomial_id2val[monomial_id_current] = value_current

        return value_current

    def __str__(self):
        return ' '.join([f.__str__() for f in self.factors])

    def __repr__(self):
        return self.__str__()


class HornerTree(object):
    __slots__ = ['dim', 'order', 'max_degree', 'coefficient', 'factors', 'sub_trees', 'monomial_id2val']

    # coefficient = 0.0
    # # factors are the monomials which have been factored out
    # #   from the polynomial represented by the corresponding sub tree
    # factors: List[int] = []
    # # list of horner trees
    # sub_trees = []

    def __init__(self, coefficients, exponents, prime_array, monomial_id2val):

        self.dim = exponents.shape[1]
        self.order = np.sum(exponents, axis=0).max()
        self.max_degree = exponents.max()

        # self.prime_array = get_prime_array(self.dim)

        # find a horner factorisation for the given polynomial

        def add_scalar_factor(factor_dimension, factor_exponent):
            self.factors.append(ScalarFactor(factor_dimension, factor_exponent, monomial_id2val))

        def add_factor(exponent_vector):
            self.factors.append(MonomialFactor(exponent_vector, monomial_id2val))

        def add_subtree(coefficients, exponents):
            self.sub_trees.append(HornerTree(coefficients, exponents, prime_array, monomial_id2val))

        def add_single_monomial_subtrees(coeffs, exponets):
            # all remaining monomials do not have common variables
            # add one subtree for every monomial
            for c, exp_vect in zip(coeffs, exponets):
                # add the monomial as a factor
                exp_vect = exp_vect.reshape(1, -1)
                add_factor(exp_vect)

                # add a subtree with just one coefficient
                c = c.reshape(1, 1)  # make column vector with a single entry
                exp_array = np.zeros(1, self.dim)  # row matrix with a single empty exponent vector entry
                add_subtree(c, exp_array)

        self.factors = []
        # self.factor_exps = []
        self.sub_trees = []

        # store link to the value lookup dict of the (parent) polynomial
        self.monomial_id2val = monomial_id2val

        inactive_exponent_rows = np.all(exponents == 0, axis=1)
        # # DEBUG:
        if np.sum(inactive_exponent_rows) > 1:
            raise ValueError('more than one empty monomial:', exponents)

        # print(exponents)
        # print(inactive_exponent_rows)
        # if len(inactive_exponents.shape) == 1:
        # else:

        if np.sum(inactive_exponent_rows) == 0:
            # TODO avoid addition with 0.0 when evaluating
            # TODO avoid adding nodes with no coefficient (= unnecessary factorisation!)
            self.coefficient = 0.0
            remaining_coefficients = np.copy(coefficients)
            remaining_exponents = np.copy(exponents)

        else:
            # there is one empty monomial
            # the coefficient corresponding to this monomial has to be added
            #   to the polynomial at this node in the horner tree
            empty_monomial_idx = np.where(inactive_exponent_rows)[0]
            self.coefficient = coefficients[empty_monomial_idx[0], 0]
            remaining_monomial_idxs = np.where(~inactive_exponent_rows)[0]
            remaining_coefficients = coefficients[remaining_monomial_idxs]
            remaining_exponents = exponents[remaining_monomial_idxs, :]

        # print('node:', self.coefficient, '\n\n')

        # print(coefficients)
        # print(remaining_coefficients)
        # print(remaining_coefficients.shape)
        # print(len(remaining_coefficients))

        # the tree must be sub divided until there are no remaining monomials
        while len(remaining_coefficients) > 0:
            # greedy heuristic:
            # always factor out the variable (x_i) which appears in the most terms
            # optimality is not guaranteed with this approach
            # there may be horner trees (factorisations) which can be evaluated with less operations
            # There is no known method for selecting an optimal factorisation order
            # TODO find algorithm to parse optimal tree (no procedure known for this!)
            active_exponents = remaining_exponents >= 1

            dimension_usage_count = np.sum(active_exponents, axis=0)
            max_usage_count = dimension_usage_count.max()
            if max_usage_count == 1:
                # the remaining monomials do not share any variables
                # no more useful factorisation can be made.
                # -> add a separate tree for all remaining monomials
                add_single_monomial_subtrees(remaining_exponents)
                break

            # TODO avoid building list
            factor_dimension = dimension_usage_count.tolist().index(max_usage_count)
            active_factor_rows = active_exponents[:, factor_dimension]
            factorized_rows = np.where(active_factor_rows)[0]
            non_factorized_rows = np.where(~active_factor_rows)[0]

            # the factor should have the highest possible exponent
            factor_exponent = remaining_exponents[factorized_rows, factor_dimension].min()

            # factors are represented by a Goedelian number corresponding to their exponent vector
            # unique id
            add_scalar_factor(factor_dimension, factor_exponent)

            subtree_coefficients = remaining_coefficients[factorized_rows]
            remaining_coefficients = remaining_coefficients[non_factorized_rows]
            subtree_exponents = remaining_exponents[factorized_rows, :]
            remaining_exponents = remaining_exponents[non_factorized_rows, :]

            # DEBUG:
            assert remaining_coefficients.shape[0] == remaining_exponents.shape[0]
            assert subtree_coefficients.shape[0] == subtree_exponents.shape[0]
            assert remaining_coefficients.shape[1] == 1 and len(remaining_coefficients.shape) == 2
            assert subtree_coefficients.shape[1] == 1 and len(subtree_coefficients.shape) == 2
            assert remaining_exponents.shape[1] == exponents.shape[1] and len(remaining_exponents.shape) == 2
            assert subtree_exponents.shape[1] == exponents.shape[1] and len(subtree_exponents.shape) == 2

            # TODO check if next factorisation would only use the same monomials
            # -> do not create nodes without a coefficient! saves adding 0.0 and speedup by lookup

            # the factor has to be deducted from the exponents of the sub tree
            subtree_exponents[:, factor_dimension] -= factor_exponent

            add_subtree(subtree_coefficients, subtree_exponents)

    def __str__(self, indent_lvl=1):
        if self.coefficient == 0.0:
            s = ''
        else:
            s = str(self.coefficient)
            if len(self.sub_trees) > 0:
                s += ' + '

        # '\t' * indent_lvl
        s += ' + '.join(['{} [ {} ]'.format(factor.__str__(), subtree.__str__())
                         for factor, subtree in zip(self.factors, self.sub_trees)])

        return s

    def __repr__(self):
        return self.__str__()

    def eval(self, x):

        # TODO avoid for loop
        # TODO numba precompilation!?
        # TODO clever binary format or optimized data structure representing tree
        # TODO is recursion problematic for huge polynomials?!

        # p(x) = c_0 + c_1 p_1(x) + c_2 p_2(x) + ...
        out = self.coefficient

        for factor_idx, (factor_id, sub_tree) in enumerate(zip(self.factor_ids, self.sub_trees)):
            # eval all sub trees
            out += get_value(factor_id, factor_idx) * sub_tree.eval(x, monomial_id2val)
        return out


class MultivarPolynomial(object):
    """
        exponents: ordering does not matter, but have to be unique!
        dim: the dimensionality of the polynomial.
            NOTE: the polygon actually needs not to depend on all dimensions
        order: (for a multivariate polynomial) maximum sum of exponents in any of its monomials
        degree: the largest exponent in any of its monomials
    """
    # prevent dynamic attribute assignment (-> safe memory)
    # __slots__ = ['quadrant', 'angle_measure', 'value']
    __slots__ = ['coefficients', 'exponents', 'dim', 'order', 'max_degree', 'horner_tree', 'prime_array',
                 'monomial_id2val']

    def __init__(self, coefficients, exponents):

        self.coefficients = np.atleast_1d(np.array(coefficients, dtype=np.float)).reshape(-1, 1)
        # coefficients must be given as a column vector
        assert self.coefficients.shape[1] == 1 and len(self.coefficients.shape) == 2

        if np.any(self.coefficients == 0.0):
            # ignore the entries with 0.0 coefficients
            non_zero_coeff_idxs = np.where(self.coefficients != 0.0)[0]
            self.coefficients = self.coefficients[non_zero_coeff_idxs, :]

            self.exponents = np.atleast_2d(np.array(exponents, dtype=np.uint)[non_zero_coeff_idxs, :])
        else:
            self.exponents = np.atleast_2d(np.array(exponents, dtype=np.uint))

        assert not np.any(self.exponents < 0)  # exponents must not be negative
        assert self.coefficients.shape[0] == self.exponents.shape[0]

        # DEBUG:
        assert not np.any(self.coefficients == 0.0)

        self.dim = self.exponents.shape[1]
        self.prime_array = get_prime_array(self.dim)
        self.order = np.sum(self.exponents, axis=0).max()
        self.max_degree = self.exponents.max()

        # value lookup for reusing already computed values of monomials
        # pass this dict as argument to all subtrees
        self.monomial_id2val = {}

        # factorize the polynomial once and store the factorisation as a tree
        # TODO option for suppressing horner tree build
        self.horner_tree = HornerTree(self.coefficients, self.exponents, self.prime_array, self.monomial_id2val)
        print(self.__str__())
        print('\n\n')

    def __str__(self):
        return self.horner_tree.__str__()

    def __repr__(self):
        return self.__str__()

    def eval(self, x):

        # IMPORTANT: reset the value lookup for every query
        # the values of the scalar monomials with exponent 1 (x_i^1) can already be stored
        self.monomial_id2val = dict(zip(self.prime_array, x))
        print(self.monomial_id2val)

        x = np.array(x)
        assert x.shape[0] == self.dim

        return self.horner_tree.eval(x)

    def partial_derivative(self, i):
        """
        f(x) = a x^b
        f'(x) = ab x^(b-1)
        :param i:
        :return: the partial derivative of this polynomial wrt. the i-th coordinate
        """

        # set all the coefficients not depending on the i-th coordinate to 0
        # this simply means not adding them to the list of coefficients
        active_rows = np.where(self.exponents[:, i] >= 1)[0]

        new_coefficients = self.coefficients[active_rows]
        new_exponents = self.exponents[active_rows, :]

        # DEBUG:
        assert new_coefficients.shape[0] == new_exponents.shape[0]
        assert new_coefficients.shape[1] == 1 and len(new_coefficients.shape) == 2
        assert new_exponents.shape[1] == self.dim and len(new_exponents.shape) == 2

        # multiply the coefficients with the exponent of the i-th coordinate
        new_coefficients *= new_exponents[:, i]

        # reduce the the exponent of the i-th coordinate by 1
        new_exponents[:, i] -= 1

        return MultivarPolynomial(new_coefficients, new_exponents)

    def gradient(self):
        """
        :return: the list of all partial derivatives
        """
        return [self.partial_derivative(i) for i in range(self.dim)]


# TODO function for checking conditions of polynomial
# no duplicate exponent vectors!
# warning if polynomial is independent of certain dimensions

# TODO multivariate newton raphson method
# TODO mention in readme


if __name__ == '__main__':
    inp, expected_out = (
        ([1.0],  # coefficients
         [1],  # exponents
         [0.0]),  # x
        0.0
    )  # p(x)

    coeff, exp, x = inp
    x = np.array(x).T
    poly = MultivarPolynomial(coeff, exp)
    p_x = poly.eval(x)
    print(p_x)
