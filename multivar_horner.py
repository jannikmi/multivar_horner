import numpy as np
from typing import List

import itertools


# TODO reference literature

# TODO calculate number of operations. Horner factorisation and vanilla


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


def get_goedel_id_of(prime_idx, exponent, prime_array):
    # return the unique ID of any monomial
    return prime_array[prime_idx] ** exponent


class AbstractFactor(object):
    __slots__ = ['monomial_id', 'value_idx']

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def eval(self, factor_values):
        # the value of all used factors has already been computed
        return factor_values[self.value_idx]

    def compute(self, x, factor_values):
        raise NotImplementedError()


class ScalarFactor(AbstractFactor):
    # a monomial with just one variable: x_i^n
    __slots__ = ['dimension', 'exponent']

    def __init__(self, factor_dimension, factor_exponent, monomial_id):
        self.dimension = factor_dimension
        self.exponent = factor_exponent
        self.monomial_id = monomial_id

    def __str__(self):
        # variable numbering starts with 1: x_1, x_2, ...
        return 'x_{}^{}'.format(self.dimension + 1, self.exponent)

    def __repr__(self):
        return self.__str__()

    def compute(self, x, factor_values):
        factor_values[self.value_idx] = x[self.dimension] ** self.exponent

    def num_ops(self):
        # count the number of operations done during compute (during eval() only looks up the computed value)
        return 1


class MonomialFactor(AbstractFactor):
    # a monomial consisting of a product of scalar factors: x_i^j * x_k^l * ...
    __slots__ = ['scalar_factors', 'factorisation_idxs']

    def __init__(self, scalar_factors):
        self.scalar_factors = scalar_factors

        # the id of a monomial is a Goedelian number corresponding to their exponent vector
        # this is a unique id for every monomial
        # the id of a monomial the product of the ids of its scalar factors
        self.monomial_id = 1
        for f in self.scalar_factors:
            self.monomial_id *= f.monomial_id

        if len(self.scalar_factors) == 1:
            raise ValueError('There is only one factor. Use ScalarFactor instead')

        if len(self.scalar_factors) == 0:
            raise ValueError()

    def __str__(self):
        return ' '.join([f.__str__() for f in self.scalar_factors])

    def __repr__(self):
        return self.__str__()

    def compute(self, x, factor_values):

        # IMPORTANT: compute() of all the sub factors has had to be called before!
        value = factor_values[self.factorisation_idxs[0]]
        for idx in self.factorisation_idxs[1:]:
            value *= factor_values[idx]

        factor_values[self.value_idx] = value

    def num_ops(self):
        # count the number of operations done during compute (during eval() only looks up the computed value)
        return len(self.factorisation_idxs)


class HornerTree(object):
    __slots__ = ['dim', 'order', 'max_degree', 'coefficient', 'factors', 'sub_trees', 'factors_n_subtrees']

    # # factors are the monomials which have been factored out
    # #   from the polynomial represented by the corresponding sub tree
    # factors: List[int] = []
    # # list of horner trees
    # sub_trees = []

    def __init__(self, coefficients, exponents, prime_array, unique_factor_id_list, unique_factors):

        self.dim = exponents.shape[1]
        self.order = np.sum(exponents, axis=0).max()
        self.max_degree = exponents.max()

        self.factors = []
        # self.factor_exps = []
        self.sub_trees = []

        # self.prime_array = get_prime_array(self.dim)

        def add_factor(factor_properties):
            # factor_dimensions, factor_exponents = factor_properties

            scalar_factors = []
            monomial_id = 1
            for d, e in zip(*factor_properties):
                scalar_id = get_goedel_id_of(d, e, prime_array)
                try:
                    scalar_factors.append(unique_factors[unique_factor_id_list.index(scalar_id)])
                except ValueError:
                    unique_factor_id_list.append(scalar_id)
                    scalar_factor = ScalarFactor(d, e, scalar_id)
                    scalar_factors.append(scalar_factor)
                    unique_factors.append(scalar_factor)

                monomial_id *= scalar_id

            if len(scalar_factors) == 0:
                raise ValueError()

            if len(scalar_factors) == 1:
                # this factor only depends on one variable = scalar factor
                self.factors.append(scalar_factors[0])
            else:
                try:
                    self.factors.append(unique_factors[unique_factor_id_list.index(monomial_id)])
                except ValueError:
                    unique_factor_id_list.append(monomial_id)
                    factor = MonomialFactor(scalar_factors)
                    self.factors.append(factor)
                    unique_factors.append(factor)

        def add_subtree(remaining_coefficients, remaining_exponents, factorized_rows, factor_properties):
            # print(factor_properties)

            add_factor(factor_properties)
            non_factorized_rows = [r for r in range(remaining_exponents.shape[0]) if r not in factor_rows]

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

            # the factor has to be deducted from the exponents of the sub tree
            for dim, exp in zip(*factor_properties):
                subtree_exponents[:, dim] -= exp

            self.sub_trees.append(
                HornerTree(subtree_coefficients, subtree_exponents, prime_array, unique_factor_id_list, unique_factors))

            return remaining_coefficients, remaining_exponents

        # determine which coefficient the polynomial represented by the current root node has
        # = the coefficient with a zero exponent vector
        inactive_exponent_rows = np.all(exponents == 0, axis=1)
        # DEBUG:
        if np.sum(inactive_exponent_rows) > 1:
            raise ValueError('more than one empty monomial:', exponents)

        if np.sum(inactive_exponent_rows) == 0:
            # avoid creating nodes without a coefficient! saves adding '0.0' at this node when evaluating

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

        # find a horner factorisation for the given polynomial
        # the tree must be sub divided until no monomials remain
        while len(remaining_coefficients) > 0:

            # greedy heuristic:
            # factor out the monomials which appears in the most terms
            # factor out the biggest factor possible (as many variables with highest degree possible)
            # NOTE: optimality is not guaranteed with this approach
            # <-> there may be a horner tree (factorisation) which can be evaluated with less operations
            # There is no known method for selecting an optimal factorisation order
            # TODO find algorithm to parse optimal tree (no procedure known for this!)

            active_exponents = remaining_exponents >= 1

            # count how many times each scalar factor is being used
            active_rows, active_dimensions = np.where(active_exponents)
            usage_statistic = np.zeros((self.dim, int(self.max_degree + 1)), dtype=np.int)
            scalar_factors2rows = {}
            for row, dim in zip(active_rows, active_dimensions):
                # do not count when exponent is 0 (= x_d^0)
                exp = remaining_exponents[row, dim]
                usage_statistic[dim, exp] += 1
                scalar_factors2rows.setdefault((dim, exp), []).append(row)

            max_usage_count = usage_statistic.max()
            max_usage_scalar_factors = np.where(usage_statistic == max_usage_count)
            #  max_usage_dimensions, max_usage_exponents = scalar_factors

            if len(max_usage_scalar_factors[0]) == 1:
                # there is only one scalar factor with maximal usage
                maximal_factor_properties = max_usage_scalar_factors[0], max_usage_scalar_factors[1]
                factor_rows = scalar_factors2rows[maximal_factor_properties[0][0], maximal_factor_properties[1][0]]
            else:
                rows2max_scalar_factors = {}
                # take the maximal amount of scalar factors with an occurrence in the same monomials
                for dim, exp in zip(*max_usage_scalar_factors):
                    factor_rows = tuple(scalar_factors2rows[(dim, exp)])
                    rows2max_scalar_factors.setdefault(factor_rows, []).append((dim, exp))

                # those scalar factors combined are the maximal factor
                factor_rows, maximal_factor_tuples = \
                    sorted(rows2max_scalar_factors.items(), key=lambda t: len(t[1]))[0]
                factor_rows = list(factor_rows)
                maximal_factor_properties = list(zip(*maximal_factor_tuples))
                # maximal_factor_properties

            remaining_coefficients, remaining_exponents = add_subtree(remaining_coefficients, remaining_exponents,
                                                                      factor_rows, maximal_factor_properties)

        self.factors_n_subtrees = zip(self.factors, self.sub_trees)

    def __str__(self, indent_lvl=1):
        if self.coefficient == 0.0:
            s = ''
        else:
            s = str(self.coefficient)
            if len(self.sub_trees) > 0:
                s += ' + '

        # '\t' * indent_lvl
        s += ' + '.join(['{} [ {} ]'.format(factor.__str__(), subtree.__str__())
                         for factor, subtree in self.factors_n_subtrees])

        return s

    def __repr__(self):
        return self.__str__()

    def eval(self, factor_values):

        # TODO avoid for loop
        # TODO numba precompilation!?
        # TODO clever binary format or optimized data structure representing tree
        # TODO is recursion problematic for huge polynomials?! (stack size limitations...)

        # p(x) = c_0 + c_1 p_1(x) + c_2 p_2(x) + ...
        # TODO avoid addition with 0.0 when evaluating
        out = self.coefficient

        for factor, sub_tree in self.factors_n_subtrees:
            # eval all sub trees
            out += factor.eval(factor_values) * sub_tree.eval(factor_values)
        return out

    def num_ops(self):
        # count the number of operations done during eval()
        num_ops = 0
        for factor, subtree in self.factors_n_subtrees:
            num_ops += 2 + subtree.num_ops()
        return num_ops


class MultivarPolynomial(object):
    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['coefficients', 'exponents', 'dim', 'order', 'max_degree', 'unused_variables']
    '''
    dim: the dimensionality of the polynomial.
    order: (for a multivariate polynomial) maximum sum of exponents in any of its monomials
    max_degree: the largest exponent in any of its monomials
        NOTE: the polynomial actually needs not to depend on all dimensions
    unused_variables: the dimensions the polynomial does not depend on
    '''

    def __init__(self, coefficients, exponents):
        """
        naive representation of a multivariate polynomial without any horner factorisation


        :param coefficients:
        :param exponents: ordering does not matter, but have to be unique!
        """

        self.coefficients = np.atleast_1d(np.array(coefficients, dtype=np.float)).reshape(-1, 1)
        # coefficients must be given as a column vector
        assert self.coefficients.shape[1] == 1 and len(self.coefficients.shape) == 2

        # TODO
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
        self.order = np.sum(self.exponents, axis=0).max()
        self.max_degree = self.exponents.max()

        self.unused_variables = np.where(np.any(self.exponents, axis=1))[0]

    def __str__(self):
        s = '[{}] p(x) = '.format(self.num_ops())
        monomials = []
        for i, exp_vect in enumerate(self.exponents):
            monomial = [str(self.coefficients[i, 0]), ]
            for dim, exp in enumerate(exp_vect):
                if exp > 0:
                    monomial.append('x_{}^{}'.format(dim + 1, exp))
            monomials.append(' '.join(monomial))

        s += ' + '.join(monomials)
        return s

    def __repr__(self):
        return self.__str__()

    def num_ops(self):
        # count the number of operations done when evaluating polynomial:
        y, x = self.exponents.shape
        # exponentiation: x*y
        # multiplication coefficient and scalar factors (monomials): x*y
        # sum: y-1
        return 2 * x * y + y - 1

    def eval(self, x):
        # TODO performance comparison, plot
        # TODO numba precompilation
        # TODO test
        x = np.array(x)
        assert x.shape[0] == self.dim
        return np.sum(self.coefficients.T * np.prod(np.power(x, self.exponents), axis=1), axis=1)

    def partial_derivative(self, i):
        """
        TODO test
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

        return self.__class__(new_coefficients, new_exponents)

    def gradient(self):
        """
        :return: the list of all partial derivatives
        """
        return [self.partial_derivative(i) for i in range(self.dim)]


class HornerMultivarPolynomial(MultivarPolynomial):
    # __slots__ declared in parents are available in child classes. However, child subclasses will get a __dict__
    # and __weakref__ unless they also define __slots__ (which should only contain names of any additional slots).
    __slots__ = ['prime_array', 'horner_tree', 'unique_factor_id_list', 'unique_factors', 'factor_values']

    def __init__(self, coefficients, exponents):
        """
          a representation of a multivariate polynomial using horner factorisation to save operations during evaluation
          dimension: the amount of variable as input
          NOTE: the polygon actually needs not to depend on all dimensions
          order: (for a multivariate polynomial) maximum sum of exponents in any of its monomials
          degree: the largest exponent in any of its monomials
          :param coefficients:
          :param exponents: ordering does not matter, but have to be unique!
          dim: the dimensionality of the polynomial.
        """

        super(HornerMultivarPolynomial, self).__init__(coefficients, exponents)

        # factor id (goedelian number, unique)
        # TODO sorted after ID?!
        # factor idx
        # factor exponents?! (=input for building tree)
        # factor values computed fully once at every evaluation query
        # factor tree, nodes are labeled with idx

        # pass this dict as argument to all subtrees
        # self.monomial_id2val = {}

        self.prime_array = get_prime_array(self.dim)

        # store all unique factors of the horner factorisation
        # NOTE: do not create all scalar factors with exponent 1 (they might not be required!)

        self.unique_factor_id_list = []
        self.unique_factors = []

        # factorize the polynomial once and store the factorisation as a tree
        self.horner_tree = HornerTree(self.coefficients, self.exponents, self.prime_array,
                                      self.unique_factor_id_list, self.unique_factors)

        # factor list is now filled with the unique factors

        # during evaluation of a polynomial the values of all the unique factors are needed at least once
        # -> compute the values of all factors (monomials) once and store them
        # store a pointer to the computed value for every unique factor
        # save operations for the evaluation -> find factorisation of all factors = again a factorisation tree
        # sort and factorize the monomials to quickly evaluate them once during a query
        self.link_monomials()
        self.factor_values = np.empty(shape=len(self.unique_factor_id_list), dtype=np.float)

    def __str__(self):
        return '[{}] p(x) = '.format(self.num_ops()) + self.horner_tree.__str__()

    def num_ops(self):
        # count the number of operations done when computing all factors
        num_ops = 0
        for f in self.unique_factors:
            num_ops += f.num_ops()

        # ... and when evaluating the horner factorisation
        num_ops += self.horner_tree.num_ops()
        return num_ops

    def link_monomials(self):
        """
        find the optimal factorisation of the unique factors themselves
        since the monomial ids are products of the ids of their scalar factors
        check for the highest possible divisor among all factor ids
        this leads to a minimal factorisation for quick evaluation of the monomial values

        all scalar factors need not be factorised any more (even with an exponent > 1),
        because it is cheapest to just compute their value every time
        (one operation, more efficient than value look up)
        for performance it is best to not factorize x_i^n, but just compute it in one go:
        -> find all unique scalar factors, add them to the tree as factors
        then find factorisation for all non scalar factors based on those scalar factors
        = link the factors

        :return:
        """

        # sort after their id
        self.unique_factor_id_list = list(sorted(self.unique_factor_id_list))
        self.unique_factors = list(sorted(self.unique_factors, key=lambda f: f.monomial_id))
        # property of the list: the scalar factors of each monomial are stored in front of it
        print(self.unique_factors)
        print(self.unique_factor_id_list)

        # IMPORTANT: properly set the indices of the values for each factor
        for idx, f in enumerate(self.unique_factors):
            f.value_idx = idx

        # start at the last factor (highest id)
        pointer1 = len(self.unique_factors) - 1

        # the smallest factor has no factorisation (stop at pointer=1)
        while pointer1 > 0:
            candidate = self.unique_factors[pointer1]
            pointer1 -= 1

            if type(candidate) is not MonomialFactor:
                # scalar factors have no factorisation
                continue

            remaining_id = candidate.monomial_id
            # store the indices of the factors that divide the id
            factorisation_idxs = []

            pointer2 = pointer1 - 1
            # find the factors with the highest id which are a factor of the current monomial
            while 1:
                monomial_id2 = self.unique_factors[pointer2].monomial_id

                if monomial_id2 == remaining_id:
                    factorisation_idxs.append(pointer2)
                    break

                quotient, remainder = divmod(remaining_id, monomial_id2)
                if remainder == 0:
                    # this factor is a factor of the monomial
                    factorisation_idxs.append(pointer2)
                    # reduce the id
                    remaining_id = quotient

                pointer2 -= 1

                if pointer2 < 0:
                    raise ValueError

            print('found factorisation:', candidate.monomial_id, factorisation_idxs)
            # store the indices of the factors that factorize the monomial
            #   to quickly access their values when computing the value of the monomial
            candidate.factorisation_idxs = factorisation_idxs

    def eval(self, x):
        # TODO use only numpy for eval. for numba precompilation
        # the values of the scalar monomials with exponent 1 (x_i^1) can already be stored

        x = np.array(x)
        assert x.shape[0] == self.dim

        # IMPORTANT: reset the value lookup for every query
        # factors later in the list require previously computed values
        for f in self.unique_factors:
            f.compute(x, self.factor_values)

        # after evaluating the whole tree,
        # all the factors existing in the factorised polynomial have been evaluated (and stored!)
        # so when evaluating the actual horner factorisation tree of the polynomial
        # the values of all factors can be looked up at the corresponding idx in the value list
        # the evaluation of the polynomial only requires the computed values of all factors (not x)
        return self.horner_tree.eval(self.factor_values)


# TODO function for checking conditions of polynomial
# no duplicate exponent vectors!
# warning if polynomial is independent of certain dimensions

# TODO multivariate newton raphson method
# TODO mention in readme


if __name__ == '__main__':
    inp, expected_out = (
        ([1.0],
         [1],
         [1.0]),
        1.0
    )  # p(x)

    coeff, exp, x = inp
    x = np.array(x).T
    poly = MultivarPolynomial(coeff, exp)
    p_x = poly.eval(x)
    print(p_x)
