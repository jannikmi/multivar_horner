import numpy as np

from .global_settings import DEBUG, ID_ADD, ID_MULT, UINT_DTYPE
from .helper_fcts import get_goedel_id_of
from .helpers_fcts_numba import (
    build_row_equality_matrix, compile_usage_rows, compile_usage_statistic, factor_out, index_of,
)


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
        self.value_idx = None  # initialize the idx with None to catch faulty evaluation tries

    def __str__(self):
        # variable numbering starts with 1: x_1, x_2, ...
        return 'x_{}^{}'.format(self.dimension + 1, self.exponent)

    def __repr__(self):
        return self.__str__()

    def num_ops(self):
        # count the number of instructions done during compute (during eval() only looks up the computed value)
        return 1

    def compute(self, x, factor_values):
        factor_values[self.value_idx] = x[self.dimension] ** self.exponent

    def get_recipe(self):
        # instruction encoding: target, source, exponent
        # values[target] = x[source] ** exponent
        return [(self.value_idx, self.dimension, self.exponent)]


class MonomialFactor(AbstractFactor):
    """
    factorisation_idxs: the indices of the values of all factors.
        cannot be set at construction time, because the list of all factors is sorted
        at the end of building the horner factorisation tree (when all factors are known)

    """
    # a monomial consisting of a product of scalar factors: x_i^j * x_k^l * ...
    __slots__ = ['scalar_factors', 'factorisation_idxs']

    def __init__(self, scalar_factors):
        self.scalar_factors = scalar_factors

        # the id of a monomial is the Goedel number of their exponent vector
        # https://en.wikipedia.org/wiki/G%C3%B6del_numbering
        # this is a unique id for every monomial
        # the id of a monomial the product of the ids of its scalar factors
        self.monomial_id = 1
        for f in self.scalar_factors:
            self.monomial_id *= f.monomial_id

        if len(self.scalar_factors) == 1:
            raise ValueError('There is only one factor. Use ScalarFactor instead')

        if len(self.scalar_factors) == 0:
            raise ValueError()

        self.value_idx = None  # initialize the idx with None to catch faulty evaluation tries

    def __str__(self):
        return ' '.join([f.__str__() for f in self.scalar_factors])

    def __repr__(self):
        return self.__str__()

    def num_ops(self):
        # count the number of instructions done during compute (during eval() only looks up the computed value)
        return len(self.factorisation_idxs) - 1

    def compute(self, x, factor_values):
        # IMPORTANT: compute() of all the sub factors has had to be called before!
        value = factor_values[self.factorisation_idxs[0]]
        for idx in self.factorisation_idxs[1:]:
            value *= factor_values[idx]

        factor_values[self.value_idx] = value

    def get_recipe(self):
        # target = source1 * source2
        # instruction encoding: target, source1, source2
        target, source1, source2 = self.value_idx, self.factorisation_idxs[0], self.factorisation_idxs[1]
        recipe = [(target, source1, source2)]

        source1 = target  # always take the previously computed value
        for source2 in self.factorisation_idxs[2:]:
            # and multiply it with the remaining factor values
            recipe += [(target, source1, source2)]

        return recipe


class HornerTree(object):
    """
    coefficient: the coefficient which has to be added at the root node of this horner tree during evaluation
    factors: the monomials which have been factored out
        from the polynomial represented by the corresponding sub tree
    sub_trees: list of factorised polynomials themselves represented by horner trees (recursion)
    """
    __slots__ = ['tree_id', 'dim', 'order', 'max_degree', 'coefficient', 'factors', 'sub_trees']

    def __init__(self, coefficients, exponents, prime_array, unique_factor_id_list, unique_factors, id_counter):

        # give every subtree its unique id
        # equal to the idx of its computed value in the value_array
        self.tree_id = id_counter.__next__()
        self.dim = exponents.shape[1]
        self.order = np.sum(exponents, axis=0).max()
        self.max_degree = exponents.max()

        self.factors = []
        self.sub_trees = []

        def add_factor(factor_dimensions, factor_exponents):
            """
            all scalar factors need not be factorised any more (even with an exponent > 1)

            cf. https://stackoverflow.com/questions/12377632/how-is-exponentiation-implemented-in-python
            The float.__pow__() method uses C's libm which takes full advantage of hardware support
                for binary floating point arithmetic.
            The latter represents numbers using logarithms. The logarithmic representation makes it possible
                to implement exponentation will just a single multiplication.

            -> for performance it is best to not factorize x_i^n, but just compute it in one go:
                x_i^n (1 power op.) is less expensive than computing  x_i * x_i^(n-1) (1 mul + 1 power op.)
                the memory cost is higher since more factors exist which have to be evaluated

            # TODO consider effects on tree size. factoring out x_i applies to more subtrees (more frequent!)

            -> find all unique scalar factors, add them to the tree as factors
            then find the minimal factorisation for all non scalar factors based on all other factors
            = link the factors
            :param factor_dimensions:
            :param factor_exponents:
            :return:
            """

            scalar_factors = []

            # the id of a monomial is the product of the ids of its scalar factors
            monomial_id = 1
            for d, e in zip(factor_dimensions, factor_exponents):
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
                factor = scalar_factors[0]
            else:
                try:
                    factor = unique_factors[unique_factor_id_list.index(monomial_id)]
                except ValueError:
                    unique_factor_id_list.append(monomial_id)
                    # NOTE: do not link the monomial to its scalar factors
                    # there might be another 'bigger' monomial that needs to be computed as well
                    # and its value can be used to compute the value of this monomial with less effort
                    factor = MonomialFactor(scalar_factors)
                    unique_factors.append(factor)

            self.factors.append(factor)

        def add_subtree(coeffs, expnts, factorized_rows, factor_dims, factor_expnts):

            add_factor(factor_dims, factor_expnts)

            # build a subtree with all the monomials where the factors appear
            subtree_coefficients = coeffs[factorized_rows]
            subtree_exponents = expnts[factorized_rows, :]

            # the factor has to be deducted from the exponents of the sub tree ("factored out")
            subtree_exponents = factor_out(factor_dims, factor_expnts, subtree_exponents)

            self.sub_trees.append(
                HornerTree(subtree_coefficients, subtree_exponents, prime_array, unique_factor_id_list, unique_factors,
                           id_counter))

            # all other monomials have to be factorized further
            non_factorized_rows = [r for r in range(expnts.shape[0]) if r not in factorized_rows]
            coeffs = coeffs[non_factorized_rows]
            expnts = expnts[non_factorized_rows, :]

            if DEBUG:
                assert subtree_coefficients.shape[0] == subtree_exponents.shape[0]
                assert subtree_coefficients.shape[1] == 1 and len(subtree_coefficients.shape) == 2
                assert subtree_exponents.shape[1] == expnts.shape[1] and len(subtree_exponents.shape) == 2
                assert coeffs.shape[0] == expnts.shape[0]
                assert coeffs.shape[1] == 1 and len(coeffs.shape) == 2
                assert expnts.shape[1] == expnts.shape[1] and len(expnts.shape) == 2

            return coeffs, expnts

        def decide_factorisation(coeffs, expnts):
            # greedy heuristic:
            # factor out the monomials which appears in the most terms
            # factor out the biggest factor possible (as many variables with highest degree possible)
            # NOTE: optimality is not guaranteed with this approach
            # <-> there may be a horner tree (factorisation) which can be evaluated with less instructions
            # There is no known method for selecting an optimal factorisation order

            usage_statistic = np.zeros((self.dim, int(self.max_degree + 1)), dtype=UINT_DTYPE)
            usage_statistic = compile_usage_statistic(expnts, usage_statistic)

            max_usage_count = usage_statistic.max()
            max_usage_scalar_factors = np.where(usage_statistic == max_usage_count)
            max_usage_dimensions, max_usage_exponents = max_usage_scalar_factors
            nr_max_usage_scalar_factors = max_usage_dimensions.shape[0]

            usage_rows = np.empty((nr_max_usage_scalar_factors, max_usage_count), dtype=UINT_DTYPE)
            usage_rows = compile_usage_rows(max_usage_count, max_usage_dimensions, max_usage_exponents,
                                            expnts, usage_rows)

            if nr_max_usage_scalar_factors == 1:
                # there is only one scalar factor with maximal usage
                factor_dimensions = max_usage_dimensions
                factor_exponents = max_usage_exponents
                factor_rows = usage_rows[0]
                return add_subtree(coeffs, expnts, factor_rows, factor_dimensions, factor_exponents)
            else:
                # there are multiple scalar factors which are being used maximally
                # check in which rows in the exponent matrix (monomials) each scalar factor is being used
                equal_usage_matrix = np.ones((nr_max_usage_scalar_factors, nr_max_usage_scalar_factors), dtype=np.bool)
                # check which of the factors have the same usage (rows)
                equal_usage_matrix = build_row_equality_matrix(usage_rows, equal_usage_matrix)
                # print(equal_usage_matrix)
                # pick the biggest set of factors with same usage
                set_sizes = np.sum(equal_usage_matrix, axis=0)
                max_set_size = set_sizes.max()
                # NOTE: even when every set is only of size 1, one cannot create separate subtrees for every factor
                # because the factors might have shared monomials (the usage ist just not completely identical)!
                set_nr = index_of(max_set_size, set_sizes)
                max_factor_ids = np.where(equal_usage_matrix[set_nr])[0]
                factor_rows = usage_rows[max_factor_ids[0]]  # equal for all factors in set!
                # those scalar factors combined are the maximal factor
                factor_dimensions = max_usage_dimensions[max_factor_ids]
                factor_exponents = max_usage_exponents[max_factor_ids]
                return add_subtree(coeffs, expnts, factor_rows, factor_dimensions, factor_exponents)

        # determine which coefficient the polynomial represented by the current root node has
        # = the coefficient with a zero exponent vector
        inactive_exponent_rows = np.all(exponents == 0, axis=1)

        if DEBUG:
            if np.sum(inactive_exponent_rows) > 1:
                raise ValueError('more than one empty monomial:', exponents)

        if np.sum(inactive_exponent_rows) == 0:
            # there is no zero exponent vector (= constant monomial 1)
            # create node without a coefficient
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

        # find a horner factorisation for the given polynomial
        # all monomials in this tree must be factorized until none remain
        while len(remaining_coefficients) > 0:
            remaining_coefficients, remaining_exponents = decide_factorisation(remaining_coefficients,
                                                                               remaining_exponents)

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

    def num_ops(self):
        # count the number of instructions done during eval()
        num_ops = 0
        for factor, subtree in zip(self.factors, self.sub_trees):
            num_ops += 2 + subtree.num_ops()
        return num_ops

    def subtree_amount(self):
        amount = len(self.sub_trees)
        for t in self.sub_trees:
            amount += t.subtree_amount()
        return amount

    def fill_value_array(self, value_array):
        # traverse tree and write the coefficients at the correct place
        # print('filling', self.tree_id, self.coefficient)
        value_array[self.tree_id] = self.coefficient
        for t in self.sub_trees:
            t.fill_value_array(value_array)

    def eval(self, factor_values):
        # p(x) = c_0 + f_1 p_1(x) + f_2 p_2(x) + ...
        out = self.coefficient
        for factor, sub_tree in zip(self.factors, self.sub_trees):
            # eval all sub trees
            out += factor.eval(factor_values) * sub_tree.eval(factor_values)
        return out

    def get_recipe(self):
        # p(x) = c_0 + c_1 p_1(x) + c_2 p_2(x) + ...
        # target = source1 * source2
        # target == source 1
        # values[self.id] = values[self.id] *op* values[source]
        # instruction encoding: target, source
        tree_recipe = []
        # separate: op (binary: 0/1)
        op_recipe = []
        for factor, sub_tree in zip(self.factors, self.sub_trees):
            # IMPORTANT: sub tree has to be evaluated BEFORE its value can be used!
            # -> add its recipe first
            tree_recipe_sub, op_recipe_sub = sub_tree.get_recipe()
            tree_recipe += tree_recipe_sub
            op_recipe += op_recipe_sub
            # now the value at values[subtree.id] is the evaluated subtree value
            tree_recipe += [
                # multiply the value of the subtree with the value of the factor
                (sub_tree.tree_id, factor.value_idx),
                # add this value to the previously computed value
                (self.tree_id, sub_tree.tree_id),
            ]
            op_recipe += [ID_MULT, ID_ADD]
        return tree_recipe, op_recipe
