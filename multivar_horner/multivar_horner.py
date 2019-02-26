# TODO s:

# matlab binding?
# optimize naive version without factorisation: compute all needed exponents only once. order exponents cleverly.
# store at which index the later entries (exponents in column) will just be 0 -> reduce the matrix size piecewise

# test gradient
# multivariate newton raphson method
# mention new features in readme
# function to predict the factorisation time and efficiency (time saved) based on dim, max_degree and num_entries
# based on system
# suggest function to use

# test factorisation for univariate polynomials! should always find optimum (unique 1D horner fact.)


import itertools
import pickle

import numpy as np

from .factorisation_classes import HeuristicFactorisationRoot, OptimalFactorisationRoot
from .global_settings import DEBUG, DEFAULT_PICKLE_FILE_NAME, FLOAT_DTYPE, UINT_DTYPE
from .helper_classes import FactorContainer, ScalarFactor
from .helper_fcts import get_prime_array, rectify, validate
from .helpers_fcts_numba import eval_recipe, naive_eval


# is not a helper function to make it an importable part of the package
def load_pickle(path=DEFAULT_PICKLE_FILE_NAME):
    print('importing polynomial from file "{}"'.format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)


class MultivarPolynomial(object):
    """
    naive representation of a multivariate polynomial without any horner factorisation

    dim: the dimensionality of the polynomial.
    order: (for a multivariate polynomial) maximum sum of exponents in any of its monomials
    max_degree: the largest exponent in any of its monomials
        NOTE: the polynomial actually needs not to depend on all dimensions
    unused_variables: the dimensions the polynomial does not depend on
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['coefficients', 'exponents', 'existing_exponents', 'num_monomials', 'dim', 'order', 'max_degree',
                 'unused_variables', 'representation', 'heuristic']

    def __init__(self, coefficients, exponents, rectify_input=False, validate_input=False):
        """
        :param coefficients: a numpy array column vector of doubles representing the coefficients of the monomials
        :param exponents: a numpy array matrix of unsigned integers representing the exponents of the monomials
            the ordering does not matter, but every exponent row has to be unique!
        :param rectify_input: whether to convert the input parameters into compatible numpy arrays
        :param validate_input: whether to check if the input parameters fulfill the requirements
        """

        if rectify_input:
            coefficients, exponents = rectify(coefficients, exponents)

        if validate_input:
            validate(coefficients, exponents)

        self.coefficients = coefficients
        self.exponents = exponents

        self.num_monomials = self.exponents.shape[0]
        self.dim = self.exponents.shape[1]
        self.order = np.sum(self.exponents, axis=0).max()
        self.max_degree = self.exponents.max()

        self.unused_variables = np.where(~np.any(self.exponents, axis=1))[0]
        self.representation = ''
        self.existing_exponents = None
        self.heuristic = None

    def __str__(self):
        return self.get_representation()

    def __repr__(self):
        return self.__str__()

    def get_representation(self):

        s = '[#ops={}] p(x) = '.format(self.get_num_ops())
        monomials = []
        for i, exp_vect in enumerate(self.exponents):
            monomial = [str(self.coefficients[i, 0])]
            for dim, exp in enumerate(exp_vect):
                # if exp > 0:
                # show all operations, even 1 * x_i^0
                monomial.append('x_{}^{}'.format(dim + 1, exp))
            monomials.append(' '.join(monomial))

        s += ' + '.join(monomials)
        return s

    def export_pickle(self, path=DEFAULT_PICKLE_FILE_NAME):
        print('storing polynomial in file "{}" ...'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print('...done.\n')

    def get_num_ops(self):
        # count the number of instructions done when evaluating polynomial:
        y, x = self.exponents.shape
        # exponentiation: x*y
        # multiplication coefficient and scalar factors (monomials): x*y
        # sum: y-1
        return 2 * x * y + y - 1

    def eval(self, x, validate_input=True):
        """
        :param x:
        :param validate_input:
        :return:
        """

        if validate_input:
            x = np.array(x)
            assert len(x.shape) == 1
            assert x.shape[0] == self.dim

        return naive_eval(x, self.coefficients, self.exponents)

    def get_partial_derivative(self, i):
        """
        TODO test
        f(x) = a x^b
        f'(x) = ab x^(b-1)
        :param i:
        :return: the partial derivative of this polynomial wrt. the i-th coordinate
        # ATTENTION: coordinate counting starts with 1! (first coordinates is #1)
        """

        assert (0 < i <= self.dim)
        coord_index = i - 1
        # IMPORTANT: do not modify the stored coefficient and exponent arrays of self!
        # set all the coefficients not depending on the i-th coordinate to 0
        # this simply means not adding them to the list of coefficients of the new polynomial class instance
        active_rows = np.where(self.exponents[:, coord_index] >= 1)[0]

        new_coefficients = self.coefficients[active_rows]
        new_exponents = self.exponents[active_rows, :]

        if DEBUG:
            assert new_coefficients.shape[0] == new_exponents.shape[0]
            assert new_coefficients.shape[1] == 1 and len(new_coefficients.shape) == 2
            assert new_exponents.shape[1] == self.dim and len(new_exponents.shape) == 2

        # multiply the coefficients with the exponent of the i-th coordinate
        new_coefficients = np.multiply(new_coefficients.flatten(), new_exponents[:, coord_index])
        new_coefficients = new_coefficients.reshape(-1, 1)

        # reduce the the exponent of the i-th coordinate by 1
        new_exponents[:, coord_index] -= 1

        # must call the proper constructor method also for inherited classes
        # TODO also use option 'univariate_factors'
        return self.__class__(new_coefficients, new_exponents)

    def get_gradient(self):
        """
        :return: the list of all partial derivatives
        """
        return [self.get_partial_derivative(i) for i in range(1, self.dim + 1)]


class HornerMultivarPolynomial(MultivarPolynomial):
    """
    a representation of a multivariate polynomial using horner factorisation to save evaluation time

    dim: the dimensionality of the polynomial.
    order: (for a multivariate polynomial) maximum sum of exponents in any of its monomials
    max_degree: the largest exponent in any of its monomials
        NOTE: the polynomial actually needs not to depend on all dimensions
    unused_variables: the dimensions the polynomial does not depend on
    """
    # __slots__ declared in parents are available in child classes. However, child subclasses will get a __dict__
    # and __weakref__ unless they also define __slots__ (which should only contain names of any additional slots).
    __slots__ = ['prime_array', 'factorisation_tree', 'factor_container', 'num_ops', 'initial_value_array',
                 'copy_recipe', 'scalar_recipe', 'monomial_recipe', 'tree_recipe', 'tree_ops']

    def __init__(self, coefficients, exponents, rectify_input=False, validate_input=False, keep_tree=False,
                 compute_representation=False, find_optimal=False):
        """
        :param coefficients: a numpy array column vector of doubles representing the coefficients of the monomials
        :param exponents: a numpy array matrix of unsigned integers representing the exponents of the monomials
            the ordering does not matter, but every exponent row has to be unique!
        :param rectify_input: whether to convert the input parameters into compatible numpy arrays
        :param validate_input: whether to check if the input parameters fulfill the requirements
        """

        def compute_num_ops():
            # count the number of instructions done during evaluation
            # (computing all factors + evaluating the factorisation subtrees)
            # for every entry in each recipe one operation is being done
            # do not count "copy" instructions (not really a mathematical operation)
            self.num_ops = self.scalar_recipe.shape[0] + self.monomial_recipe.shape[0] + self.tree_ops.shape[0]

        def get_string_representation():
            representation = '[#ops={}] p(x)'.format(self.num_ops)
            if compute_representation:
                representation += ' = ' + self.factorisation_tree.__str__()
            return representation

        super(HornerMultivarPolynomial, self).__init__(coefficients, exponents, rectify_input, validate_input)

        # the needed prime numbers for computing all goedel numbers of all used factors
        self.prime_array = get_prime_array(self.dim)

        # store all unique factors of the horner factorisation
        # NOTE: do NOT create all scalar factors with exponent 1 (they might be unused!)
        self.factor_container = FactorContainer(self.prime_array)

        # factorize the polynomial once and store the factorisation as a tree
        id_counter = itertools.count(0)
        tree_coefficients = []
        if find_optimal:
            self.factorisation_tree = OptimalFactorisationRoot(self.coefficients, tree_coefficients, id_counter,
                                                               self.factor_container, exponents=self.exponents)
        else:
            self.factorisation_tree = HeuristicFactorisationRoot(self.coefficients, tree_coefficients, id_counter,
                                                                 self.factor_container, exponents=self.exponents)

        # factor list is now filled with the unique factors
        # during evaluation of a polynomial the values of all the unique factors are needed at least once
        # -> compute the values of all factors (monomials) once and store them
        # store a pointer to the computed value for every unique factor
        # save instructions for the evaluation -> find factorisation of all factors = again a factorisation tree
        # sort and factorize the monomials to quickly evaluate them once during a query
        self.link_monomials()

        # compile and store a "recipe" for evaluating the polynomial with just numpy arrays
        self.initial_value_array, self.copy_recipe, self.scalar_recipe, self.monomial_recipe, self.tree_recipe, \
            self.tree_ops = self.compile_recipes(tree_coefficients)

        compute_num_ops()
        self.representation = get_string_representation()

        if not keep_tree:
            # the trees and factors are not needed any more
            # a value lookup can be done with just the recipe
            # free up the memory
            del self.factorisation_tree
            del self.factor_container
            del self.prime_array

    def __str__(self):
        return self.representation

    def get_num_ops(self):
        return self.num_ops

    def link_monomials(self):
        """
        find the optimal factorisation of the unique factors themselves
        since the monomial ids are products of the ids of their scalar factors
        check for the highest possible divisor among all factor ids
        this leads to a minimal factorisation for quick evaluation of the monomial values
        :return:
        """
        # sort after their id
        unique_factor_id_list = list(sorted(self.factor_container.id2idx.keys()))
        unique_factors = list(sorted(self.factor_container.factors, key=lambda f: f.monomial_id))
        # property of the list: the scalar factors of each monomial are stored in front of it

        # IMPORTANT: properly set the indices of the values for each factor
        # the values of every factor are stored after the coefficients
        value_idx_offset = self.num_monomials
        for idx, factor in enumerate(unique_factors):
            factor.value_idx = idx + value_idx_offset

        # start at the last factor (highest id)
        pointer1 = len(unique_factors) - 1

        # the smallest factor has no factorisation (stop at pointer=1)
        while pointer1 > 0:
            candidate = unique_factors[pointer1]

            if type(candidate) is ScalarFactor:
                # scalar factors have no useful factorisation
                # (can be directly computed with the same amount of operations)
                pointer1 -= 1
                continue

            # print(1, self.unique_factors[pointer1])
            remaining_factor_id = candidate.monomial_id
            # store the indices of the factors that divide the id
            factorisation_idxs = []

            pointer2 = pointer1 - 1
            # find the factors with the highest id which are a factor of the current monomial
            # TODO back track when no factorisation is possible with that factor
            # try with the remaining
            while 1:
                if pointer2 < 0:
                    # no factorisation of this monomial has been found, because the remainder after
                    # picking a factorizing monomial cannot be factorised itself
                    # just pick the scalar factors of the monomial
                    # ATTENTION: offset needed!
                    factorisation_idxs = [unique_factor_id_list.index(scalar_factor.monomial_id) + value_idx_offset
                                          for scalar_factor
                                          in candidate.scalar_factors]
                    break

                monomial_id2 = unique_factors[pointer2].monomial_id

                if remaining_factor_id < monomial_id2:
                    # this monomial has a higher id than the remaining factor. it cannot be a factorisation
                    pointer2 -= 1
                    continue

                if monomial_id2 == remaining_factor_id:
                    # the last factor has been found
                    # ATTENTION: offset needed!
                    factorisation_idxs.append(pointer2 + value_idx_offset)
                    break

                quotient, remainder = divmod(remaining_factor_id, monomial_id2)
                if remainder == 0:
                    # this factor is a factor of the monomial
                    # ATTENTION: offset needed!
                    factorisation_idxs.append(pointer2 + value_idx_offset)
                    # reduce the id
                    remaining_factor_id = quotient

                pointer2 -= 1

            # print('found factorisation: {} ='.format(self.unique_factors[pointer1]),
            #       ' * '.join([str(self.unique_factors[idx]) for idx in reversed(factorisation_idxs)]))
            # store the indices of the factors that factorize the monomial
            #   to quickly access their values when computing the value of the monomial
            candidate.factorisation_idxs = factorisation_idxs

            pointer1 -= 1

    def compile_recipes(self, tree_coefficients):
        # compile a recipe encoding all needed instructions in order to evaluate the polynomial
        # = clever data structure for representing the factorisation tree
        # -> avoid recursion and function call overhead while evaluating

        # for evaluating the polynomial in tree form intermediary computation results have to be stored in a value array
        # the value array has one entry for every subtree and one for every factor
        value_array_length = self.num_monomials + len(self.factor_container.factors)

        initial_value_array = np.empty(value_array_length, dtype=FLOAT_DTYPE)

        # initial_value_array = np.array(tree_coefficients, dtype=FLOAT_DTYPE)..reshape(value_array_length)
        # the initial value array has the coefficients of all subtrees stored at the index of their id
        for i, v in enumerate(tree_coefficients):
            initial_value_array[i] = v

        # self.horner_tree.fill_value_array(value_array)

        # compile the recipes for computing the factors
        # scalar factors (depending on x) are being evaluated differently
        #   from the monomial factors (depending on scalar factors)
        copy_recipe = []  # skip computing factors with exp 1, just copy x vals
        scalar_recipe = []
        monomial_recipe = []
        for f in self.factor_container.factors:
            copy_instr, scalar_instr, monomial_instr = f.get_recipe()
            copy_recipe += copy_instr
            scalar_recipe += scalar_instr
            monomial_recipe += monomial_instr

        # compile the recipe for evaluating the horner factorisation tree
        tree_recipe, tree_ops = self.factorisation_tree.get_recipe()
        # convert and store the recipes
        # for the recipes numba is expecting the data types:
        #   array(uint, 2d, C),
        #   separate boolean array for operations, uint not needed (just 0 or 1)
        return (initial_value_array,
                np.array(copy_recipe, dtype=UINT_DTYPE).reshape((-1, 2)),
                np.array(scalar_recipe, dtype=UINT_DTYPE).reshape((-1, 3)),
                np.array(monomial_recipe, dtype=UINT_DTYPE).reshape((-1, 3)),
                np.array(tree_recipe, dtype=UINT_DTYPE).reshape((-1, 2)),
                np.array(tree_ops, dtype=np.bool))

    def eval(self, x, validate_input=False):
        """
        make use of numba precompiled evaluation function:
        IDEA: encode factorisation in numpy array. "recipe" which values to add or multiply when
        TODO parallel computing, split up the recipe in independent parts (evaluation of sub trees)
        :param x:
        :param validate_input: whether to check if the input parameters fulfill the requirements
        :return:
        """

        if validate_input:
            x = np.array(x)
            assert len(x.shape) == 1
            assert x.shape[0] == self.dim

        # IMPORTANT: copy the initial value array
        # the array is being used as temporal storage and values would get the overwritten
        return eval_recipe(x, self.initial_value_array.copy(), self.copy_recipe, self.scalar_recipe,
                           self.monomial_recipe, self.tree_recipe, self.tree_ops)
