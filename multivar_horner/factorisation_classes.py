import numpy as np

from .global_settings import DEBUG, ID_ADD, ID_MULT, UINT_DTYPE
from .helper_classes import PriorityQueue2D
from .helpers_fcts_numba import compile_valid_options, count_usage, factor_num_ops, num_ops_1D_horner, true_num_ops


class FactorisationNode(object):
    """
    A node representing a factorisation of a polynomial:
    p = f_1 * p_1 + p_2
    its sub problems are polynomials as well -> 'divide and conquer'
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['factor', 'node1_fact', 'node2', 'factorized_rows', 'non_factorized_rows', 'value_idx']

    def __init__(self, factor, node1_fact, node2, factorized_rows, non_factorized_rows):
        self.factor = factor
        self.node1_fact = node1_fact
        self.node2 = node2  # can be None if all monomials shared the factor
        self.factorized_rows = factorized_rows
        self.non_factorized_rows = non_factorized_rows
        self.value_idx = None

    def __str__(self):
        if type(self.factor) == tuple:
            dim, exp = self.factor
            if exp == 1:
                s = 'x_{}'.format(dim + 1)
            else:
                s = 'x_{}^{}'.format(dim + 1, exp)
        else:
            s = self.factor.__str__()
        s += ' (' + self.node1_fact.__str__() + ')'
        if self.node2 is not None:
            s += ' + ' + self.node2.__str__()
        return s

    def __gt__(self, other):
        # arbitrary, required for sorting in heap
        return True

    def compile_factors(self, coefficients, tree_coefficients, factor_container, id_counter):
        # create and store the unique ScalarFactor instance
        property_list = [self.factor]  # own factor is a scalar factor, only one property
        self.factor = factor_container.get_factor(property_list)

        coeffs1 = coefficients[self.factorized_rows]
        self.node1_fact.compile_factors(coeffs1, tree_coefficients, factor_container, id_counter)

        if self.node2 is not None:
            coeffs2 = coefficients[self.non_factorized_rows]
            self.node2.compile_factors(coeffs2, tree_coefficients, factor_container, id_counter)

        # this node does not need its own index in the value array, but can reuse the index of its first node
        self.value_idx = self.node1_fact.value_idxs[0]

    def get_recipe(self):
        # p = f_1 * p_1 + p_2
        # values[target] = values[target] *op* values[source]

        # IMPORTANT: sub trees have to be evaluated BEFORE their values can be used!
        # -> add their recipes first
        tree_recipe, op_recipe = self.node1_fact.get_recipe()

        # the value at values[node.idx] is the evaluated value of this node
        tree_recipe += [
            # instruction encoding: target, source
            # multiply the value of the node1 with the value of the factor
            (self.node1_fact.value_idxs[0], self.factor.value_idx),
        ]
        # separate: op (binary: 0/1)
        op_recipe += [ID_MULT]

        if self.node2 is not None:
            tree_recipe_sub, op_recipe_sub = self.node2.get_recipe()
            tree_recipe += tree_recipe_sub
            op_recipe += op_recipe_sub

            tree_recipe += [
                # add the value of node2 to this value
                (self.node1_fact.value_idxs[0], self.node2.value_idxs[0]),
            ]
            op_recipe += [ID_ADD]

        return tree_recipe, op_recipe


class OptimalFactorisationNode(FactorisationNode):
    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['cost_estimate', 'factorisation_measure', 'fully_factorized']

    def __init__(self, factor, node1_fact, node2, factorized_rows, non_factorized_rows):
        super(OptimalFactorisationNode, self).__init__(factor, node1_fact, node2, factorized_rows, non_factorized_rows)
        self.cost_estimate = 0
        self.factorisation_measure = 0
        self.fully_factorized = False
        self.update_properties()

    def update_properties(self):
        self.cost_estimate = factor_num_ops(*self.factor) + self.node1_fact.cost_estimate
        self.factorisation_measure = self.node1_fact.factorisation_measure

        if self.node2 is not None:
            self.cost_estimate += self.node2.cost_estimate
            self.factorisation_measure += self.node2.factorisation_measure
            self.fully_factorized = self.node1_fact.fully_factorized and self.node2.fully_factorized
        else:
            self.fully_factorized = self.node1_fact.fully_factorized

    def refine(self):
        assert not self.fully_factorized
        self.node1_fact.refine()
        if self.node2 is not None:
            self.node2.refine()
        self.update_properties()


class BasePolynomialNode(object):
    """
    The base class representing a multivariate polynomial as a child node in a factorisation tree for finding
        a good factorisation of a bigger multivariate polynomial
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['exponents', 'unique_exponents', 'num_monomials', 'dim', 'children', 'value_idxs', 'factors',
                 'has_children', 'children_class', 'factorisation_class']

    def __init__(self, exponents, *args, **kwargs):
        self.exponents = exponents
        self.dim = self.exponents.shape[1]
        self.num_monomials = self.exponents.shape[0]

        self.unique_exponents = []
        for d in range(self.dim):
            dim_exponents = self.exponents[:, d]
            # sort in ascending order (used for finding the most common scalar factor)
            dim_exponents_unique = np.unique(dim_exponents)
            dim_exponents_unique.sort()
            # exponent 0 is never a valid factorisation option
            if dim_exponents_unique[0] == 0:
                self.unique_exponents.append(dim_exponents_unique[1:])
            else:
                self.unique_exponents.append(dim_exponents_unique)

        self.value_idxs = None
        self.factors = None
        self.has_children = False
        self.children = None
        self.children_class = None
        self.factorisation_class = None
        self.post_init()

    def post_init(self):
        self.children_class = BasePolynomialNode
        self.factorisation_class = FactorisationNode

        # fully factorize automatically
        option = self.choose_option()
        if option is None:
            self.has_children = False
        else:
            self.has_children = True
            self.factorize(*option)

    def choose_option(self):
        """
        :return: the option with the maximum usage in all monomials
        """

        if self.num_monomials <= 1:
            # there are no options when only one monomial is left in a node!
            return None

        max_usage = 0
        max_usage_option = None

        for dim, dim_unique_exponents in enumerate(self.unique_exponents):
            # of all existing exponents, the smallest ones for sure have the highest usage
            for exp in dim_unique_exponents:
                usage = count_usage(dim, exp, self.exponents)
                if usage > max_usage:
                    max_usage = usage
                    max_usage_option = (dim, exp)
                # no other exponent in this dimension can have a higher usage
                break

        return max_usage_option

    def __str__(self):
        if self.has_children:
            return self.get_child().__str__()
        else:
            monomials = []
            for i, exp_vect in enumerate(self.exponents):
                monomial = ['c']
                for dim, exp in enumerate(exp_vect):
                    if exp > 0:
                        if exp == 1:
                            monomial.append('x_{}'.format(dim + 1))
                        else:
                            monomial.append('x_{}^{}'.format(dim + 1, exp))
                monomials.append(' '.join(monomial))
            return ' + '.join(monomials)

    def factorize(self, dim, exp):
        # create a representation of the polynomial with all the monomials 'benefiting' from factorisation
        factorized_rows = self.exponents[:, dim] >= exp
        exponents1_fact = self.exponents[factorized_rows, :]
        # the factor has to be deducted from the exponents (="factored out")
        exponents1_fact[:, dim] -= exp

        node1_fact = self.children_class(exponents1_fact)

        # create a representation of the polynomial with all the remaining monomials
        non_factorized_rows = np.invert(factorized_rows)
        if not np.any(non_factorized_rows):
            # no monomials remain, do not create a polynomial representation
            node2 = None
        else:
            exponents2 = self.exponents[non_factorized_rows, :]
            node2 = self.children_class(exponents2)
            if DEBUG:
                assert exp > 0
                assert not np.any(exponents1_fact < 0)
                assert not np.any(exponents2 < 0)
                assert exponents1_fact.shape[1] == exponents2.shape[1] and len(exponents2.shape) == 2
                assert exponents1_fact.shape[0] + exponents2.shape[0] == self.num_monomials

        factor = (dim, exp)
        child = self.factorisation_class(factor, node1_fact, node2, factorized_rows, non_factorized_rows)
        self.store_child(child)

    def store_child(self, child):
        self.children = child

    def get_child(self):
        return self.children

    def compile_factors(self, coefficients, tree_coefficients, factor_container, id_counter):
        """

        :param coefficients: numpy array of coefficients of the root polynomial
        :param tree_coefficients: list of coefficients in the order of appearance in the value array
        :param factor_container: class storing and managing all existing factors
        :param id_counter: counter object for assigning indices in the value array
        :return:
        """
        if self.has_children:
            child = self.get_child()
            child.compile_factors(coefficients, tree_coefficients, factor_container, id_counter)
            # this node does not need its own index in the value array, but can reuse the value index of its sub problem
            self.value_idxs = [child.value_idx]
        else:
            # this node cannot be factorized (no options)
            # for evaluation the sum of all evaluated monomials multiplied with their coefficients has to be computed
            # p = c1 * mon1 + c2 * mon2 ...
            self.factors = factor_container.compile_factors(self.exponents)
            # the coefficients are stored in the value array and all need their own index
            self.value_idxs = []
            for i, factor in enumerate(self.factors):
                self.value_idxs.append(id_counter.__next__())
                tree_coefficients.append(coefficients[i])
            # TODO
            assert (len(coefficients) == len(self.factors))

    def get_recipe(self):
        """
        :return: the list of instructions ("recipe") required for evaluating the represented factorisation
        """
        if self.has_children:
            # the own recipe is the recipe of its factorisation
            return self.get_child().get_recipe()
        else:
            # this node cannot be factorized
            # for evaluation the sum of all evaluated monomials multiplied with their coefficients has to be computed
            # p = c1 * mon1 + c2 * mon2 ...
            # the final evaluated value of this node has to be stored at the first of its value indices
            initial_coeff_value_idx = self.value_idxs[0]
            factor = self.factors[0]
            tree_recipe = []
            op_recipe = []

            if factor is not None:
                # multiply the first coefficients with the value of the first factor
                tree_recipe += [(initial_coeff_value_idx, self.factors[0].value_idx)]
                op_recipe += [ID_MULT]

            for i in range(1, len(self.value_idxs)):
                coeff_value_idx = self.value_idxs[i]
                factor = self.factors[i]
                if factor is not None:
                    # multiply each coefficients with the value of the factor
                    tree_recipe += [
                        (coeff_value_idx, factor.value_idx),
                    ]
                    op_recipe += [ID_MULT]

                # add this value to the previously computed value
                tree_recipe += [
                    (initial_coeff_value_idx, coeff_value_idx),
                ]
                op_recipe += [ID_ADD]
            return tree_recipe, op_recipe


class OptimalPolynomialNode(BasePolynomialNode):
    """
    A class representing a multivariate polynomial with all possible factorisation options.
    The different factorisation options are being stored sorted in a heap (self.children)
    An instance of this class is a node (=subproblem) in search tree for finding
    the optimal factorisation of a bigger multivariate polynomial
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['options', 'cost_estimate', 'factorisation_measure', 'fully_factorized']

    def __init__(self, exponents, *args, **kwargs):

        self.cost_estimate = 0
        self.factorisation_measure = 0
        self.fully_factorized = True
        self.options = None

        # self.post_init() is being called at the end
        super(OptimalPolynomialNode, self).__init__(exponents, *args, **kwargs)

    def post_init(self):
        self.children_class = OptimalPolynomialNode
        self.factorisation_class = OptimalFactorisationNode
        self.get_all_options()
        self.estimate_cost()

    def __eq__(self, other):
        """
        TODO improvement:
        different orders of factorisations can lead to identical sub problems
        solving them independently leads to unnecessary computations
        -> before creating a new node check if it already exists (equality comparison required)
        equality test can be aborted when certain global properties of nodes do not match
        (cost_estimate, dimensions of exponent matrix...)
        only actually compare exponent matrices for equality after all previous tests passed
        ATTENTION: after changing a multiply used node all parents have to be updated!
        """
        raise NotImplementedError

    def store_child(self, child):
        self.children.put(child)

    def get_child(self):
        # return the currently most promising factorisation
        return self.children.get()

    def estimate_cost(self):
        """
        :return: the estimated amount of operations required needed when evaluating this polynomial in fully
            factorized form. true amount when fully factorized otherwise a lower bound.

        Every existing factor (represented by exponents) has to be evaluated, which minimally
        requires the amount of operations all respective 1D-Horner factorisations would take.
        NOTE: not counting the amount of ADD operations and operations needed because of the coefficients,
            since they stay constant in sum irrespective of the factorisation (= number of monomials)

        TODO: improvement idea: more ops certainly needed when exponents appear multiple times in exponent vectors
         (monomials are unique) and can hence by definition not be fully factorized... but do not count multiple times
         -> derive rule.
        """
        if self.fully_factorized:
            # use the true number of operations
            self.cost_estimate = true_num_ops(self.exponents)
        else:
            heuristic = 0
            for dim_unique_exp in self.unique_exponents:
                heuristic += num_ops_1D_horner(dim_unique_exp)
            self.cost_estimate = heuristic

    def get_all_options(self):
        """
        :return: a list of all meaningful options (dim, exp) for factorisation

        NOTE: scalar factors (<-> exponents) which appear only in one monomial are not meaningful factors
            Picking them would only lead to separating this one monomial = equal to not factorising at all
            no decrease in #ops -> the heuristic would not change -> also no change in search
            monomials without any common exponents should not be factorized (can be computed without factorisation)
        """

        options = []
        if self.num_monomials <= 1:
            # there are no options when only one monomial is left in a node!
            return options

        for dim, dim_unique_exponents in enumerate(self.unique_exponents):
            usage_vector = np.zeros(dim_unique_exponents.shape, dtype=UINT_DTYPE)
            valid_option_vector = np.zeros(dim_unique_exponents.shape, dtype=bool)
            # TODO test
            valid_option_vector = compile_valid_options(dim, valid_option_vector, usage_vector, dim_unique_exponents,
                                                        self.exponents)

            for exp in dim_unique_exponents[valid_option_vector]:
                options.append((dim, exp))

        self.factorisation_measure = len(options)
        self.fully_factorized = len(options) == 0
        self.options = options
        pass

    def refine(self):
        """
        factorize further (grow tree)

        TODO improvement:
        initially compute good factorisation (with heuristic). use as upper bound for the #ops
        do not keep factorisations which have a higher or equal! estimated #ops (<- save memory)
        when heap becomes empty, the heuristic solution is optimal
        apply to all layers, pass upper bound down the tree (reduce accordingly)
        NOTE: when searching all optimal solutions having the same #ops as the upper bound is allowed!
        ATTENTION: cost estimate is not counting additions (cf. estimate_cost() ) do not compare with total #ops!

        TODO improvement:
        while loop keep factorising until no other option has a lower heuristic or fully factorised
        -> check what the next lowest heuristic is. corner case: no other option available!
        """
        # TODO remove
        # print('\nrefining', self.fully_factorized, self.cost_estimate, self)
        if self.fully_factorized:
            # this check is required, because nodes at the same depth in the factorisation tree might have different
            # factorisation statuses (.refine() is still being called)
            return

        if self.has_children:
            # the heap has been initialized
            # pick the most promising factorisation subtree, factorise it further
            promising_node = self.children.pop()  # pop, not get!
            # print('refining', promising_node.fully_factorized,  promising_node.cost_estimate, promising_node)
            promising_node.refine()
            # print('done', promising_node.fully_factorized,  promising_node.cost_estimate, promising_node)
            self.store_child(promising_node)
        else:
            # no factorisation has been done before
            # initialize heap
            # the cost estimate of less factorised bad factorisations and more factorised good factorisation is similar
            # to reach a fully factorised solution faster, more factorised options have to be favoured
            # -> have a "2D heap" sorted first after the cost estimate and then after the factorisation progress
            self.children = PriorityQueue2D()
            # use all options and store the different factorisation results in heap
            # there might be many options and trying all is often unnecessary,
            # but trying just one is also not efficient, because the unfactorized node most certainly
            # has a lower cost estimate and hence would get picked next anyway
            for dim, exp in self.options:
                self.factorize(dim, exp)

            self.has_children = True

        # print('all children:')
        # for c in self.children.get_all():
        #     print('\t', c.fully_factorized, c.cost_estimate,c)

        # the new properties of this node are equal to the properties of the most promising new subtree
        promising_node = self.get_child()
        # print('most promising:', promising_node.fully_factorized, promising_node.cost_estimate, promising_node)

        self.factorisation_measure = promising_node.factorisation_measure
        self.fully_factorized = promising_node.fully_factorized
        # cost_estimate_prev = self.cost_estimate
        self.cost_estimate = promising_node.cost_estimate
        # the initial heuristic is a lower bound for the true cost
        # the cost estimate should always increase or stay constant (= getting more realistic) with every refinement
        # assert cost_estimate_prev <= self.cost_estimate


class OptimalFactorisationRoot(OptimalPolynomialNode):
    """
    Class for finding optimal factorisations of given multivariate polynomials
    Basic idea: adapted A* search
        allow all meaningful possible factorisation while ranking them according to
        their lowest possible amount of operations needed for evaluation (heuristic)
        iteratively factorize the most promising factorisation further until a full factorisation has been found

    guaranteed to yield a factorisation with the minimal amount of operations required during evaluation (optimal)
    memory and time consumption is much higher than with using a heuristic to pick a single factorisation

    TODO improvement: analyse and optimize runtime of optimal factorisation search
    tradeoff between accuracy of the heuristic (-> ignore irrelevant factorisations)
        and time to compute heuristic (-> better to just try factorisations)


    TODO improvement idea: instead of just optimizing for the lowest amount of operations,
    also consider the different costs of the different operations on computer architectures
    (optimize total evaluation time).
    ATTENTION: a completely new heuristic for estimating the lower bound would be required!
    """

    def __init__(self, coefficients, tree_coefficients, id_counter, factor_container, *args, **kwargs):
        """
        :param coefficients: a numpy array column vector of doubles representing the coefficients of the monomials
        :param exponents: a numpy array matrix of unsigned integers representing the exponents of the monomials
            the ordering does not matter, but every exponent row has to be unique!
        :param rectify_input: whether to convert the input parameters into compatible numpy arrays
        :param validate_input: whether to check if the input parameters fulfill the requirements
        """
        super(OptimalFactorisationRoot, self).__init__(*args, **kwargs)

        while not self.fully_factorized:
            # TODO report current search status, verbosity option, not every loop
            # (#total factorisations done, #max depth, #open possibilities, upper bound, lower bound)
            self.refine()

        # the most promising factorisation (lowest cost estimate) is fully factorized now
        # collect all appearing factors in the best solution and assign ids (=idx in the value array) to the nodes
        self.compile_factors(coefficients, tree_coefficients, factor_container, id_counter)

    def find_all_optimal(self):
        """
        TODO refine and collect all fully factorized optimal solutions until the next best factorisation
            has a higher cost estimate
        Attention: do not remove entire subtree, but just one realisation each time!
        TODO analyse obtained optimal factorisations
        use this data to derive better heuristics for factorizing more optimally a priori
        :return:
        """
        raise NotImplementedError


class HeuristicFactorisationRoot(BasePolynomialNode):
    """
    Class for finding a good factorisation of a given multivariate polynomial by iteratively factoring out the most
    commonly used factor (heuristic).
    """

    def __init__(self, coefficients, tree_coefficients, id_counter, factor_container, *args, **kwargs):
        super(HeuristicFactorisationRoot, self).__init__(*args, **kwargs)
        # polynomial is fully factorized now
        # collect all appearing factors in the factorisation tree and assign ids (=idx in the value array) to the nodes
        self.compile_factors(coefficients, tree_coefficients, factor_container, id_counter)
