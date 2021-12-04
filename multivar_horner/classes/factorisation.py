import numpy as np

from multivar_horner.classes.helpers import AbstractFactor, FactorContainer, PriorityQueue2D
from multivar_horner.global_settings import BOOL_DTYPE, ID_ADD, ID_MULT, UINT_DTYPE
from multivar_horner.helpers_fcts_numba import (
    compile_valid_options,
    count_num_ops_naive,
    count_usage,
    factor_num_ops,
    num_ops_1D_horner,
)


class FactorisationNode:
    """
    A node representing a factorisation of a polynomial:
    p = f_1 * p_1 + p_2
    its sub problems are polynomials as well ('divide and conquer' -> recursion)
    factorisation in this way results in building a binary "Horner Factorisation Tree"
    TODO factorisation is ambiguous: f = f1 f2 = f2 f1
        increases the amount of possible factorisations (relevant for search over all factorisations!)
        allow whole monomials as factors or remove ambiguity by uniquely identifying identical polynomials
        just important for optimal factorisation search. not a priority atm.
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = [
        "factor",
        "node1_fact",
        "node2",
        "factorized_rows",
        "non_factorized_rows",
        "value_idx",
        "num_ops",
    ]

    def __init__(self, factor, node1_fact, node2, factorized_rows, non_factorized_rows):
        self.factor: AbstractFactor = factor
        self.node1_fact: BasePolynomialNode = node1_fact
        self.node2: BasePolynomialNode = node2  # can be None if all monomials shared the factor
        self.factorized_rows = factorized_rows
        self.non_factorized_rows = non_factorized_rows
        self.value_idx: int
        self.num_ops: int

    def get_string_representation(self, *args, **kwargs):
        s = self.factor.__str__(*args, **kwargs)
        s += " ({})".format(self.node1_fact.get_string_representation(*args, **kwargs))
        if self.node2 is not None:
            s += " + {}".format(self.node2.get_string_representation(*args, **kwargs))
        return s

    def __str__(self, *args, **kwargs):
        return self.get_string_representation(*args, **kwargs)

    def __gt__(self, other):  # arbitrary, required for sorting in heap
        return True

    def compile_factors(self, factor_container, coefficient_idxs):
        # create and store the unique ScalarFactor instance
        # own factor is a scalar factor: only one property
        self.factor = factor_container.get_factor([self.factor])

        coeff_idxs1 = coefficient_idxs[self.factorized_rows]
        self.node1_fact.compile_factors(factor_container, coeff_idxs1)

        if self.node2 is not None:
            coeff_idxs2 = coefficient_idxs[self.non_factorized_rows]
            self.node2.compile_factors(factor_container, coeff_idxs2)

        # this node does not need its own index in the value array, but can reuse the index of its first node
        # child nodes are being evaluated earlier
        # NOTE: all nodes have at least one value index, since a node always contains a coefficient
        self.value_idx = self.node1_fact.value_idxs[0]

    def get_recipe(self):
        # p = f_1 * p_1 + p_2
        # values[target] = values[target] *op* values[source]

        # IMPORTANT: sub trees have to be evaluated BEFORE their values can be used!
        # -> add their recipes first
        tree_recipe, op_recipe = self.node1_fact.get_recipe()

        # the value at values[node.idx] is the evaluated value of this node
        target = self.node1_fact.value_idx
        tree_recipe += [
            # instruction encoding: target, source
            # multiply the value of the node1 with the value of the factor
            (target, self.factor.value_idx),
        ]
        # separate: op (binary: 0/1)
        op_recipe += [ID_MULT]

        if self.node2 is not None:
            tree_recipe_sub, op_recipe_sub = self.node2.get_recipe()
            tree_recipe += tree_recipe_sub
            op_recipe += op_recipe_sub

            tree_recipe += [
                # add the value of node2 to this value
                (target, self.node2.value_idx),
            ]
            op_recipe += [ID_ADD]

        self.num_ops = len(tree_recipe) + len(op_recipe)
        return tree_recipe, op_recipe

    def get_instructions(self, coeff_array: str, factor_array: str) -> str:
        """
        :return: the instructions for computing the value
        of the polynomial represented by this factorisation in C syntax
        """
        # eval node 1 -> the value will be stored at its first position
        node1 = self.node1_fact
        instr = node1.get_instructions(coeff_array, factor_array)
        self.num_ops = node1.num_ops
        # MULTIPLY the value of the node1 with the value of the factor
        target = node1.value_idxs[0]
        source = self.factor.value_idx
        target_instr = f"{coeff_array}[{target}]"
        instr += f"{target_instr} *= {factor_array}[{source}];\n"
        self.num_ops += 1

        node2 = self.node2
        if node2 is not None:
            # evaluate the node 2
            instr += node2.get_instructions(coeff_array, factor_array)
            self.num_ops += node2.num_ops

            # ADD this value to the value of node 1
            source = node2.value_idxs[0]
            instr += f"{target_instr} += {coeff_array}[{source}];\n"
            self.num_ops += 1

        return instr


class OptimalFactorisationNode(FactorisationNode):
    __slots__ = ["cost_estimate", "factorisation_measure", "fully_factorized"]

    node1_fact: "OptimalPolynomialNode"
    node2: "OptimalPolynomialNode"

    def __init__(self, factor, node1_fact, node2, factorized_rows, non_factorized_rows):
        super(OptimalFactorisationNode, self).__init__(factor, node1_fact, node2, factorized_rows, non_factorized_rows)
        self.cost_estimate: int = 0
        # IDEA: when different factorisations have the same cost estimate,
        # favour the one which is factorised the most already
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


class BasePolynomialNode:
    """
    The base class representing a multivariate polynomial as a child node in a factorisation tree for finding
        a good factorisation of a bigger multivariate polynomial

    TODO document
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = [
        "exponents",
        "unique_exponents",
        "num_monomials",
        "num_ops",
        "dim",
        "children",
        "value_idxs",
        "factors",
        "has_children",
        "children_class",
        "factorisation_class",
    ]

    def __init__(self, exponents, *args, **kwargs):
        self.exponents = exponents
        self.num_monomials = self.exponents.shape[0]
        self.num_ops: int
        self.dim = self.exponents.shape[1]

        self.unique_exponents = []

        #  all monomials are independent...
        # if np.all(np.sum(exponent_matrix, axis=0) == np.max(exponent_matrix, axis=0)):
        # NOTE: not useful (too expensive)

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

        self.children = None
        self.value_idxs = None
        self.factors = None
        self.has_children = False
        self.children_class = None
        self.factorisation_class = None
        self.post_init()

    @property
    def value_idx(self) -> int:
        return self.value_idxs[0]

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
        :return: the option with the maximum usage in all monomials (<- 'heuristic')
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

    def get_string_representation(self, coefficients=None, coeff_fmt_str="{:.2}", factor_fmt_str="x_{dim}^{exp}"):

        if self.has_children:
            return self.get_child().get_string_representation(
                coefficients=coefficients,
                coeff_fmt_str=coeff_fmt_str,
                factor_fmt_str=factor_fmt_str,
            )
        else:
            monomial_representations = []
            for i, exp_vect in enumerate(self.exponents):
                if coefficients is None:
                    coeff_repr = "c"
                else:
                    coeff_idx = self.value_idxs[i]  # look up the correct index of the coefficient
                    coeff_repr = coeff_fmt_str.format(coefficients[coeff_idx, 0])

                monomial_repr = [coeff_repr]
                for dim, exp in enumerate(exp_vect):
                    if exp > 0:
                        monomial_repr.append(factor_fmt_str.format(**{"dim": dim + 1, "exp": exp}))

                monomial_representations.append(" ".join(monomial_repr))
            return " + ".join(monomial_representations)

    def __str__(self):
        return self.get_string_representation()

    def factorize(self, dim, exp):
        """
        factorize the polynomial represented by this node:
        find all the monomials 'benefiting' from this factorisation
        create and store a factorized representation of this polynomial
        'top down' approach
        """

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

            # if DEBUG:  # TODO move to tests, create unit tests
            #     assert exp > 0
            #     assert not np.any(exponents1_fact < 0)
            #     assert not np.any(exponents2 < 0)
            #     assert exponents1_fact.shape[1] == exponents2.shape[1] and len(exponents2.shape) == 2
            #     assert exponents1_fact.shape[0] + exponents2.shape[0] == self.num_monomials

        factor = (dim, exp)
        child = self.factorisation_class(factor, node1_fact, node2, factorized_rows, non_factorized_rows)
        self.store_child(child)

    def store_child(self, child: "BasePolynomialNode"):
        self.children = child  # allow only one factorisation

    def get_child(self):
        return self.children

    def compile_factors(self, factor_container: FactorContainer, coefficient_idxs):
        """
        factorisation has been done (fixed)
        now "collect" all existing factors and link value addresses correctly
        'bottom up' approach
        :param factor_container: class storing and managing all existing factors
        :param coefficient_idxs: the indices of the given coefficients in the global coefficient array
            of the root polynomial
        :return:
        """
        if self.has_children:
            child = self.get_child()
            child.compile_factors(factor_container, coefficient_idxs)
            # this node does not need its own index (storage space) in the value array,
            # it can reuse the value index of its sub problem
            self.value_idxs = [child.value_idx]
        else:
            # this node cannot be factorized (no options)
            # for evaluation the sum of all evaluated monomials multiplied with their coefficients has to be computed
            # p = c1 * mon1 + c2 * mon2 ...
            # create factors representing the remaining monomials
            self.factors = factor_container.get_factors(self.exponents)  # retains the ordering!
            # remember where in the coefficient array the coefficients of this polynomial are being stored
            self.value_idxs = coefficient_idxs

    def get_recipe(self):
        """
        :return: the list of instructions ("recipe") required for evaluating the represented factorisation
        """
        if self.has_children:
            # the own recipe is the recipe of its factorisation
            tree_recipe, op_recipe = self.get_child().get_recipe()
        else:
            # this node has not been factorized (represents a regular polynomial)
            # for evaluation, the sum of all evaluated monomials (='factors')
            # multiplied with their coefficients has to be computed
            # p = c1 * mon1 + c2 * mon2 ...
            # the value of the coefficients in the value array are only being used once
            # -> their address can be reused for storing intermediary results
            # the final evaluated value of this node must be stored at the first of its value indices
            initial_coeff_value_idx = self.value_idxs[0]
            factor = self.factors[0]
            tree_recipe = []
            op_recipe = []

            if factor is not None:
                # multiply the first coefficient with the value of the first factor
                tree_recipe += [(initial_coeff_value_idx, factor.value_idx)]
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

        self.num_ops = len(tree_recipe) + len(op_recipe)
        return tree_recipe, op_recipe

    def get_instructions(self, coeff_array: str, factor_array: str) -> str:
        """
        :return: the instructions for computing the value
        of the polynomial represented by this factorisation in C syntax
        """
        if self.has_children:
            # the own recipe is the recipe of its factorisation
            child = self.get_child()
            instr = child.get_instructions(coeff_array, factor_array)
            # only after the isntructions have been compiled the num_ops is available
            self.num_ops = child.num_ops
            return instr

        # this node has not been factorized (represents a regular polynomial)
        # for evaluation, the sum of all evaluated monomials (='factors')
        # multiplied with their coefficients has to be computed
        # p = c1 * mon1 + c2 * mon2 ...
        # the value of the coefficients in the value array are only being used once
        # -> their address can be reused for storing intermediary results
        # the final evaluated value of this node must be stored at the first of its value indices
        instr = ""
        self.num_ops = 0
        first_target = self.value_idx
        first_target_instr = f"{coeff_array}[{first_target}]"
        factor = self.factors[0]
        if factor is not None:
            # multiply the first coefficient with the value of the first factor
            source = factor.value_idx
            instr += f"{first_target_instr} *= {factor_array}[{source}];\n"
            self.num_ops += 1

        for target, factor in zip(self.value_idxs[1:], self.factors[1:]):
            if factor is not None:
                # multiply each coefficient with the value of the factor
                source = factor.value_idx
                instr += f"{coeff_array}[{target}] *= {factor_array}[{source}];\n"
                self.num_ops += 1

            # add this value to the previously computed value (stored at the first coeff idx)
            instr += f"{first_target_instr}  += {coeff_array}[{target}];\n"
            self.num_ops += 1

        return instr


class OptimalPolynomialNode(BasePolynomialNode):
    """
    A class representing a multivariate polynomial with all possible factorisation options.
    The different factorisation options are being stored sorted in a heap (self.children)
    An instance of this class is a node (=subproblem) in search tree for finding
    the optimal factorisation of a bigger multivariate polynomial
    """

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = [
        "options",
        "cost_estimate",
        "factorisation_measure",
        "fully_factorized",
    ]

    def __init__(self, exponents, *args, **kwargs):

        self.options = None
        self.cost_estimate = 0
        self.factorisation_measure = 0
        self.fully_factorized = True

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
        return self.children.get()  # return the currently most promising factorisation

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
            # the actual operation count can be computed:
            self.cost_estimate = count_num_ops_naive(self.exponents)
        else:
            # count one multiplication with the coefficients for each monomial
            heuristic = np.count_nonzero(np.any(self.exponents, axis=1))
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
            valid_option_vector = np.zeros(dim_unique_exponents.shape, dtype=BOOL_DTYPE)
            # TODO test
            compile_valid_options(
                dim,
                valid_option_vector,
                usage_vector,
                dim_unique_exponents,
                self.exponents,
            )

            for exp in dim_unique_exponents[valid_option_vector]:
                options.append((dim, exp))

        self.factorisation_measure = len(options)
        self.fully_factorized = len(options) == 0
        self.options = options

    def refine(self):
        """
        factorize further (grow tree)

        TODO improvement:
        initially compute good factorisation (with heuristic). use as upper bound for the #ops
        do not keep factorisations which have a higher or !equal! estimated #ops (<- save memory)
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

    def __init__(self, exponents, factor_container, *args, **kwargs):
        """
        :param coefficients: a numpy array column vector of doubles representing the coefficients of the monomials
        :param exponents: a numpy array matrix of unsigned integers representing the exponents of the monomials
            the ordering does not matter, but every exponent row has to be unique!
        :param rectify_input: whether to convert the input parameters into compatible numpy arrays
        :param validate_input: whether to check if the input parameters fulfill the requirements
        """
        super(OptimalFactorisationRoot, self).__init__(exponents, *args, **kwargs)

        while not self.fully_factorized:
            # TODO report current search status, verbosity option, not every loop
            # (#total factorisations done, #max depth, #open possibilities, upper bound, lower bound)
            self.refine()

        # the most promising factorisation (lowest cost estimate) is fully factorized now
        # collect all appearing factors in the best solution and assign ids (=idx in the value array) to the nodes
        # need to store which coefficient is being used where in the factorisation tree (coeff_id -> value_idx)
        coefficient_idxs = np.arange(self.num_monomials, dtype=int)
        self.compile_factors(factor_container, coefficient_idxs)

    def find_all_optimal(self):
        """
        TODO refine and collect all fully factorized optimal solutions until the next best factorisation
            has a higher cost estimate
        Attention: do not remove entire subtree, but just one realisation each time!
        TODO analyse obtained optimal factorisations
        use this data to derive better heuristics for factorizing more optimally a priori
        :return:
        """
        # optimal_factorisations = []
        raise NotImplementedError


class HeuristicFactorisationRoot(BasePolynomialNode):
    """
    Class for finding a good factorisation of a given multivariate polynomial by iteratively factoring out the most
    commonly used factor (heuristic).
    """

    def __init__(self, exponents, factor_container, *args, **kwargs):
        super(HeuristicFactorisationRoot, self).__init__(exponents, *args, **kwargs)
        # polynomial is fully factorized now
        # collect all appearing factors in the factorisation tree and assign ids (=idx in the value array) to the nodes
        # need to store which coefficient is being used where in the factorisation tree (coeff_id -> value_idx)
        coefficient_idxs = np.arange(self.num_monomials, dtype=int)
        self.compile_factors(factor_container, coefficient_idxs)


# TODO define factorisation taking the numerical stability (coefficients) into account
