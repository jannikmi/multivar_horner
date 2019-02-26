import heapq  # implementation of the heap queue algorithm, also known as the priority queue algorithm (binary tree)

import numpy as np

from .helper_fcts import get_goedel_id_of


# modified sample code from https://www.redblobgames.com/pathfinding/a-star/
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def __len__(self):
        return len(self.elements)

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, cost):
        heapq.heappush(self.elements, (cost, item))

    def pop(self):
        # remove and returns (only) the item with the lowest cost
        return heapq.heappop(self.elements)

    def get(self):
        return self.elements[0]


class PriorityQueue2D:
    def __init__(self):
        self.lvl1_heap = PriorityQueue()
        self.cost2id = {}
        self.id2heap = {}

    def __len__(self):
        length = 0
        for lvl2_heap in self.id2heap.values():
            length += len(lvl2_heap)
        return length

    def is_empty(self):
        return self.lvl1_heap.is_empty()

    def put(self, item):
        cost1 = item.cost_estimate
        cost2 = item.factorisation_measure
        try:
            # look up the heap with this lvl1 cost
            heap_id = self.cost2id[cost1]
            lvl2_heap = self.id2heap[heap_id]
        except KeyError:
            # there exists no second level heap with this cost, create an empty one
            lvl2_heap = PriorityQueue()
            heap_id = id(lvl2_heap)
            self.cost2id[cost1] = heap_id
            self.id2heap[heap_id] = lvl2_heap
            # the lvl1 heap stores the ids of the corresponding heaps
            self.lvl1_heap.put(heap_id, cost1)

        # the lvl2 heaps store the actual items
        lvl2_heap.put(item, cost2)

    def pop(self):
        cost1, heap_id = self.lvl1_heap.get()
        lvl2_heap = self.id2heap[heap_id]
        cost2, item = lvl2_heap.pop()
        if lvl2_heap.is_empty():
            # remove entries referring to an empty heap
            self.lvl1_heap.pop()
            self.id2heap.pop(heap_id)
            self.cost2id.pop(cost1)
            del lvl2_heap

        return item

    def get(self):
        # try:
        cost1, heap_id = self.lvl1_heap.get()
        lvl2_heap = self.id2heap[heap_id]
        cost2, item = lvl2_heap.get()
        return item  # return only the item without the costs
        # except IndexError:
        #     return None

    def get_all(self):
        all_items = []
        for lvl2_heap in self.id2heap.values():
            for (cost, item) in lvl2_heap.elements:
                all_items.append(item)
        return all_items


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

    def __init__(self, factor_dimension, factor_exponent, factor_id):
        self.dimension = factor_dimension
        self.exponent = factor_exponent
        self.monomial_id = factor_id
        self.value_idx = None  # initialize the idx with None to catch faulty evaluation tries

    def __str__(self):
        # variable numbering starts with 1: x_1, x_2, ...
        if self.exponent == 1:
            return 'x_{}'.format(self.dimension + 1)
        else:
            return 'x_{}^{}'.format(self.dimension + 1, self.exponent)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def num_ops(self):
        # count the number of instructions done during compute (during eval() only looks up the computed value)
        return 1

    def compute(self, x, factor_values):
        factor_values[self.value_idx] = x[self.dimension] ** self.exponent

    def get_recipe(self):
        """
        for evaluation the input value has to be either copied or exponentiated
        :return: copy_recipe, scalar_recipe, monomial_recipe
        """
        if self.exponent == 1:
            # just copy x[dim] value
            # values[target] = x[source]
            return [(self.value_idx, self.dimension)], [], []
        else:
            # instruction encoding: target, source, exponent
            # values[target] = x[source] ** exponent
            return [], [(self.value_idx, self.dimension, self.exponent)], []


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
        """
        for evaluation all scalar factors have to be multiplied (= monomial_recipe)
        :return: copy_recipe, scalar_recipe, monomial_recipe
        """
        # target = source1 * source2
        # instruction encoding: target, source1, source2
        target, source1, source2 = self.value_idx, self.factorisation_idxs[0], self.factorisation_idxs[1]
        monomial_recipe = [(target, source1, source2)]

        source1 = target  # always take the previously computed value
        for source2 in self.factorisation_idxs[2:]:
            # and multiply it with the remaining factor values
            monomial_recipe += [(target, source1, source2)]

        return [], [], monomial_recipe


class FactorContainer:
    def __init__(self, prime_array):
        self.factors = []
        self.id2idx = {}
        self.prime_array = prime_array

    def get_factor(self, property_list):

        scalar_factors = []
        monomial_id = 1
        for d, e in property_list:
            scalar_id = get_goedel_id_of(d, e, self.prime_array)
            try:
                scalar_idx = self.id2idx[scalar_id]
                scalar_factor = self.factors[scalar_idx]
            except KeyError:
                scalar_factor = ScalarFactor(d, e, scalar_id)
                self.id2idx[scalar_id] = len(self.factors)
                self.factors.append(scalar_factor)

            scalar_factors.append(scalar_factor)
            monomial_id *= scalar_id

        assert len(scalar_factors) > 0

        if len(scalar_factors) == 1:
            # the requested factor is scalar factor
            return scalar_factors[0]

        # the requested factor is a monomial consisting of multiple factors
        try:
            monomial_idx = self.id2idx[monomial_id]
            monomial_factor = self.factors[monomial_idx]
        except KeyError:
            monomial_factor = MonomialFactor(scalar_factors)
            self.id2idx[monomial_id] = len(self.factors)
            self.factors.append(monomial_factor)
        return monomial_factor

    def compile_factors(self, exponent_matrix):
        # all monomials are independent
        if np.all(np.sum(exponent_matrix, axis=0) != np.max(exponent_matrix, axis=0)):
            print(exponent_matrix)

        all_factors = []
        for mon_nr in range(exponent_matrix.shape[0]):
            exponent_vector = exponent_matrix[mon_nr, :]
            scalar_factors = []
            property_list = []
            for dim, exp in enumerate(exponent_vector):
                if exp > 0:
                    prop = (dim, exp)
                    property_list.append(prop)
                    scalar_factor = self.get_factor([prop])
                    scalar_factors.append(scalar_factor)

            if len(scalar_factors) == 0:
                factor = None
            elif len(scalar_factors) == 1:
                factor = scalar_factors[0]
            else:
                factor = self.get_factor(property_list)

            all_factors.append(factor)

        return all_factors
