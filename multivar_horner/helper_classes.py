import heapq  # implementation of the heap queue algorithm, also known as the priority queue algorithm (binary tree)
from abc import ABC, abstractmethod
from typing import List


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
        cost1, heap_id = self.lvl1_heap.get()
        lvl2_heap = self.id2heap[heap_id]
        cost2, item = lvl2_heap.get()
        return item  # return only the item without the costs

    def get_all(self):
        all_items = []
        for lvl2_heap in self.id2heap.values():
            for (_cost, item) in lvl2_heap.elements:
                all_items.append(item)
        return all_items


class AbstractFactor(ABC):
    __slots__ = ["value_idx"]

    @abstractmethod
    def __str__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def compute(self, x, value_array):
        pass

    @abstractmethod
    def num_ops(self):
        pass

    @abstractmethod
    def get_recipe(self):
        pass

    def eval(self, value_array):
        """
        looks up the computed value in the value array
        self.compute() has to be called before!
        then the value is stored at the value index in the value array
        :param value_array:
        :return: the computed value of self
        """
        return value_array[self.value_idx]


class ScalarFactor(AbstractFactor):
    """
    a factor depending on just one variable: :math:`f(x) = x_d^e`
    """

    __slots__ = ["dimension", "exponent"]

    def __init__(self, factor_dimension, factor_exponent):
        self.dimension = factor_dimension
        self.exponent = factor_exponent
        self.value_idx = None  # initialize the idx with None to catch faulty evaluation tries

    def __str__(self, factor_fmt_str="x_{dim}^{exp}", *args, **kwargs):
        # NOTE: variable numbering starts with 1: x_1, x_2, ...
        # if self.exponent == 1:
        #     return 'x_{}'.format(self.dimension + 1)

        return factor_fmt_str.format(**{"dim": self.dimension + 1, "exp": self.exponent})

    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    @property
    def num_ops(self) -> int:
        # the number of instructions required for computing the value
        if self.exponent > 1:
            return 2
        else:
            return 1

    def compute(self, x, value_array):
        value_array[self.value_idx] = x[self.dimension] ** self.exponent

    def get_recipe(self):
        """
        for evaluation the input value has to be either copied or exponentiated
        :return: copy_recipe, scalar_recipe
        """
        if self.exponent == 1:
            # just copy x[dim] value
            # values[target] = x[source]
            return [(self.value_idx, self.dimension)], []
        else:
            # instruction encoding: target, source, exponent
            # values[target] = x[source] ** exponent
            return [], [(self.value_idx, self.dimension, self.exponent)]

    def get_instructions(self, array_name: str) -> str:
        """
        :return: the instructions for computing the value of this factor in C syntax
        """
        dim = self.dimension
        exp = self.exponent
        idx = self.value_idx
        instr = f"x[{dim}]"
        if exp > 1:
            instr = f"pow({instr},{exp})"
        instr = f"{array_name}[{idx}] = {instr};\n"
        return instr


class MonomialFactor(AbstractFactor):
    """
    a factor ('monomial') consisting of a product of scalar factors:
    :math:`m(x) = x_i^j * ... * x_k^l`

    Parameters
    ----------

    :param scalar_factors: a list of scalar factors the monomial consists of

    Attributes
    ----------

    factorisation_idxs: the indices of the values of all scalar 'sub' factors in the value array of the polynomial.
        cannot be set at construction time, because all scalar factors need to receive their index first,
        but not all required scalar factors might exist
    """

    __slots__ = ["scalar_factors", "factorisation_idxs"]

    def __init__(self, scalar_factors: List["ScalarFactor"]):

        # assert len(scalar_factors) > 1, 'a monomial must consist of at least two scalar factors' # DEBUG
        self.scalar_factors = scalar_factors
        self.value_idx = None  # initialize the idx with None to catch faulty evaluation tries

    def __str__(self):
        return " ".join([f.__str__() for f in self.scalar_factors])

    def __repr__(self):
        return self.__str__()

    @property
    def num_ops(self):
        # count the number of instructions done during compute (during eval() only looks up the computed value)
        return len(self.scalar_factors) - 1  # product of all scalar factors

    def compute(self, x, value_array):
        # IMPORTANT: compute() of all the sub factors has had to be called before!
        value = value_array[self.factorisation_idxs[0]]
        for idx in self.factorisation_idxs[1:]:
            value *= value_array[idx]

        value_array[self.value_idx] = value

    def get_recipe(self):
        """
        for evaluation all scalar factors have to be multiplied (= monomial_recipe)
        :return: monomial_recipe
        """
        # target = source1 * source2
        # instruction encoding: target, source1, source2
        target, source1, source2 = (
            self.value_idx,
            self.factorisation_idxs[0],
            self.factorisation_idxs[1],
        )
        monomial_recipe = [(target, source1, source2)]

        source1 = target  # always take the previously computed value
        for source2 in self.factorisation_idxs[2:]:
            # and multiply it with the remaining factor values
            monomial_recipe += [(target, source1, source2)]

        return monomial_recipe

    def get_instructions(self, array_name: str) -> str:
        """
        :return: the instructions for computing the value of this factor in C syntax
        """
        idx = self.value_idx
        # target = source1 * source2
        target = f"{array_name}[{idx}]"

        # first instruction: copy the value of the first scalar factor
        source = self.factorisation_idxs[0]
        instr = f"{target} = {array_name}[{source}];\n"
        # always take the previously computed value
        for source in self.factorisation_idxs[1:]:
            # and multiply it with the value of the next scalar factor
            instr += f"{target} *= {array_name}[{source}];\n"
        return instr


class FactorContainer:
    """
    a class for storing and reusing all factors appearing in a factorisation
    """

    def __init__(self):
        self.scalar_factors: List[ScalarFactor] = []
        self.monomial_factors: List[MonomialFactor] = []
        self.property2idx_scalar = {}
        self.property2idx_monomial = {}

    def __len__(self):
        """
        :return: the total amount of factors
        """
        return len(self.scalar_factors) + len(self.monomial_factors)

    def get_factor(self, property_list):
        """
        creates and stores objects of all required (sub-)factors if necessary
        :param property_list: a list of dimension and exponent tuples [(d1,e1), (d2,e2)...]
            representing the scalar factors of a monomial
        :return: the object representing the factor with the given properties
        """

        assert len(property_list) > 0  # TODO DEBUG
        # create all required scalar factors
        scalar_factors = []
        for prop in property_list:
            d, e = prop
            try:
                scalar_idx = self.property2idx_scalar[prop]
                scalar_factor = self.scalar_factors[scalar_idx]
            except KeyError:
                scalar_factor = ScalarFactor(d, e)
                self.property2idx_scalar[prop] = len(self.scalar_factors)
                self.scalar_factors.append(scalar_factor)

            scalar_factors.append(scalar_factor)

        assert len(scalar_factors) == len(property_list)  # TODO DEBUG

        if len(scalar_factors) == 1:
            # the requested factor is a scalar factor
            return scalar_factors[0]

        # the requested factor is a monomial consisting of multiple factors
        monomial_id = tuple(property_list)
        try:
            monomial_idx = self.property2idx_monomial[monomial_id]
            monomial_factor = self.monomial_factors[monomial_idx]
        except KeyError:
            monomial_factor = MonomialFactor(scalar_factors)
            self.property2idx_monomial[monomial_id] = len(self.monomial_factors)
            self.monomial_factors.append(monomial_factor)

        return monomial_factor

    def get_factors(self, exponent_matrix):
        """
        :param exponent_matrix:
        :return: a list of all factors represented by the exponent_matrix with an identical ordering
        """

        all_factors = []
        # TODO optimise, modularise into functions: monomial factor, scalar factor... use numpy routines!?
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
                factor = None  # monomial consists of no factors
            elif len(scalar_factors) == 1:
                factor = scalar_factors[0]  # monomial consists of a single scalar factor
            else:
                factor = self.get_factor(property_list)  # monomial consists of a multiple scalar factors

            all_factors.append(factor)

        return all_factors
