# -*- coding:utf-8 -*-
"""
    TODO
    POSSIBLE IMPROVEMENTS:

    MultivarPoly:

    - also make use of the concept of 'recipes' for efficiently evaluating the polynomial
        skipping the most unnecessary operations
    - add option to skip this optimisation

    HornerMultivarPoly:


    - optimise factor evaluation (save instructions, 'factor factorisation'):
    a monomial factor consists of scalar factors and in turn some monomial factors consist of other monomial factors
    -> the result of evaluating a factor can be reused for evaluating other factors containing it
    -> find the optimal 'factorisation' of the factors themselves
    -> set the factorisation_idxs of each factor in total_degree to link the evaluation appropriately
    idea:
        choose  'Goedel IDs' as the monomial factor ids
        then the id of a monomial is the product of the ids of its scalar factors
        find the highest possible divisor among all factor ids
        (corresponds to the 'largest' factor included in the monomial)
        this leads to a minimal factorisation for evaluating the monomial values quickly

    - add option to skip this optimisation to save build time

    - optimise space requirement:
     after building a factorisation tree for the factors themselves,
     then use its structure to cleverly reuse storage space
     -> use compiler construction theory: minimal assembler register assignment, 'graph coloring'...

    - optimise 'copy recipe': avoid copy operations for accessing values of x
        problem: inserting x into the value array causes operations as well and
            complicates address assigment and recipe compilation

    -  when the polynomial does not depend on all variables, build a wrapper to maintain the same "interface"
        but internally reduce the dimensionality, this reduced the size of the numpy arrays -> speed, storage benefit

    - the evaluation of subtrees is independent and could theoretically be done in parallel
        probably not worth the effort. more reasonable to just evaluate multiple polynomials in parallel
"""
import abc
import ctypes
import pickle
import shutil
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np

from multivar_horner.factorisation_classes import (
    BasePolynomialNode,
    HeuristicFactorisationRoot,
    OptimalFactorisationRoot,
)
from multivar_horner.global_settings import (
    BOOL_DTYPE,
    DEBUG,
    DEFAULT_PICKLE_FILE_NAME,
    FLOAT_DTYPE,
    TYPE_1D_FLOAT,
    TYPE_2D_INT,
    UINT_DTYPE,
)
from multivar_horner.helper_classes import FactorContainer, MonomialFactor, ScalarFactor
from multivar_horner.helper_fcts import (
    rectify_coefficients,
    rectify_construction_parameters,
    rectify_query_point,
    validate_coefficients,
    validate_construction_parameters,
    validate_query_point,
)
from multivar_horner.helpers_fcts_numba import count_num_ops, eval_recipe, naive_eval

PATH2C_FILES = Path(__file__).parent / "__pycache__"


# is not a helper function to make it an importable part of the package
def load_pickle(path: str = DEFAULT_PICKLE_FILE_NAME) -> "AbstractPolynomial":
    print('importing polynomial from file "{}"'.format(path))
    with open(path, "rb") as f:
        return pickle.load(f)


# TODO properties: num ops...

# TODO docstring attributes in parent class, automatic inheritance in sphinx autodoc api docu
#   -> prevent duplicate docstring in MultivarPolynomial
class AbstractPolynomial(ABC):
    """an abstract class for representing a multivariate polynomial"""

    # prevent dynamic attribute assignment (-> safe memory)
    # FIXME: creates duplicate entries in Sphinx autodoc
    __slots__ = [
        "compute_representation",
        "coefficients",
        "euclidean_degree",
        "exponents",
        "num_monomials",
        "num_ops",
        "dim",
        "maximal_degree",
        "total_degree",
        "unused_variables",
        "representation",
    ]

    def __init__(
        self,
        coefficients: TYPE_1D_FLOAT,
        exponents: TYPE_2D_INT,
        rectify_input: bool = False,
        compute_representation: bool = False,
    ):

        if rectify_input:
            coefficients, exponents = rectify_construction_parameters(coefficients, exponents)

        validate_construction_parameters(coefficients, exponents)

        self.coefficients: np.ndarray = coefficients
        self.exponents: np.ndarray = exponents
        self.compute_representation: bool = compute_representation

        self.num_monomials: int = self.exponents.shape[0]
        self.dim: int = self.exponents.shape[1]
        self.unused_variables = np.where(~np.any(self.exponents.astype(BOOL_DTYPE), axis=1))[0]
        self.total_degree: int = np.max(np.sum(self.exponents, axis=0))
        self.euclidean_degree: float = np.max(np.linalg.norm(self.exponents, ord=2, axis=0))
        self.maximal_degree: int = np.max(self.exponents)

        self.num_ops: int = 0
        self.representation: str

    def __str__(self):
        return self.representation

    def __repr__(self):
        return self.representation

    def __call__(self, *args, **kwargs) -> float:
        return self.eval(*args, **kwargs)

    def get_num_ops(self) -> int:
        return self.num_ops

    @abc.abstractmethod
    def compute_string_representation(self, *args, **kwargs) -> str:
        """computes a string representation of the polynomial and sets self.representation

        Returns:
            a string representing this polynomial instance
        """
        ...

    def export_pickle(self, path: str = DEFAULT_PICKLE_FILE_NAME):
        print('storing polynomial in file "{}" ...'.format(path))
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print("...done.\n")

    # TODO test
    def get_partial_derivative(self, i: int, *args, **kwargs) -> "AbstractPolynomial":
        """retrieves a partial derivative

        Note:
            all given additional arguments will be passed to the constructor of the derivative polynomial

        Args:
            i: dimension to derive with respect to.
                ATTENTION: dimension counting starts with 1 (i >= 1)

        Returns:
            the partial derivative of this polynomial wrt. the i-th dimension
        """

        assert 0 < i <= self.dim, "invalid dimension i given"
        coord_index = i - 1
        # IMPORTANT: do not modify the stored coefficient and exponent arrays of self!
        # set all the coefficients not depending on the i-th coordinate to 0
        # this simply means not adding them to the list of coefficients of the new polynomial class instance
        active_rows = np.where(self.exponents[:, coord_index] >= 1)[0]

        new_coefficients = self.coefficients[active_rows]
        new_exponents = self.exponents[active_rows, :]

        if DEBUG:  # TODO move to tests
            assert new_coefficients.shape[0] == new_exponents.shape[0]
            assert new_coefficients.shape[1] == 1 and len(new_coefficients.shape) == 2
            assert new_exponents.shape[1] == self.dim and len(new_exponents.shape) == 2

        # multiply the coefficients with the exponent of the i-th coordinate
        # f(x) = a x^b
        # f'(x) = ab x^(b-1)
        new_coefficients = np.multiply(new_coefficients.flatten(), new_exponents[:, coord_index])
        new_coefficients = new_coefficients.reshape(-1, 1)

        # reduce the the exponent of the i-th coordinate by 1
        new_exponents[:, coord_index] -= 1

        # must call the proper constructor method also for inherited classes
        return self.__class__(new_coefficients, new_exponents, *args, **kwargs)

    def get_gradient(self, *args, **kwargs) -> List["AbstractPolynomial"]:
        """
        Note:
            all arguments will be passed to the constructor of the derivative polynomials

        Returns:
             the list of all partial derivatives
        """
        return [self.get_partial_derivative(i, *args, **kwargs) for i in range(1, self.dim + 1)]

    def change_coefficients(
        self,
        coefficients: TYPE_1D_FLOAT,
        rectify_input: bool = False,
        compute_representation: bool = False,
        in_place: bool = False,
        *args,
        **kwargs,
    ) -> "AbstractPolynomial":

        if rectify_input:
            coefficients = rectify_coefficients(coefficients)

        validate_coefficients(coefficients)

        assert len(coefficients) == self.num_monomials

        if in_place:
            updated_poly = self
        else:
            updated_poly = deepcopy(self)

        updated_poly.compute_representation = compute_representation

        updated_poly.coefficients = coefficients
        updated_poly.compute_string_representation(*args, **kwargs)
        return updated_poly

    @abstractmethod
    def eval(
        self,
        x: TYPE_1D_FLOAT,
        rectify_input: bool = False,
    ) -> float:
        pass


class MultivarPolynomial(AbstractPolynomial):
    """a representation of a multivariate polynomial in 'canonical form' (without any factorisation)

    Args:
        coefficients: ndarray of floats with shape (N,1)
            representing the coefficients of the monomials
            NOTE: coefficients with value 0 and 1 are allowed and will not affect the internal representation,
            because coefficients must be replaceable
        exponents: ndarray of unsigned integers with shape (N,m)
            representing the exponents of the monomials
            where m is the number of dimensions (self.dim),
            the ordering corresponds to the ordering of the coefficients, every exponent row has to be unique!
        rectify_input: bool, default=False
            whether to convert coefficients and exponents into compatible numpy arrays
            with this set to True, coefficients and exponents can be given in standard python arrays
        compute_representation: bool, default=False
            whether to compute a string representation of the polynomial

    Attributes:
        num_monomials: the amount of coefficients/monomials N of the polynomial
        dim: the dimensionality m of the polynomial
            NOTE: the polynomial needs not to actually depend on all m dimensions
        unused_variables: the dimensions the polynomial does not depend on
        num_ops: the amount of mathematical operations required to evaluate the polynomial in this representation
        representation: a human readable string visualising the polynomial representation

    Raises:
        TypeError: if coefficients or exponents are not given as ndarrays
            of the required dtype
        ValueError: if coefficients or exponents do not have the required shape or
            do not fulfill the other requirements or ``rectify_input=True`` and there are negative exponents
    """

    def __init__(
        self,
        coefficients: TYPE_1D_FLOAT,
        exponents: TYPE_2D_INT,
        rectify_input: bool = False,
        compute_representation: bool = False,
        *args,
        **kwargs,
    ):

        super(MultivarPolynomial, self).__init__(coefficients, exponents, rectify_input, compute_representation)

        # NOTE: count the number of multiplications of the representation
        # not the actual amount of operations required by the naive evaluation with numpy arrays
        self.num_ops = count_num_ops(self.exponents)
        self.compute_string_representation(*args, **kwargs)

    def compute_string_representation(
        self,
        coeff_fmt_str: str = "{:.2}",
        factor_fmt_str: str = "x_{dim}^{exp}",
        *args,
        **kwargs,
    ) -> str:
        representation = "[#ops={}] p(x)".format(self.num_ops)
        if self.compute_representation:
            representation += " = "
            monomials = []
            for i, exp_vect in enumerate(self.exponents):
                monomial = [coeff_fmt_str.format(self.coefficients[i, 0])]
                for dim, exp in enumerate(exp_vect):
                    # show all operations, even 1 * x_i^0
                    monomial.append(factor_fmt_str.format(**{"dim": dim + 1, "exp": exp}))

                monomials.append(" ".join(monomial))

            representation += " + ".join(monomials)

        self.representation = representation
        return self.representation

    def eval(self, x: TYPE_1D_FLOAT, rectify_input: bool = False) -> float:
        """computes the value of the polynomial at query point x

        makes use of fast ``Numba`` just in time compiled functions

        Args:
            x: ndarray of floats with shape = [self.dim] representing the query point
            rectify_input: bool, default=False
                whether to convert coefficients and exponents into compatible numpy arrays
                with this set to True, the query point x can be given in standard python arrays

        Returns:
             the value of the polynomial at point x

        Raises:
            TypeError: if x is not given as ndarray of dtype float
            ValueError: if x does not have the shape ``[self.dim]``
        """
        if rectify_input:
            x = rectify_query_point(x)
        validate_query_point(x, self.dim)

        return naive_eval(x, self.coefficients.flatten(), self.exponents)


class HornerMultivarPolynomial(AbstractPolynomial):
    """a representation of a multivariate polynomial using Horner factorisation

    the polynomial is being evaluated by fast just in time compiled functions
    using precompiled "recipes" of instructions.

    Args:
        coefficients: ndarray of floats with shape (N,1)
            representing the coefficients of the monomials
            NOTE: coefficients with value 0 and 1 are allowed and will not affect the internal representation,
            because coefficients must be replaceable
        exponents: ndarray of unsigned integers with shape (N,m)
            representing the exponents of the monomials
            where m is the number of dimensions (self.dim),
            the ordering corresponds to the ordering of the coefficients, every exponent row has to be unique!
        rectify_input: bool, default=False
            whether to convert coefficients and exponents into compatible numpy arrays
            with this set to True, coefficients and exponents can be given in standard python arrays
        compute_representation: bool, default=False
            whether to compute a string representation of the polynomial

        keep_tree: whether the factorisation tree object should be kept in memory after finishing factorisation
        find_optimal: whether a search over all possible factorisations should be done in total_degree to find
            an optimal factorisation in the sense of a minimal amount required operations for evaluation


    Attributes:
        num_monomials: the amount of coefficients/monomials N of the polynomial
        dim: the dimensionality m of the polynomial
            NOTE: the polynomial needs not to actually depend on all m dimensions
        unused_variables: the dimensions the polynomial does not depend on
        num_ops: the amount of mathematical operations required to evaluate the polynomial in this representation
        representation: a human readable string visualising the polynomial representation

        total_degree: the usual notion of degree for a polynomial.
            = the maximum sum of exponents in any of its monomials
            = the maximum l_1-norm of the exponent vectors of all monomials
            in contrast to 1D polynomials, different concepts of degrees exist for polynomials in multiple dimensions.
            following the naming in [1] L. Trefethen, “Multivariate polynomial approximation in the hypercube”,
            Proceedings of the American Mathematical Society, vol. 145, no. 11, pp. 4837–4844, 2017.
        euclidean_degree: the maximum l_2-norm of the exponent vectors of all monomials.
            NOTE: this is not in general an integer
        maximal_degree: the largest exponent in any of its monomials
            = the maximum l_infinity-norm of the exponent vectors of all monomials

        factorisation_tree: the object oriented, recursive data structure representing the factorisation
            (only if keep_tree=True)
        factor_container: the object containing all (unique) factors of the factorisation (only if keep_tree=True)
        root_value_idx: the index in the value array where the value of this polynomial
            (= root of the factorisation_tree) will be stored
        value_array_length: the amount of addresses (storage) required to evaluate the polynomial.
            for evaluating the polynomial in tree form intermediary results have to be stored in a value array.
            the value array begins with the coefficients of the polynomial.
            (without further optimisation) every factor requires its own address.

        copy_recipe: ndarray encoding the operations required to evaluate all scalar factors with exponent 1
        scalar_recipe: ndarray encoding the operations required to evaluate all remaining scalar factors
        monomial_recipe: ndarray encoding the operations required to evaluate all monomial factors
        tree_recipe: ndarray encoding the addresses required to evaluate
            the polynomial values of the factorisation_tree.
        tree_ops: ndarray encoding the type of operation required to evaluate
            the polynomial values of the factorisation_tree.
            encoded as a boolean ndarray separate from tree_recipe,
            since only the two operations ADD & MUL need to be encoded.

    Raises:
        TypeError: if coefficients or exponents are not given as ndarrays
            of the required dtype
        ValueError: if coefficients or exponents do not have the required shape or
            do not fulfill the other requirements
    """

    # __slots__ declared in parents are available in child classes. However, child subclasses will get a __dict__
    # and __weakref__ unless they also define __slots__ (which should only contain names of any additional slots).
    # FIXME: creates duplicate entries in Sphinx autodoc
    __slots__ = [
        "copy_recipe",
        "factorisation_tree",
        "factor_container",
        "monomial_recipe",
        "root_value_idx",
        "tree_recipe",
        "tree_ops",
        "scalar_recipe",
        "value_array_length",
        "find_optimal",
        "keep_tree",
        "_hash_val",
        "use_c_eval",
        "num_ops",
    ]

    def __init__(
        self,
        coefficients,
        exponents,
        rectify_input=False,
        keep_tree=False,
        compute_representation=False,
        find_optimal=False,
        *args,
        **kwargs,
    ):
        super(HornerMultivarPolynomial, self).__init__(coefficients, exponents, rectify_input, compute_representation)
        self.find_optimal: bool = find_optimal
        self.keep_tree: bool = keep_tree
        self.value_array_length = None
        self.representation = None
        self._hash_val: int

        self._configure_evaluation()
        self.compute_string_representation(*args, **kwargs)

        if not self.keep_tree:
            try:
                del self.factorisation_tree
                del self.factor_container
            except AttributeError:
                pass

    def _configure_evaluation(self):
        # by default use faster C evaluation
        self.use_c_eval: bool = True
        try:
            self._compile_c_file()
        except ValueError as exc:
            print(exc)
            # if C compilation does not work, use recipe (numpy+Numba based) evaluation as a fallback
            self.use_c_eval = False
            self._compile_recipes()

    def _compute_factorisation(self):
        print("computing factorisation...")
        # NOTE: do NOT automatically create all scalar factors with exponent 1
        # (they might be unused, since the polynomial must not actually depend on all variables)
        self.factor_container = FactorContainer()
        if self.find_optimal:
            root_class = OptimalFactorisationRoot
        else:
            root_class = HeuristicFactorisationRoot
        self.factorisation_tree: BasePolynomialNode = root_class(self.exponents, self.factor_container)
        self.root_value_idx = self.factorisation_tree.value_idx
        self.value_array_length = self.num_monomials + len(self.factor_container)

    def __hash__(self):
        """
        compare polynomials (including their factorisation) based on their properties
        NOTE: coefficients can be changed
        without affecting the fundamental properties of the polynomial (factorisation)
        NOTE: optimal factorisations might be different from the ones found with the default approach

        Returns: an integer encoding the fundamental properties of the polynomial including its factorisation
        """
        try:
            return self._hash_val
        except AttributeError:
            props = (self.dim, self.num_monomials, self.find_optimal, *self.exponents.flatten())
            self._hash_val = hash(props)
        return self._hash_val

    def __eq__(self, other):
        """
        Returns: true when ``other`` is of the same class and has equal properties (encoded by hash)
        """
        if not isinstance(other, self.__class__):
            return False
        # we consider polynomials equal when they share their properties (-> hash)
        return hash(self) == hash(other)

    def compute_string_representation(
        self,
        coeff_fmt_str: str = "{:.2}",
        factor_fmt_str: str = "x_{dim}^{exp}",
        *args,
        **kwargs,
    ) -> str:
        repre = ""
        if not self.compute_representation:
            self.representation = repre
            return repre
        repre = f"[#ops={self.num_ops}] p(x)"
        try:
            repre += " = " + self.factorisation_tree.get_string_representation(
                self.coefficients, coeff_fmt_str, factor_fmt_str
            )
            # exponentiation with 1 won't cause an operation in this representation
            # but are present in the string representation due to string formatting restrictions
            # -> they should not be displayed (misleading)
            repre = repre.replace("^1", "")  # <- workaround for the default string format
        except AttributeError:
            pass  # self.factorisation_tree does not exist

        self.representation = repre
        return self.representation

    def _compile_recipes(self):
        """encode all instructions needed for evaluating the polynomial in 'recipes'

        recipes are represented as numpy ndarrays (cf. assembler instructions)

        -> acquire a data structure representing the factorisation tree
        -> avoid recursion and function call overhead during evaluation
        -> enables the use of jit compiled functions

        the factor container must now contain all unique factors used in the chosen factorisation
        during evaluation of a polynomial the values of all the factors are needed at least once
        -> compute the values of all factors once and store them
        -> store a pointer to the computed value for every factor ('value index' = address in the value array)
        this is required for compiling evaluation instructions depending on the factor values
        monomial factors exist only if their value is required during the evaluation of the parent polynomial
        scalar factors exist only if their value is required during the evaluation of existing monomial factors
        (scalar factors can be 'standalone' factors as well)
        -> values must not be overwritten (reusing addresses), because they might be needed again by another factor
        -> (without further optimisation) each factor requires its own space in the value array

        Returns:
            the compiled recipes (numpy ndarrays)
        """
        # TODO pickle. check if present
        # for compiling the recipes the factorisation must be computed first
        self._compute_factorisation()
        # the values of the factors are being stored after the coefficients
        # start the address assignment with the correct offset
        value_idx = self.num_monomials

        # compile the recipes for computing the value of all factors
        copy_recipe = []  # skip computing factors with exp 1, just copy x value
        scalar_recipe = []
        monomial_recipe = []
        self.num_ops = 0

        # count the amount of multiplications encoded by the recipes
        # NOTE: count exponentiations as exponent-1 multiplications, e.g. x^3 <-> 2 operations

        # -> IMPORTANT: value idx assignment must happen first for the scalar factors
        for scalar_factor in self.factor_container.scalar_factors:
            scalar_factor.value_idx = value_idx
            value_idx += 1
            copy_instr, scalar_instr = scalar_factor.get_recipe()
            copy_recipe += copy_instr
            scalar_recipe += scalar_instr
            if len(scalar_instr) > 0:
                exponent = scalar_instr[0][2]
                self.num_ops += exponent - 1

        for monomial_factor in self.factor_container.monomial_factors:
            monomial_factor.value_idx = value_idx
            value_idx += 1
            monomial_factor.factorisation_idxs = [f.value_idx for f in monomial_factor.scalar_factors]
            monomial_instr = monomial_factor.get_recipe()
            monomial_recipe += monomial_instr
            self.num_ops += 1  # every monomial instruction encodes one multiplication

        # compile the recipe for evaluating the Horner factorisation tree
        tree_recipe, tree_ops = self.factorisation_tree.get_recipe()
        # convert the recipes into the data types expected by the jit compiled functions
        # and store them
        tree_ops = np.array(tree_ops, dtype=BOOL_DTYPE)
        self.num_ops += len(tree_ops) - np.count_nonzero(tree_ops)  # every 0/False encodes a multiplication
        self.tree_ops = tree_ops

        self.copy_recipe = np.array(copy_recipe, dtype=UINT_DTYPE).reshape((-1, 2))
        self.scalar_recipe = np.array(scalar_recipe, dtype=UINT_DTYPE).reshape((-1, 3))
        self.monomial_recipe = np.array(monomial_recipe, dtype=UINT_DTYPE).reshape((-1, 3))
        self.tree_recipe = np.array(tree_recipe, dtype=UINT_DTYPE).reshape((-1, 2))

    def eval(self, x: TYPE_1D_FLOAT, rectify_input: bool = False, validate_input: bool = False) -> float:
        """computes the value of the polynomial at query point x

        either uses C  or numpy+Numba evaluation

        Args:
            x: ndarray of floats with shape = [self.dim] representing the query point
            rectify_input: whether to convert coefficients and exponents into compatible numpy arrays
                with this set to True, the query point x can be given in standard python arrays

        Returns:
             the value of the polynomial at point x

        Raises:
            TypeError: if x is not given as ndarray of dtype float
            ValueError: if x does not have the shape ``[self.dim]``
        """
        if rectify_input:
            x = rectify_query_point(x)
        if self.use_c_eval:
            if validate_input:
                validate_query_point(x, self.dim)
            return self._eval_c(x)

        # use numpy+Numba recipe evaluation as fallback
        # input type and shape should always be validated.
        #   otherwise the numba jit compiled functions may fail with cryptic error messages
        validate_query_point(x, self.dim)
        return self._eval_recipe(x)

    def _eval_recipe(self, x: TYPE_1D_FLOAT) -> float:
        """computes the value of the polynomial at query point x

        makes use of fast ``Numba`` just in time compiled functions
        """
        value_array = np.empty(self.value_array_length, dtype=FLOAT_DTYPE)
        # the coefficients are being stored at the beginning of the value array
        # TODO remove flatten, always store coefficients as a 1D array (also for horner fact.)?!
        #   also in MultivarPolynomial.eval()
        value_array[: self.num_monomials] = self.coefficients.flatten()
        return eval_recipe(
            x,
            value_array,
            self.copy_recipe,
            self.scalar_recipe,
            self.monomial_recipe,
            self.tree_recipe,
            self.tree_ops,
            self.root_value_idx,
        )

    def _eval_c(self, x: TYPE_1D_FLOAT) -> float:
        dim = self.dim
        compiled_file = self.c_file_compiled
        if not compiled_file.exists():
            raise ValueError(f"missing compiled C file: {compiled_file}")
        cdll = ctypes.CDLL(str(compiled_file))
        double = ctypes.c_double
        type_x = double * dim
        type_coeffs = double * self.num_monomials
        x_typed = type_x(*x)
        coeffs = self.coefficients.flatten()
        coeffs_typed = type_coeffs(*coeffs)
        func_name = "eval"
        function = getattr(cdll, func_name)
        function.argtypes = (type_x, type_coeffs)
        function.restype = double
        p_x = function(x_typed, coeffs_typed)
        return p_x

    def get_c_file_name(self, ending: str = ".c") -> str:
        return f"eval_poly_{hash(self)}{ending}"

    @property
    def c_file(self) -> Path:
        return PATH2C_FILES / self.get_c_file_name()

    @property
    def c_file_compiled(self):
        return PATH2C_FILES / self.get_c_file_name(ending=".so")

    def _write_c_file(self):
        path_out = self.c_file
        if path_out.exists():
            return
        # for compiling the instructions the factorisation must be computed first
        self._compute_factorisation()
        print("compiling C instructions ...")
        try:
            tree = self.factorisation_tree
            factor_container = self.factor_container
        except AttributeError:
            raise ValueError(
                "need a stored factorisation tree, but the reference to it has already been deleted. "
                "initialise class with 'keep_tree=True`"
            )
        f_type = "double"
        # scalar and monomial factors together
        factor_array = "f"
        coeff_array = "c"
        func_name = "eval"
        instr = "#include <math.h>\n"

        nr_coeffs = self.num_monomials
        nr_dims = self.dim
        self.num_ops = 0

        func_def = f"{f_type} {func_name}({f_type} x[{nr_dims}], {f_type} {coeff_array}[{nr_coeffs}])"
        # declare function
        instr += f"{func_def};\n"
        instr += f"{func_def}"
        instr += "{\n"

        # NOTE: the order of computing the values important: scalar factors, monomial factors, monomials,...
        scalar: ScalarFactor
        monomial: MonomialFactor
        scalars = factor_container.scalar_factors
        monomials = factor_container.monomial_factors
        nr_factors = len(factor_container)

        # initialise arrays to capture the intermediary results of factor evaluation to 0
        if nr_factors > 0:
            instr += f"{f_type} {factor_array}[{nr_factors}]= {{0.0}};\n"
        idx: int = 0
        # set the index of each factor in order to remember it for later reference
        # ATTENTION: different from the ones used in the recipes!
        for idx, scalar in enumerate(scalars):
            scalar.value_idx = idx
            instr += scalar.get_instructions(factor_array)
            self.num_ops += scalar.num_ops

        for monomial in monomials:
            idx += 1
            monomial.value_idx = idx
            monomial.factorisation_idxs = [f.value_idx for f in monomial.scalar_factors]
            instr += monomial.get_instructions(factor_array)
            self.num_ops += monomial.num_ops

        # compile the instructions for evaluating the Horner factorisation tree
        instr += tree.get_instructions(coeff_array, factor_array)
        self.num_ops += tree.num_ops

        # return the entry of the coefficient array corresponding to the root node of the factorisation tree
        root_idx = tree.value_idx
        instr += f"return {coeff_array}[{root_idx}];\n"
        instr += "}\n"
        print(f"writing in file {path_out}")
        with open(path_out, "w") as c_file:
            c_file.write(instr)

    def get_c_instructions(self) -> str:
        path = self.c_file
        print(f"reading C instructions from file {path}")
        with open(path) as c_file:
            instr = c_file.read()
        return instr

    def _compile_c_file(self):
        path_out = self.c_file_compiled
        if path_out.exists():
            print(f"using existing compiled C file {path_out}")
            return

        compiler = "gcc"
        if shutil.which(compiler) is None:
            compiler = "cc"
            if shutil.which(compiler) is None:
                raise ValueError("compiler (`gcc` or `cc`) not present. install one first.")

        self._write_c_file()
        path_in = self.c_file
        print(f"compiling to file {path_out}")
        cmd = [compiler, "-shared", "-o", str(path_out), "-fPIC", str(path_in)]
        subprocess.call(cmd)
        if not path_out.exists():
            raise ValueError(f"expected compiled file missing: {path_out}")
