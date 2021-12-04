import ctypes
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

from multivar_horner.classes.abstract_poly import AbstractPolynomial
from multivar_horner.classes.factorisation import (
    BasePolynomialNode,
    HeuristicFactorisationRoot,
    OptimalFactorisationRoot,
)
from multivar_horner.classes.helpers import FactorContainer, MonomialFactor, ScalarFactor
from multivar_horner.global_settings import BOOL_DTYPE, FLOAT_DTYPE, TYPE_1D_FLOAT, UINT_DTYPE
from multivar_horner.helper_fcts import rectify_query_point, validate_query_point
from multivar_horner.helpers_fcts_numba import eval_recipe

PATH2CACHE = Path(__file__).parent.parent / "__pycache__"


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
    __slots__ = [
        "factorisation_tree",
        "factor_container",
        "root_value_idx",
        "value_array_length",
        "find_optimal",
        "keep_tree",
        "_hash_val",
        "use_c_eval",
        "recipe",
    ]

    # FIXME: creates duplicate entries in Sphinx autodoc
    def __init__(
        self,
        coefficients,
        exponents,
        rectify_input: bool = False,
        compute_representation: bool = False,
        verbose: bool = False,
        keep_tree: bool = False,
        find_optimal: bool = False,
        store_c_instr: bool = False,
        store_numpy_recipe: bool = False,
        *args,
        **kwargs,
    ):
        super(HornerMultivarPolynomial, self).__init__(
            coefficients, exponents, rectify_input, compute_representation, verbose
        )
        self.find_optimal: bool = find_optimal
        self.keep_tree: bool = keep_tree
        self.value_array_length: int
        self._hash_val: int
        self.recipe: Tuple

        self._configure_evaluation(store_c_instr, store_numpy_recipe)
        self.compute_string_representation(*args, **kwargs)

        if not self.keep_tree:
            try:
                del self.factorisation_tree
                del self.factor_container
            except AttributeError:
                pass

    def _configure_evaluation(
        self,
        store_c_instr: bool = False,
        store_numpy_recipe: bool = False,
    ):
        self.use_c_eval: bool = True
        if store_c_instr:
            # force storing a C file
            self._compile_c_file()
        if store_numpy_recipe:
            # force storing the numpy+Numba recipe
            self._compile_recipes()

        if not (store_c_instr or store_numpy_recipe):
            # by default use faster C evaluation -> compile C file
            # if the compilers are not installed this will raise a `ValueError`
            try:
                self._compile_c_file()
            except ValueError as exc:
                self.print(exc)
                # if C compilation does not work, use recipe (numpy+Numba based) evaluation as a fallback
                self.use_c_eval = False
                self._compile_recipes()

    def _compute_factorisation(self):
        # do not compute the factorisation when it is already present
        try:
            self.factorisation_tree
            return
        except AttributeError:
            pass

        self.print("computing factorisation...")
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
        # reuse pickled recipe if present
        try:
            self._load_recipe()
            return
        except ValueError:
            pass

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

        copy_recipe = np.array(copy_recipe, dtype=UINT_DTYPE).reshape((-1, 2))
        scalar_recipe = np.array(scalar_recipe, dtype=UINT_DTYPE).reshape((-1, 3))
        monomial_recipe = np.array(monomial_recipe, dtype=UINT_DTYPE).reshape((-1, 3))
        tree_recipe = np.array(tree_recipe, dtype=UINT_DTYPE).reshape((-1, 2))
        # IMPORTANT: strict ordering required!
        self.recipe = (copy_recipe, scalar_recipe, monomial_recipe, tree_recipe, tree_ops)
        self._pickle_recipe()

    def _pickle_recipe(self):
        path = self.recipe_file
        self.print(f'storing recipe in file "{path}"')
        pickle_obj = (self.recipe, self.num_ops, self.value_array_length, self.root_value_idx)
        with open(path, "wb") as f:
            pickle.dump(pickle_obj, f)

    def _load_recipe(self):
        path = self.recipe_file
        if not path.exists():
            raise ValueError("recipe pickle file does not exist.")
        self.print(f'loading recipe from file "{path}"')
        with open(path, "r") as f:
            pickle_obj = pickle.load(f)
        self.recipe, self.num_ops, self.value_array_length, self.root_value_idx = pickle_obj

    def eval(self, x: TYPE_1D_FLOAT, rectify_input: bool = False) -> float:
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

        # use numpy+Numba recipe evaluation as fallback
        # input type and shape should always be validated.
        #   otherwise the numba jit compiled functions may fail with cryptic error messages
        validate_query_point(x, self.dim)

        if self.use_c_eval:
            return self._eval_c(x)

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
            *self.recipe,
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
        return PATH2CACHE / self.get_c_file_name()

    @property
    def c_file_compiled(self):
        return PATH2CACHE / self.get_c_file_name(ending=".so")

    @property
    def recipe_file(self) -> Path:
        return PATH2CACHE / f"numpy_recipe_{hash(self)}.pickle"

    def _write_c_file(self):
        path_out = self.c_file
        if path_out.exists():
            return
        # for compiling the instructions the factorisation must be computed first
        self._compute_factorisation()
        self.print("compiling C instructions ...")
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

        # NOTE: the coefficient array will be used to store intermediary results
        # -> copy to use independent instance of array (NO pointer to external array!)
        func_def = f"{f_type} {func_name}({f_type} x[{nr_dims}], {f_type} {coeff_array}[{nr_coeffs}])"
        # declare function ("header")
        instr += f"{func_def};\n"
        # function definition
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
        self.print(f"writing in file {path_out}")
        with open(path_out, "w") as c_file:
            c_file.write(instr)

    def get_c_instructions(self) -> str:
        path = self.c_file
        self.print(f"reading C instructions from file {path}")
        with open(path) as c_file:
            instr = c_file.read()
        return instr

    def _compile_c_file(self):
        path_out = self.c_file_compiled
        if path_out.exists():
            self.print(f"using existing compiled C file {path_out}")
            return

        compiler = "gcc"
        if shutil.which(compiler) is None:
            compiler = "cc"
            if shutil.which(compiler) is None:
                raise ValueError("compiler (`gcc` or `cc`) not present. install one first.")

        self._write_c_file()
        path_in = self.c_file
        self.print(f"compiling to file {path_out}")
        cmd = [compiler, "-shared", "-o", str(path_out), "-fPIC", str(path_in)]
        subprocess.call(cmd)
        if not path_out.exists():
            raise ValueError(f"expected compiled file missing: {path_out}")
