# -*- coding:utf-8 -*-

import abc
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import numpy as np

from multivar_horner.global_settings import BOOL_DTYPE, DEBUG, DEFAULT_PICKLE_FILE_NAME, TYPE_1D_FLOAT, TYPE_2D_INT
from multivar_horner.helper_fcts import (
    rectify_coefficients,
    rectify_construction_parameters,
    validate_coefficients,
    validate_construction_parameters,
)


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
        "verbose",
        "_hash_val",
    ]

    def __init__(
        self,
        coefficients: TYPE_1D_FLOAT,
        exponents: TYPE_2D_INT,
        rectify_input: bool = False,
        compute_representation: bool = False,
        verbose: bool = False,
    ):
        self._hash_val: int
        self.verbose: bool = verbose
        self.compute_representation: bool = compute_representation

        if rectify_input:
            coefficients, exponents = rectify_construction_parameters(coefficients, exponents)
        validate_construction_parameters(coefficients, exponents)
        self.coefficients: np.ndarray = coefficients
        self.exponents: np.ndarray = exponents

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
            # lazy initialisation: compute the expensive hash value only once on demand
            props = (self.dim, self.num_monomials, *self.exponents.flatten())
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

    def print(self, *args):
        if self.verbose:
            print(*args)

    @abc.abstractmethod
    def compute_string_representation(self, *args, **kwargs) -> str:
        """computes a string representation of the polynomial and sets self.representation

        Returns:
            a string representing this polynomial instance
        """
        ...

    def export_pickle(self, path: str = DEFAULT_PICKLE_FILE_NAME):
        self.print(f'storing polynomial in file "{path}"')
        with open(path, "wb") as f:
            pickle.dump(self, f)

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


def load_pickle(path: str = DEFAULT_PICKLE_FILE_NAME) -> AbstractPolynomial:
    print('importing polynomial from file "{}"'.format(path))
    with open(path, "rb") as f:
        return pickle.load(f)
