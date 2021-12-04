from multivar_horner.classes.horner_poly import AbstractPolynomial
from multivar_horner.global_settings import TYPE_1D_FLOAT, TYPE_2D_INT
from multivar_horner.helper_fcts import rectify_query_point, validate_query_point
from multivar_horner.helpers_fcts_numba import count_num_ops_naive, naive_eval


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
        verbose: bool, default=False
            whether to print status statements

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
        verbose: bool = False,
        *args,
        **kwargs,
    ):

        super(MultivarPolynomial, self).__init__(
            coefficients, exponents, rectify_input, compute_representation, verbose
        )

        # NOTE: count the number of multiplications of the representation
        # not the actual amount of operations required by the naive evaluation with numpy arrays
        self.num_ops = count_num_ops_naive(self.exponents)
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
