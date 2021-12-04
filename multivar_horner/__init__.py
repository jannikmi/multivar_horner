# -*- coding:utf-8 -*-
from multivar_horner.classes.horner_poly import HornerMultivarPolynomial
from multivar_horner.classes.regular_poly import MultivarPolynomial

# https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
# determines which objects will be imported with "import *"
__all__ = ["HornerMultivarPolynomial", "MultivarPolynomial"]
