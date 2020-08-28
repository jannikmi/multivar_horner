from typing import List, Union

import numpy

ID_MULT = False
ID_ADD = True  # ATTENTION: is being used in helpers_fcts_numba.py/eval_recipe()

# numba is expecting certain data types (static typing):
# INT_DTYPE = numpy.int64  # i8 =  8byte integer
UINT_DTYPE = numpy.uint32  # u4 =  4byte unsigned integer
FLOAT_DTYPE = numpy.float64  # f8 =  8byte float, default
BOOL_DTYPE = numpy.bool

# python typing
TYPE_1D_FLOAT = Union[numpy.ndarray, List[float]]
TYPE_2D_INT = Union[numpy.ndarray, List[List[int]]]

DEFAULT_PICKLE_FILE_NAME = 'multivar_polynomial.pickle'

DEBUG = False
# DEBUG = True
