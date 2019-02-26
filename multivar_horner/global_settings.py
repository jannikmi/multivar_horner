from numpy import float64, int64, uint32

ID_MULT = False
ID_ADD = True  # ATTENTION: is being used in helpers_fcts_numba.py/eval_recipe()

# numba is expecting certain data types (static typing):
INT_DTYPE = int64  # i8 =  8byte integer
UINT_DTYPE = uint32  # u4 =  4byte unsigned integer
FLOAT_DTYPE = float64  # f8 =  8byte float

DEFAULT_PICKLE_FILE_NAME = 'multivar_polynomial.pickle'

DEBUG = False
# DEBUG = True
