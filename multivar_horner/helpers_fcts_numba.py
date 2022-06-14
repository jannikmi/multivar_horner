# TODO
# for Ahead-Of-Time Compilation:
# from numba.pycc import CC
# cc = CC('compiled_helpers', )
# # Uncomment the following line to print out the compilation steps
# cc.verbose = True


import numpy as np

using_numba = True
try:
    from numba import b1, f8, i8, njit, u4, void
except ImportError:
    using_numba = False
    # replace numba functionality with "transparent" implementations
    from multivar_horner._numba_replacements import b1, f8, i8, njit, u4, void

# DTYPES:
F = f8
F_1D = F[:]
UINT = u4
UINT_1D = UINT[:]
UINT_2D = UINT[:, :]
INT = i8
BOOL_1D = b1[:]


# time critical helper functions. just in time compiled
# ATTENTION: due to `chace=True`


@njit(F(F_1D, F_1D, UINT_2D), cache=True)
def naive_eval(x, coefficients, exponents):
    nr_coeffs = len(coefficients)
    # nr_monomials,nr_dims = exponents.shape
    # assert nr_monomials == nr_coeffs
    # assert len(x) == nr_dims
    acc = 0.0
    for i in range(nr_coeffs):
        acc = acc + coefficients[i] * np.prod(np.power(x, exponents[i]))
    return acc

    # equivalent one liner:
    # TypingError: numba doesn't support kwarg for prod
    # return np.sum(coefficients.T * np.prod(np.power(x, exponents), axis=1), axis=1)[0]


# @cc.export('eval_compiled', 'f8(f8[:], f8[:], UINT_2, UINT_2, UINT_2, UINT_2, b1[:], u4)')
@njit(
    F(F_1D, F_1D, UINT_2D, UINT_2D, UINT_2D, UINT_2D, BOOL_1D, UINT),
    cache=True,
    debug=True,
)
def eval_recipe(
    x,
    value_array,
    copy_recipe,
    scalar_recipe,
    monomial_recipe,
    tree_recipe,
    tree_ops,
    root_value_address,
):
    # IMPORTANT: the order of following the recipes is not arbitrary!
    #   scalar factors need to be evaluated before monomial factors depending on them...

    # in order to evaluate scalar factors with exponent 1, no exponentiation operation is required
    # simply copy the values of x to the value array
    # copy recipe instruction encoding: target, source
    # [target, source] = copy_recipe[i, :]
    for i in range(copy_recipe.shape[0]):
        # value_array[target] = x[source1]
        value_array[copy_recipe[i, 0]] = x[copy_recipe[i, 1]]

    # print('computing scalar factors: ...')
    # scalar recipe instruction encoding: target, source, exponent
    # [target, source1, exponent] = scalar_recipe[i, :]
    for i in range(scalar_recipe.shape[0]):
        # print('value[{}] = {} ^ {}'.format(target, x[source1], exponent))
        # value_array[target] = x[source1] ** exponent
        value_array[scalar_recipe[i, 0]] = x[scalar_recipe[i, 1]] ** scalar_recipe[i, 2]

    # # DEBUG:
    # accessed_idxs = set()

    # print('computing monomial factors: ...')
    # monomial recipe instruction encoding: target, source1, source2
    # [target, source1, source2] = monomial_recipe[i, :]
    for i in range(monomial_recipe.shape[0]):
        # print('value[{}] = {} * {} (idx: {}, {})'.format(target, value_array[source1], value_array[source2], source1,
        #                                                 source2))
        # value_array[target] = value_array[source1] * value_array[source2]
        value_array[monomial_recipe[i, 0]] = value_array[monomial_recipe[i, 1]] * value_array[monomial_recipe[i, 2]]

        # # DEBUG:
        # accessed_idxs.add(monomial_recipe[i, 1])
        # accessed_idxs.add(monomial_recipe[i, 2])

    # print('evaluating factorisation tree: ...')
    # tree recipe instruction encoding: target, source
    # [target, source] = tree_recipe[i, :]
    # separate operation array: *op_id*
    # value_array[target] = value_array[target] *op* value_array[source]
    for i in range(tree_recipe.shape[0]):
        target = tree_recipe[i, 0]
        source = tree_recipe[i, 1]
        if tree_ops[i]:  # ADDITION: 1
            # print('value[{}] = {} + {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] + value_array[source1]
            value_array[target] += value_array[source]
        else:  # MULTIPLICATION: 0
            # print('value[{}] = {} * {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] * value_array[source1]
            value_array[target] *= value_array[source]

        # # DEBUG:
        # accessed_idxs.add(target)
        # accessed_idxs.add(source)

    # # DEBUG:
    # all_idxs = {x for x in range(len(value_array))}
    # non_accessed_idxs = all_idxs - accessed_idxs
    # # coefficients are stored in the indices until one before the root address
    # coefficient_idxs = {x for x in range(root_value_address)}
    # non_accessed_coefficients = non_accessed_idxs & coefficient_idxs
    # if len(non_accessed_coefficients) > 0:
    #     raise ValueError(f'BUG: these coefficients have been accessed: {non_accessed_coefficients}')
    #
    # # NOTE: no indices must be accessed when the polynomial is a constant
    # if len(non_accessed_idxs) > 0 and len(value_array) > 1:
    #     raise ValueError(f'BUG: these idxs have been accessed: {non_accessed_idxs}')

    return value_array[root_value_address]  # return value of the root node


@njit(UINT(UINT_1D), cache=True)
def num_ops_1D_horner(unique_exponents):
    """
    :param unique_exponents: np array of unique exponents sorted in increasing order without 0
    :return: the number of operations of the one dimensional Horner factorisation
        without counting additions (just MUL & POW) and without considering the coefficients
    do not count additions and exponentiations. just multiplications
    do not consider the multiplications with the coefficients (amount stays constant)


    NOTE: in 1D the Horner factorisation is both unique and optimal (minimal amount of operations)
    -> gives a lower bound for the amount of required operations
    (=required property for usage as cost estimation heuristic)
    """
    nr_unique_exponents = unique_exponents.shape[0]
    # assert not np.any(unique_exponents == 0), "the exponent 0 must not be present"
    if nr_unique_exponents == 0:
        return 0

    # one MUL operation is required !between! all factors in the factorisation chain
    # the amount of factors (= "length of factorisation chain") is equal to the amount of unique existing exponents
    num_ops = -1

    # start with exponent 0 (not in unique exponents)
    # the difference between one and the next exponent determines if a POW operation is needed to evaluate a factor
    # unique exponents are sorted: prev_exp < exp
    prev_exp = 0
    for i in range(nr_unique_exponents):
        exp = unique_exponents[i]
        # the exponents MUST increase -> count at least one operation (multiplication)
        # for every exponent difference >1
        # count exponent-1 additional multiplications for computing the exponentiations
        exp_diff = exp - prev_exp
        num_ops += exp_diff
        prev_exp = exp

    return num_ops


@njit(UINT(UINT_2D), cache=True)
def count_num_ops_naive(exponent_matrix):
    """counts the amount of multiplications required during evaluation

    under the assumption: this polynomial representation does not get factorised any further
    do not count additions and exponentiations

    1 multiplication required for every coefficient when there is a monomial (non zero exponent) present
    one multiplication for multiplying the scalar factors with each other
    (amount of non zero exponents in each monomial -1)
    exponent - 1 multiplications for every evaluation of an exponentiation of every scalar factor
    = the total sum of exponents
    """
    return np.sum(exponent_matrix)


@njit(UINT(UINT, UINT), cache=True)
def factor_num_ops(dim, exp):
    """
    NOTE: all factors are scalars: x^i
    count every exponentiation as exponent-1 multiplications

    :return: the amount of operations required to evaluate the scalar factor
    """
    return exp - 1


@njit(void(INT, UINT_1D, UINT_1D, UINT_2D), cache=True)
def compile_usage(dim, usage_vector, unique_exponents, exponent_matrix):
    """
    computes the vector with the usage count of every unique exponent
    """

    for i in range(exponent_matrix.shape[0]):
        exp = exponent_matrix[i, dim]
        for j in range(len(unique_exponents)):
            if exp < unique_exponents[j]:
                break
            usage_vector[j] += 1


@njit(void(INT, BOOL_1D, UINT_1D, UINT_1D, UINT_2D), cache=True)
def compile_valid_options(dim, valid_option_vector, usage_vector, unique_exponents, exponent_matrix):
    """compute the vector of valid options

    :param dim:
    :param valid_option_vector:
    :param usage_vector:
    :param unique_exponents:
    :param exponent_matrix:
    """
    if len(valid_option_vector) == 0:
        # there are no unique exponents
        return

    compile_usage(dim, usage_vector, unique_exponents, exponent_matrix)

    for exp_idx in range(usage_vector.size):
        # stop at the highest exponent having a usage >=2
        if usage_vector[exp_idx] >= 2:
            valid_option_vector[exp_idx] = True
        else:
            # all higher exponents have a lower usage!
            break


@njit(UINT(UINT, UINT, UINT_2D), cache=True)
def count_usage(dim, exp, exponent_matrix):
    """
    :return: the amount of times a scalar factor appears in the monomials
    """

    usage_cnt = 0
    for i in range(exponent_matrix.shape[0]):
        if exponent_matrix[i, dim] >= exp:
            usage_cnt += 1

    return usage_cnt
