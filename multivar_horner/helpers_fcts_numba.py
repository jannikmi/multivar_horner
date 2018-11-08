# TODO force all to u4, no i8?!
# TODO
# for Ahead-Of-Time Compilation:
# from numba.pycc import CC
# cc = CC('compiled_helpers', )
# # Uncomment the following line to print out the compilation steps
# cc.verbose = True
# TODO     TypingError: numba doesn't support kwarg for prod
# @jit(f8(f8[:], f8[:], u4[:, :]), nopython=True, cache=True)


# precompiled time critical helper functions
import numpy as np
from numba import b1, f8, i8, jit, u4


def eval_naive(x, coefficients, exponents):
    return np.sum(coefficients.T * np.prod(np.power(x, exponents), axis=1), axis=1)[0]


# @cc.export('eval_compiled', 'f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :])')
@jit(f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :], b1[:]), nopython=True, cache=True)
def eval_recipe(x, value_array, scalar_recipe, monomial_recipe, tree_recipe, tree_ops):
    # print(value_array)

    # print('computing scalar factors: ...')
    # scalar recipe instruction encoding: target, source, exponent
    for i in range(scalar_recipe.shape[0]):
        # print('value[{}] = {} ^ {}'.format(target, x[source1], exponent))
        # value_array[target] = x[source1] ** exponent
        # target, source1, exponent = scalar_recipe[i, 0], scalar_recipe[i, 1], scalar_recipe[i, 2]
        value_array[scalar_recipe[i, 0]] = x[scalar_recipe[i, 1]] ** scalar_recipe[i, 2]

    # print('computing monomial factors: ...')
    # monomial recipe instruction encoding: target, source1, source2
    for i in range(monomial_recipe.shape[0]):
        # print('value[{}] = {} * {} (idx: {}, {})'.format(target, value_array[source1], value_array[source2], source1,
        #                                                 source2))
        # target, source1, source2 = monomial_recipe[i, 0], monomial_recipe[i, 1], monomial_recipe[i, 2]
        # value_array[target] = value_array[source1] * value_array[source2]
        value_array[monomial_recipe[i, 0]] = value_array[monomial_recipe[i, 1]] * value_array[monomial_recipe[i, 2]]

    # print('evaluating factorisation tree: ...')
    # tree recipe instruction encoding: target, source
    # separate operation array: *op_id*
    # value_array[target] = value_array[target] *op* value_array[source]
    for i in range(tree_recipe.shape[0]):
        # target, operation, source1 = tree_recipe[i, 0], tree_recipe[i, 1], tree_recipe[i, 2]
        target = tree_recipe[i, 0]
        if tree_ops[i]:  # ADDITION: 1
            # print('value[{}] = {} + {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] + value_array[source1]
            value_array[target] = value_array[target] + value_array[tree_recipe[i, 1]]
        else:
            # print('value[{}] = {} * {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] * value_array[source1]
            value_array[target] = value_array[target] * value_array[tree_recipe[i, 1]]

    # the value at the first position is the value of the polynomial
    return value_array[0]


@jit(u4[:, :](i8[:], i8[:], u4[:, :]), nopython=True, cache=True)
def factor_out(factor_dimensions, factor_exponents, exponent_matrix):
    for f in range(factor_dimensions.shape[0]):
        dim = factor_dimensions[f]
        exp = factor_exponents[f]
        # reduce all entries in this dimension (=column) by exp
        for r in range(exponent_matrix.shape[0]):
            exponent_matrix[r, dim] -= exp

    return exponent_matrix


@jit(u4[:, :](u4[:, :], u4[:, :]), nopython=True, cache=True)
def compile_usage_statistic(exponents, usage_statistic):
    # count how many times each scalar factor x_dim^exp is being used
    for i in range(exponents.shape[0]):
        for dim in range(exponents.shape[1]):
            exp = exponents[i, dim]
            # do not count entries with an exponent = 0 (not needed for factorising)
            if exp > 0:
                usage_statistic[dim, exp] += 1

    return usage_statistic


@jit(u4[:, :](u4, i8[:], i8[:], u4[:, :], u4[:, :]), nopython=True, cache=True)
def compile_usage_rows(usage_count, scalar_dimensions, scalar_exponents, exponents, usage_rows):
    # find out where (which monomials) the maximally used scalar factors appear
    for i in range(scalar_dimensions.shape[0]):  # for every scalar factor
        found_rows = 0
        dim = scalar_dimensions[i]
        exp = scalar_exponents[i]
        for row in range(exponents.shape[0]):
            if exponents[row, dim] == exp:
                usage_rows[i, found_rows] = row
                found_rows += 1
                if found_rows == usage_count:
                    # early stopping: all usages have been found
                    break
    return usage_rows


@jit(b1[:, :](b1[:, :]), nopython=True, cache=True)
def make_symmetric(square_matrix):
    # mirror the 'upper right' triangle to the lower left
    size = square_matrix.shape[0]
    for r in range(0, size):
        # the diagonal (r=c) does not have to be mirrored
        for c in range(r + 1, size):
            square_matrix[c, r] = square_matrix[r, c]
    return square_matrix


#
# import numpy as np
# A = np.array([[0,1,1],[0,0,1],[0,0,1]])
# print(A)
# print(make_symmetric(A))

@jit(b1[:, :](u4[:, :], b1[:, :]), nopython=True, cache=True)
def build_row_equality_matrix(matrix, equality_matrix):
    # equality matrix is a square bool matrix initialized to all true

    # do not check the last column (noting to compare to)
    nr_rows = matrix.shape[0]
    nr_cols = matrix.shape[1]
    for r1 in range(nr_rows):
        for r2 in range(r1 + 1, nr_rows):
            for c in range(nr_cols):
                if matrix[r1, c] != matrix[r2, c]:
                    # ATTENTION: fill only the 'upper right' triangle of the equality matrix (should be symmetric)!
                    # r2 > r1
                    equality_matrix[r1, r2] = False
                    break

    return make_symmetric(equality_matrix)


@jit(i8(i8, i8[:]), nopython=True, cache=True)
def index_of(value, vector):
    """
    returns the index of the first appearance of value in row vector
    :param value:
    :param vector:
    :return:
    """
    for i in range(vector.shape[0]):
        if vector[i] == value:
            return i
    return -1
