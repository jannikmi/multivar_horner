# TODO
# for Ahead-Of-Time Compilation:
# from numba.pycc import CC
# cc = CC('compiled_helpers', )
# # Uncomment the following line to print out the compilation steps
# cc.verbose = True


# precompiled time critical helper functions
import numpy as np
from numba import b1, f8, jit, u4


# TODO     TypingError: numba doesn't support kwarg for prod
# @jit(f8(f8[:], f8[:], u4[:, :]), nopython=True, cache=True)
def naive_eval(x, coefficients, exponents):
    return np.sum(coefficients.T * np.prod(np.power(x, exponents), axis=1), axis=1)[0]


# @cc.export('eval_compiled', 'f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :])')
@jit(f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :], u4[:, :], b1[:]), nopython=True, cache=True)
def eval_recipe(x, value_array, copy_recipe, scalar_recipe, monomial_recipe, tree_recipe, tree_ops):
    # IMPORTANT: the order of following the recipes is not arbitrary!
    # print(value_array)

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

    # print('computing monomial factors: ...')
    # monomial recipe instruction encoding: target, source1, source2
    # [target, source1, source2] = monomial_recipe[i, :]
    for i in range(monomial_recipe.shape[0]):
        # print('value[{}] = {} * {} (idx: {}, {})'.format(target, value_array[source1], value_array[source2], source1,
        #                                                 source2))
        # value_array[target] = value_array[source1] * value_array[source2]
        value_array[monomial_recipe[i, 0]] = value_array[monomial_recipe[i, 1]] * value_array[monomial_recipe[i, 2]]

    # print('evaluating factorisation tree: ...')
    # tree recipe instruction encoding: target, source
    # [target, source] = tree_recipe[i, :]
    # separate operation array: *op_id*
    # value_array[target] = value_array[target] *op* value_array[source]
    for i in range(tree_recipe.shape[0]):
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


@jit(u4(u4[:]), nopython=True, cache=True)
def num_ops_1D_horner(unique_exponents):
    """
    :param unique_exponents: np array of unique exponents sorted in increasing order without 0
    :return: the number of operations of the one dimensional horner factorisation
        without counting additions (just MUL & POW) and without considering the coefficients


    NOTE: in 1D the horner factorisation is both unique and optimal (minimal amount of operations)
    """
    nr_unique_exponents = unique_exponents.shape[0]
    # the exponent 0 is not present!
    assert not np.any(unique_exponents == 0)

    if nr_unique_exponents == 0:
        return 0

    # one MUL operation is required !between! all factors in the factorisation chain
    # the amount of factors (= "length of factorisation chain") is equal to the amount of unique existing exponents
    num_ops = nr_unique_exponents - 1

    # start with exponent 0 (not in unique exponents)
    # the difference between one and the next exponent determines if a POW operation is needed to evaluate a factor
    # unique exponents are sorted: prev_exp < exp
    prev_exp = 0
    for i in range(nr_unique_exponents):
        exp = unique_exponents[i]
        if exp - prev_exp >= 2:
            num_ops += 1  # 1 POW operation
        prev_exp = exp

    return num_ops


@jit(u4(u4[:, :]), nopython=True, cache=True)
def true_num_ops(exponent_matrix):
    """
    without counting additions (just MUL & POW) and but WITH considering the coefficients (1 MUL per monomial)
    """
    num_ops = 0
    for monomial_nr in range(exponent_matrix.shape[0]):
        for dim in range(exponent_matrix.shape[1]):
            exp = exponent_matrix[monomial_nr, dim]
            if exp > 0:
                # per scalar factor 1 MUL operation is required
                num_ops += 1
                if exp >= 2:
                    # for scalar factors with exponent >= 2 additionally 1 POW operation is required
                    num_ops += 1

    return num_ops


@jit(u4[:](u4, u4[:], u4[:], u4[:, :]), nopython=True, cache=True)
def compile_usage(dim, usage_vector, unique_exponents, exponent_matrix):
    """
    :return: a vector with the usage count of every unique exponent
    """

    for i in range(exponent_matrix.shape[0]):
        exp = exponent_matrix[i, dim]
        for j in range(len(unique_exponents)):
            if exp < unique_exponents[j]:
                break
            usage_vector[j] += 1

    return usage_vector


@jit(b1[:](u4, b1[:], u4[:], u4[:], u4[:, :]), nopython=True, cache=True)
def compile_valid_options(dim, valid_option_vector, usage_vector, unique_exponents, exponent_matrix):
    if len(valid_option_vector) == 0:
        # there are no unique exponents
        return valid_option_vector

    usage_vector = compile_usage(dim, usage_vector, unique_exponents, exponent_matrix)

    for exp_idx in range(usage_vector.size):
        # stop at the highest exponent having a usage >=2
        if usage_vector[exp_idx] >= 2:
            valid_option_vector[exp_idx] = True
        else:
            # all higher exponents have a lower usage!
            break

    return valid_option_vector


@jit(u4(u4, u4, u4[:, :]), nopython=True, cache=True)
def count_usage(dim, exp, exponent_matrix):
    """
    :return: the amount of times a scalar factor appears in the monomials
    """

    usage_cnt = 0
    for i in range(exponent_matrix.shape[0]):
        if exponent_matrix[i, dim] >= exp:
            usage_cnt += 1

    return usage_cnt


@jit(u4(u4, u4), nopython=True, cache=True)
def factor_num_ops(dim, exp):
    """
    :param factor: a tuple (dim, exp) representing the scalar factor: x_dim^exp
    :return: the amount of operations required to evaluate the given scalar factor
    """
    if exp >= 2:
        # 1 MUL + 1 POW
        return 2
    else:
        # 1 MUL
        return 1
