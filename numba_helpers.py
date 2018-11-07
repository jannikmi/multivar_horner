from numba import b1, f8, i2, i4, jit, typeof, u2, u8, u4

# TODO
# for Ahead-Of-Time Compilation:
from numba.pycc import CC

# cc = CC('compiled_helpers', )
# # Uncomment the following line to print out the compilation steps
# cc.verbose = True

ID_MULT = 0


@cc.export('eval_compiled', 'f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :])')
@jit(f8(f8[:], f8[:], u4[:, :], u4[:, :], u4[:, :]), nopython=True, cache=True)
def eval_recipe(x, value_array, scalar_recipe, monomial_recipe, tree_recipe):
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
    # tree recipe instruction encoding: target, op,
    for i in range(tree_recipe.shape[0]):
        # target, operation, source1 = tree_recipe[i, 0], tree_recipe[i, 1], tree_recipe[i, 2]
        target = tree_recipe[i, 0]
        if tree_recipe[i, 1] == ID_MULT:
            # print('value[{}] = {} * {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] * value_array[source1]
            value_array[target] = value_array[target] * value_array[tree_recipe[i, 2]]
        else:
            # print('value[{}] = {} + {}'.format(target, value_array[target], value_array[source1]))
            # value_array[target] = value_array[target] + value_array[source1]
            value_array[target] = value_array[target] + value_array[tree_recipe[i, 2]]

    # the value at the first position is the value of the polynomial
    return value_array[0]
