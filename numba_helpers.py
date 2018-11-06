# from numba import b1, f8, i2, i4, jit, typeof, u2, u8


# # for Ahead-Of-Time Compilation:
# from numba.pycc import CC
# cc = CC('compiled_helpers', )
# # Uncomment the following line to print out the compilation steps
# # cc.verbose = True


# dtype_3floattuple = typeof((1.0, 1.0, 1.0))
# dtype_2floattuple = typeof((1.0, 1.0))

ID_MULT = 0


# @cc.export('inside_polygon', 'b1(i4, i4, i4[:, :])')
# @jit(b1(i4, i4, i4[:, :]), nopython=True, cache=True)
def eval_compiled(x, value_array, scalar_recipe, monomial_recipe, tree_recipe):
    print(value_array)

    print('computing factors: ...')
    # scalar recipe instruction encoding: target, source, exponent
    for target, source1, exponent in scalar_recipe:
        print('value[{}] = {} ^ {}'.format(target, x[source1], exponent))
        value_array[target] = x[source1] ** exponent

    # monomial recipe instruction encoding: target, source1, source2
    for target, source1, source2 in monomial_recipe:
        value_array[target] = value_array[source1] * value_array[source2]

    print('evaluating tree: ...')
    # tree recipe instruction encoding: target, op, source
    for target, operation, source1 in tree_recipe:
        if operation == ID_MULT:
            print('value[{}] = {} * {}'.format(target,value_array[target],value_array[source1]))
            value_array[target] = value_array[target] * value_array[source1]
        else:
            print('value[{}] = {} + {}'.format(target,value_array[target],value_array[source1]))
            value_array[target] = value_array[target] + value_array[source1]

    # the value at the first position is the value of the polynomial
    return value_array[0]
