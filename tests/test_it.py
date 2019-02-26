import random
import timeit
import unittest
from math import log10

import numpy as np
import pytest

from multivar_horner.global_settings import UINT_DTYPE
from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial

MAX_DIM = 2
MAX_DEGREE = 2
NR_SAMPLES = 200

poly_settings_list = []
input_list = []
poly_class_instances = []


# TODO automatically also test only_scalar_factor=True
# TODO compare difference in computed values (error)
# TODO test consecutive evaluations of the same polynomial!

def proto_test_case(data, fct):
    all_good = True
    for input, expected_output in data:
        # print(input, expected_output, fct(input))
        actual_output = fct(input)
        if actual_output != expected_output:
            print('input: {} expected: {} got: {}'.format(input, expected_output, actual_output))
            all_good = False

    assert all_good


#
# def id2exponent_vect(prime_list, monomial_id):
#     # find the exponent vector corresponding to a monomial id
#     # = prime decomposition
#     exponent_vect = np.zeros(prime_list.shape, dtype=UINT_DTYPE)
#     current_id = monomial_id
#     for i, prime in enumerate(prime_list):
#         while 1:
#             quotient, remainder = divmod(current_id, prime)
#             if remainder == 0:
#                 exponent_vect[i] += 1
#                 current_id = quotient
#             else:
#                 break
#
#         if current_id == 0:
#             break
#
#     if current_id != 0:
#         raise ValueError('no factorisation found')
#
#     return exponent_vect
#
#
# def _sparse_range_generator(max_value, density):
#     for i in range(max_value):
#         if random.random() < density:
#             yield i


def random_polynomial_settings(all_exponents, max_abs_coeff=1.0):
    # every exponent can take the values in the range [0; max_degree]
    max_nr_exponent_vects = all_exponents.shape[0]

    # decide how many entries the polynomial should have
    # desired for meaningful speed test results:
    # every possible polynomial should appear with equal probability
    # there must be at least 1 entry
    nr_exponent_vects = random.randint(1, max_nr_exponent_vects)

    row_idxs = list(range(max_nr_exponent_vects))
    assert max_nr_exponent_vects >= nr_exponent_vects
    for length in range(max_nr_exponent_vects, nr_exponent_vects, -1):
        # delete random entry from list
        row_idxs.pop(random.randint(0, length - 1))

    assert len(row_idxs) == nr_exponent_vects

    exponents = all_exponents[row_idxs, :]
    coefficients = (np.random.rand(nr_exponent_vects, 1) - 0.5) * (2 * max_abs_coeff)
    return coefficients, exponents


def all_possible_exponents(dim, max_degree):
    def cntr2exp_vect(cntr):
        exp_vect = np.empty((dim), dtype=UINT_DTYPE)
        for d in range(dim - 1, -1, -1):
            divisor = (max_degree + 1) ** d
            # cntr = quotient*divisor + remainder
            quotient, remainder = divmod(cntr, divisor)
            exp_vect[d] = quotient
            cntr = remainder
        return exp_vect

    max_nr_exponent_vects = (max_degree + 1) ** dim
    all_possible = np.empty((max_nr_exponent_vects, dim), dtype=UINT_DTYPE)
    for i in range(max_nr_exponent_vects):
        # print(i, cntr2exp_vect(i))
        all_possible[i] = cntr2exp_vect(i)

    return all_possible


def rnd_settings_list(length, dim, max_degree):
    all_exponent_vect = all_possible_exponents(dim, max_degree)
    settings_list = [random_polynomial_settings(all_exponent_vect) for i in range(length)]

    # # random settings should have approx. half the amount of maximal entries on average
    # num_monomial_entries = 0
    # for settings in settings_list:
    #     num_monomial_entries += settings[0].shape[0]
    #
    # avg_monomial_entries = num_monomial_entries / length
    # max_monomial_entries = int((max_degree + 1) ** dim)
    # print(avg_monomial_entries, max_monomial_entries)
    return settings_list


def rnd_input_list(length, dim, max_abs_val):
    return [(np.random.rand(dim) - 0.5) * (2 * max_abs_val) for i in range(length)]


def setup_time_fct(poly_class):
    # store instances globally to directly use them for eval time test
    global poly_settings_list, poly_class_instances

    poly_class_instances = []
    for settings in poly_settings_list:
        poly_class_instances.append(poly_class(*settings))


def eval_time_fct():
    global poly_class_instances, input_list

    for instance, input in zip(poly_class_instances, input_list):
        instance.eval(input)


def get_avg_num_ops():
    global poly_class_instances

    num_ops = 0
    for instance in poly_class_instances:
        num_ops += instance.get_num_ops()

    avg_num_ops = round(num_ops / len(poly_class_instances))
    return avg_num_ops


def time_preprocess(time):
    valid_digits = 4
    zero_digits = abs(min(0, int(log10(time))))
    digits_to_print = zero_digits + valid_digits
    return str(round(time, digits_to_print))


def difference(quantity1, quantity):
    speedup = round((quantity / quantity1 - 1), 1)
    if speedup < 0:
        speedup = round((quantity1 / quantity - 1), 1)
        if speedup > 10.0:
            speedup = round(speedup)

        return str(speedup) + ' x less'
    else:
        if speedup > 10.0:
            speedup = round(speedup)
        return str(speedup) + ' x more'


def compute_lucrativity(setup_horner, setup_naive, eval_horner, eval_naive):
    benefit_eval = eval_naive - eval_horner
    loss_setup = setup_horner - setup_naive
    # x * benefit = loss
    return round(loss_setup / benefit_eval)


def speed_test_run(dim, max_degree, nr_samples, template):
    global poly_settings_list, input_list, poly_class_instances

    poly_settings_list = rnd_settings_list(nr_samples, dim, max_degree)
    input_list = rnd_input_list(nr_samples, dim, max_abs_val=1.0)

    setup_time_naive = timeit.timeit("setup_time_fct(MultivarPolynomial)", globals=globals(), number=1)
    setup_time_naive = setup_time_naive / NR_SAMPLES  # = avg. per sample

    # poly_class_instances is not populated with the naive polynomial class instances
    # print(poly_class_instances[0])
    num_ops_naive = get_avg_num_ops()

    eval_time_naive = timeit.timeit("eval_time_fct()", globals=globals(), number=1)
    eval_time_naive = eval_time_naive / NR_SAMPLES  # = avg. per sample

    setup_time_horner = timeit.timeit("setup_time_fct(HornerMultivarPolynomial)", globals=globals(), number=1)
    setup_time_horner = setup_time_horner / NR_SAMPLES  # = avg. per sample

    # poly_class_instances is not populated with the horner polynomial class instances
    # print(poly_class_instances[0])
    num_ops_horner = get_avg_num_ops()

    eval_time_horner = timeit.timeit("eval_time_fct()", globals=globals(), number=1)
    eval_time_horner = eval_time_horner / NR_SAMPLES  # = avg. per sample

    setup_delta = difference(setup_time_naive, setup_time_horner)
    eval_delta = difference(eval_time_naive, eval_time_horner)
    ops_delta = difference(num_ops_naive, num_ops_horner)
    lucrative_after = compute_lucrativity(setup_time_horner, setup_time_naive, eval_time_horner, eval_time_naive)

    print(template.format(str(dim), str(max_degree), time_preprocess(setup_time_naive),
                          time_preprocess(setup_time_horner), str(setup_delta),
                          time_preprocess(eval_time_naive), time_preprocess(eval_time_horner),
                          str(eval_delta), str(num_ops_naive), str(num_ops_horner), ops_delta, str(lucrative_after)))


class MainTest(unittest.TestCase):

    def test_eval(self):
        def cmp_value_fct(inp):
            print()
            coeff, exp, x = inp
            x = np.array(x).T
            poly = MultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True)
            res1 = poly.eval(x, validate_input=True)
            print(str(poly))

            horner_poly = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                   compute_representation=True, find_optimal=False)
            res2 = horner_poly.eval(x, validate_input=True)
            print(str(horner_poly))
            # print('x=',x.tolist())

            horner_poly_opt = HornerMultivarPolynomial(coeff, exp, rectify_input=True, validate_input=True,
                                                       compute_representation=True, find_optimal=True)
            res3 = horner_poly_opt.eval(x, validate_input=True)
            print(str(horner_poly_opt))
            # print('x=',x.tolist())

            if res1 != res2 or res2 != res3:
                print('results differ:', res1, res2, res3)
                assert False

            assert horner_poly.num_ops >= horner_poly_opt.num_ops

            return poly.eval(x, validate_input=True)

        invalid_test_data = [
            # calling with x of another dimension
            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            # negative exponents are not allowed
            (([1.0, 2.0, 3.0],
              [[3, -1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            # duplicate exponent entries are not allowed
            # negative exponents are not allowed
            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0, 4.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

        ]

        for inp, expected_output in invalid_test_data:
            with pytest.raises(AssertionError):
                cmp_value_fct(inp)

        invalid_test_data = [
            #
            # p(x) =  5.0
            (([5.0],  # coefficients
              [0],  # exponents
              [0.0]),  # x
             5.0),  # p(x)

            # p(1.0) = 1.0
            (([5.0],
              [0],
              [1.0]),
             5.0),

            # p(-1.0) = -1.0
            (([5.0],
              [0],
              [-1.0]),
             5.0),

            # p(33.5) =33.5
            (([5.0],
              [0],
              [33.5]),
             5.0),

            # p(x) =  1.0* x_1^1
            # p(0.0) = 0.0
            (([1.0],  # coefficients
              [1],  # exponents
              [0.0]),  # x
             0.0),  # p(x)

            # p(1.0) = 1.0
            (([1.0],
              [1],
              [1.0]),
             1.0),

            # p(-1.0) = -1.0
            (([1.0],
              [1],
              [-1.0]),
             -1.0),

            # p(33.5) =33.5
            (([1.0],
              [1],
              [33.5]),
             33.5),

            # p(x) =  1.0* x_1^1 + 0.0* * x_2^1
            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [0.0, 0.0]),
             0.0),

            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [1.0, 0.0]),
             1.0),

            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [-1.0, 0.0]),
             -1.0),

            (([1.0, 0.0],
              [[1, 0], [0, 1]],
              [33.5, 0.0]),
             33.5),

            # p(x) =  5.0 +  1.0* x_1^1
            (([5.0, 1.0],
              [[0, 0], [1, 0]],
              [0.0, 0.0]),
             5.0),

            (([5.0, 1.0],
              [[0, 0], [1, 0]],
              [1.0, 0.0]),
             6.0),

            (([5.0, 1.0],
              [[0, 0], [1, 0]],
              [-1.0, 0.0]),
             4.0),

            (([5.0, 1.0],
              [[0, 0], [1, 0]],
              [33.5, 0.0]),
             38.5),

            # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2
            (([5.0, 2.0, 1.0],
              [[0, 0], [1, 0], [2, 0]],
              [0.0, 0.0]),
             5.0),

            (([5.0, 2.0, 1.0],
              [[0, 0], [1, 0], [2, 0]],
              [1.0, 0.0]),
             8.0),  # p(x) =  5.0 + 2.0 + 1.0

            (([5.0, 2.0, 1.0],
              [[0, 0], [1, 0], [2, 0]],
              [-1.0, 0.0]),
             4.0),  # p(x) =  5.0 - 2.0 + 1.0

            (([5.0, 2.0, 1.0],
              [[0, 0], [1, 0], [2, 0]],
              [2.0, 0.0]),
             13.0),  # p(x) =  5.0 + 2.0* 2.0^1 + 1.0* 2.0^2

            # p(x) =  5.0 + 2.0* x_1^1 + 1.0* x_1^2 + 2.0* x_1^2 *x_2^1
            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [0.0, 0.0]),
             5.0),

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [1.0, 0.0]),
             8.0),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *0^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [1.0, 1.0]),
             10.0),  # p(x) =  5.0 + 2.0* 1^1 + 1.0* 1^2 + 2.0* 1^2 *1^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [-1.0, 0.0]),
             4.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *0^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [-1.0, 1.0]),
             6.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *1^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [-1.0, 2.0]),
             8.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *2^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [-1.0, 3.0]),
             10.0),  # p(x) =  5.0 + 2.0* (-1)^1 + 1.0* (-1)^2 + 2.0* (-1)^2 *3^1

            (([5.0, 2.0, 1.0, 2.0],
              [[0, 0], [1, 0], [2, 0], [2, 1]],
              [-2.0, 3.0]),
             # p(x) = 5.0 + 2.0* (-2)^1 + 1.0* (-2)^2 + 2.0* (-2)^2 *3^1 = 5.0 + 2.0* (-2) + 1.0* 4 + 2.0* 4 *3
             29.0),

            # [20] p(x) = 1.0 x_1^3 x_2^1 + 2.0 x_1^2 x_3^1 + 3.0 x_1^1 x_2^1 x_3^1
            # [17] p(x) = x_2^1 [ x_1^3 [ 1.0 ] + x_1^1 x_3^1 [ 3.0 ] ] + x_1^2 x_3^1 [ 2.0 ]
            (([1.0, 2.0, 3.0],
              [[3, 1, 0], [2, 0, 1], [1, 1, 1]],
              [-2.0, 3.0, 1.0]),
             -34.0),

            # [27] p(x) = 1.0 x_3^1 + 2.0 x_1^3 x_2^3 + 3.0 x_1^2 x_2^3 x_3^1 + 4.0 x_1^1 x_2^5 x_3^1
            (([1.0, 2.0, 3.0, 4.0],
              [[0, 0, 1], [3, 3, 0], [2, 3, 1], [1, 5, 1]],
              [-2.0, 3.0, 1.0]),
             -2051.0),
        ]

        proto_test_case(invalid_test_data, cmp_value_fct)

    def test_speed(self):
        # this test also fulfills the purpose of testing the robustness of the code
        # (many random polygons are being created and evaluated)

        # todo write results to file
        # TODO test for larger dimensions
        # TODO plot speed
        # TODO compare & plot the performance wrt. the "density" of the polynomials
        # naive should stay constant, horner should get slower

        print('\nSpeed test:')
        print('testing {} evenly distributed random polynomials'.format(NR_SAMPLES))
        print('average timings per polynomial:\n')

        print(' {0:11s}  |  {1:38s} |  {2:35s} |  {3:35s} | {4:20s}'.format('parameters', 'setup time (/s)',
                                                                            'eval time (/s)',
                                                                            '# operations', 'lucrative after '))
        template = '{0:3s} | {1:7s} | {2:10s} | {3:10s} | {4:13s} | {5:10s} | {6:10s} | {7:10s} | {8:10s} | ' \
                   '{9:10s} | {10:10s} | {11:10s}'

        print(template.format('dim', 'max_deg', 'naive', 'horner', 'delta', 'naive',
                              'horner', 'delta', 'naive', 'horner', 'delta', '   # evals'))
        print('=' * 160)

        for dim in range(1, MAX_DIM + 1):
            for max_degree in range(1, MAX_DEGREE + 1):
                # TODO make num samples dependent on parameters, print
                speed_test_run(dim, max_degree, NR_SAMPLES, template)
            print()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MainTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
