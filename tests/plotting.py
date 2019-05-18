import pickle
import random
import time
import timeit
from math import log10
from os.path import abspath, join, pardir

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from multivar_horner.global_settings import UINT_DTYPE
from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial

# SPEED TESTS
# TODO allow other setting ranges
MAX_DIM = 4
MAX_DEGREE = 10
DIM_RANGE = list(range(1, MAX_DIM + 1))
DEGREE_RANGE = list(range(1, MAX_DEGREE + 1))

NR_SAMPLES = 100

EXPORT_RESOLUTION = 300  # dpi
EXPORT_SIZE_X = 19.0  # inch
EXPORT_SIZE_Y = 11.0  # inch
SHOW_PLOTS = False
PLOTTING_DIR = abspath(join(pardir, 'plots'))
plt.rcParams.update({'font.size': 30})
PATH2DATA = 'results.pickle'


def get_plot_name(file_name='plot'):
    return abspath(join(PLOTTING_DIR, file_name + '_' + str(time.time())[:-7] + '.png'))


def export_plot(fig, file_name):
    fig.set_size_inches(EXPORT_SIZE_X, EXPORT_SIZE_Y, forward=True)
    plt.savefig(get_plot_name(file_name), dpi=EXPORT_RESOLUTION)


poly_settings_list = []
input_list = []
poly_class_instances = []


def random_polynomial_settings(degree, all_exponents, max_abs_coeff=1.0):
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

    # ensure that the polynomial also has exactly the requested degree
    if np.max(np.sum(exponents, axis=1)) < degree:
        # add one monomials with that degree
        max_degr_mon_idxs = np.sum(all_exponents, axis=1) == degree
        length = max_degr_mon_idxs.shape[0]
        row_idxs.append(random.randint(0, length - 1))
        exponents = all_exponents[row_idxs, :]
        nr_exponent_vects += 1

    # the magnitude of the coefficients is arbitrary
    coefficients = (np.random.rand(nr_exponent_vects, 1) - 0.5) * (2 * max_abs_coeff)
    assert (coefficients.shape[0] == exponents.shape[0])
    return coefficients, exponents


def all_possible_exponents(dim, deg):
    def cntr2exp_vect(cntr):
        exp_vect = np.empty((dim), dtype=UINT_DTYPE)
        for d in range(dim - 1, -1, -1):
            divisor = (deg + 1) ** d
            # cntr = quotient*divisor + remainder
            quotient, remainder = divmod(cntr, divisor)
            exp_vect[d] = quotient
            cntr = remainder
        return exp_vect

    max_nr_exponent_vects = (deg + 1) ** dim
    all_possible = np.empty((max_nr_exponent_vects, dim), dtype=UINT_DTYPE)
    for i in range(max_nr_exponent_vects):
        # print(i, cntr2exp_vect(i))
        all_possible[i] = cntr2exp_vect(i)

    return all_possible


def rnd_settings_list(length, dim, degree):
    all_exponent_vect = all_possible_exponents(dim, degree)
    settings_list = [random_polynomial_settings(degree, all_exponent_vect) for i in range(length)]

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


def test_setup_time_naive():
    setup_time_fct(MultivarPolynomial)


def test_setup_time_horner():
    setup_time_fct(HornerMultivarPolynomial)


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


def speed_test_run(dim, degree, nr_samples, template):
    global poly_settings_list, input_list, poly_class_instances

    poly_settings_list = rnd_settings_list(nr_samples, dim, degree)
    input_list = rnd_input_list(nr_samples, dim, max_abs_val=1.0)

    setup_time_naive = timeit.timeit("test_setup_time_naive()", globals=globals(), number=1)
    setup_time_naive = setup_time_naive / NR_SAMPLES  # = avg. per sample

    # poly_class_instances is not populated with the naive polynomial class instances
    # print(poly_class_instances[0])
    num_ops_naive = get_avg_num_ops()

    eval_time_naive = timeit.timeit("eval_time_fct()", globals=globals(), number=1)
    eval_time_naive = eval_time_naive / NR_SAMPLES  # = avg. per sample

    setup_time_horner = timeit.timeit("test_setup_time_horner()", globals=globals(), number=1)
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

    print(template.format(str(dim), str(degree), time_preprocess(setup_time_naive),
                          time_preprocess(setup_time_horner), str(setup_delta),
                          time_preprocess(eval_time_naive), time_preprocess(eval_time_horner),
                          str(eval_delta), str(num_ops_naive), str(num_ops_horner), ops_delta, str(lucrative_after)))

    return (setup_time_naive, eval_time_naive, num_ops_naive), (setup_time_horner, eval_time_horner, num_ops_horner)


def test_speed():
    # this also fulfills the purpose of testing the robustness of the code
    # (many random polygons are being created and evaluated)

    # TODO test for larger dimensions
    # TODO compare & plot the performance wrt. the "density" of the polynomials. sparse <-> fully occupied
    # naive should stay constant, horner should get slower

    # TODO make num tested polynomials dependent on parameters (size, time)

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

    all_results = []

    for dim in DIM_RANGE:
        dim_run_results = []
        for max_degree in DEGREE_RANGE:
            degree_run_results = speed_test_run(dim, max_degree, NR_SAMPLES, template)
            dim_run_results.append(degree_run_results)

        print()  # empty line
        # dim_run_results = list(zip(dim_run_results))
        all_results.append(dim_run_results)

    print('writing results to file:', PATH2DATA)
    with open(PATH2DATA, 'wb') as f:
        pickle.dump(all_results, f)
    print('...done.\n')


def generate_plots():
    def extract_data(results, data_idx):
        data = []
        for dim_run_res in results:
            y_naive = []
            y_horner = []
            for degree_run_res in dim_run_res:
                y_naive.append(degree_run_res[0][data_idx])
                y_horner.append(degree_run_res[1][data_idx])

            data.append((y_naive, y_horner))

        return data

    def extract_data_diff(results, data_idx):
        data = []
        for dim_run_res in results:
            y_diff = []
            for degree_run_res in dim_run_res:
                y1 = degree_run_res[0][data_idx]
                y2 = degree_run_res[1][data_idx]
                y_diff.append(abs(y1 - y2))

            data.append(y_diff)

        return data

    def extract_data_abs_horner(results, data_idx):
        data = []
        for dim_run_res in results:
            y_abs_horner = []
            for degree_run_res in dim_run_res:
                y = degree_run_res[1][data_idx]
                y_abs_horner.append(y)

            data.append(y_abs_horner)

        return data

    print('importing polynomial from file "{}"'.format(PATH2DATA))
    with open(PATH2DATA, 'rb') as f:
        all_results = pickle.load(f)

    print('plotting now...')

    labels = ['avg. setup time increase / s', 'avg. evaluation time reduction / s', 'avg. #operations reduction']
    file_names = ['setup_time_increase', 'eval_time_decrease', 'num_ops_decrease']

    # equal "spaced" colors
    color_idx = np.linspace(0, 1, MAX_DIM)
    cm = plt.cm.gist_rainbow
    # use_logarithmic = [True, True, False]

    for run_idx in range(3):
        print(run_idx, file_names[run_idx])

        label = labels[run_idx]

        fig, ax = plt.subplots()

        obj_handles = []
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        obj_handles.append(extra)

        # data = extract_data(all_results, run_idx)
        data = extract_data_diff(all_results, run_idx)
        plt.plot([], [], ' ', label="dimension")
        for dim, dim_run_data in zip(reversed(DIM_RANGE), reversed(data)):
            # y_naive, y_horner = dim_run_data

            # plt.plot(x, y, color=c, alpha=alpha, **kwargs)
            c = cm(color_idx[dim - 1])
            plt.semilogy(DEGREE_RANGE, dim_run_data, color=c, label=str(dim))
            # plt.semilogy(DEGREE_RANGE, y_horner, color=c)
            # plt.semilogy(DEGREE_RANGE, y_naive, 'o--', color=c)

        # plt.semilogy(t, np.exp(-t / 5.0))
        plt.xticks(DEGREE_RANGE)
        plt.xlabel('polynomial degree')
        plt.ylabel(label)
        # plt.title(label)
        plt.legend()
        plt.grid(True)

        export_plot(fig, file_names[run_idx])
        if SHOW_PLOTS:
            plt.show()

    # TODO plot relative time improvement, but relative to naive not horner!

    print('...done.')


# TODO also plot degree over dim
# TODO plot "lucrative after"

if __name__ == '__main__':
    # generate data
    # test_speed()

    generate_plots()
