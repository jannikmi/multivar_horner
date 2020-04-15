import os
import pickle
import time
import timeit
from math import log10

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import seaborn as sns

from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial
from tests.test_helpers import rnd_settings_list, rnd_input_list, TEST_RESULTS_PICKLE

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
PLOTTING_DIR = os.path.abspath(os.path.join(os.path.pardir, 'plots'))
plt.rcParams.update({'font.size': 40})
SPEED_RUN_PICKLE = 'speed_results.pickle'


def get_plot_name(file_name='plot'):
    file_name = file_name.replace(' ', '_')
    return os.path.abspath(os.path.join(PLOTTING_DIR, file_name + '_' + str(time.time())[:-7] + '.png'))


def export_plot(fig, plot_title):
    fig.set_size_inches(EXPORT_SIZE_X, EXPORT_SIZE_Y, forward=True)
    plt.savefig(get_plot_name(plot_title), dpi=EXPORT_RESOLUTION)
    plt.clf()  # clear last figure


poly_settings_list = []
input_list = []
poly_class_instances = []


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


def time_preprocess(time, time_fmt='{:.2e}'):
    # valid_digits = 4
    # zero_digits = abs(min(0, int(log10(time))))
    # digits_to_print = zero_digits + valid_digits
    return time_fmt.format(time)


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
    if lucrative_after < 0:
        lucrative_after = '-'  # never lucrative

    entries = [str(dim), str(degree), time_preprocess(setup_time_naive),
               time_preprocess(setup_time_horner), str(setup_delta),
               time_preprocess(eval_time_naive), time_preprocess(eval_time_horner),
               str(eval_delta), str(num_ops_naive), str(num_ops_horner), ops_delta, str(lucrative_after)]
    print(template.format(*entries))

    return (setup_time_naive, eval_time_naive, num_ops_naive), (setup_time_horner, eval_time_horner, num_ops_horner)


# TODO
def run_speed_benchmark():
    if os.path.isfile(SPEED_RUN_PICKLE):
        print(f'result file {SPEED_RUN_PICKLE} is already present. skipping data generation.')
        return

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

    print('writing results to file:', SPEED_RUN_PICKLE)
    with open(SPEED_RUN_PICKLE, 'wb') as f:
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
                y_diff.append(y1 - y2)

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

    print('importing polynomial from file "{}"'.format(SPEED_RUN_PICKLE))
    with open(SPEED_RUN_PICKLE, 'rb') as f:
        all_results = pickle.load(f)

    print('plotting now...')

    # labels = ['avg. setup time increase / s', 'avg. evaluation time reduction / s', 'avg. #operations reduction']
    labels = ['time [s]', 'time [s]', 'avg. #operations reduction']
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
            # TODO own plot setting for each, operations save log...
            plt.plot(DEGREE_RANGE, dim_run_data, 'x:', color=c, label=str(dim), linewidth=3, markersize=15,
                     markeredgewidth=2.5)
            # plt.semilogy(DEGREE_RANGE, dim_run_data, 'x:', color=c, label=str(dim), linewidth=3, markersize=15,
            #              markeredgewidth=2.5)
            # plt.semilogy(DEGREE_RANGE, y_horner, color=c)
            # plt.semilogy(DEGREE_RANGE, y_naive, 'o--', color=c)

        # plt.semilogy(t, np.exp(-t / 5.0))
        plt.xticks(DEGREE_RANGE)
        plt.xlabel('polynomial max_degree')
        plt.ylabel(label)
        # plt.title(label)
        plt.legend()
        plt.grid(True)

        export_plot(fig, file_names[run_idx])
        if SHOW_PLOTS:
            plt.show()

    # TODO plot relative time improvement, but relative to naive not horner!

    print('...done.')


def remove_zeros(l):
    return [x if x != 0.0 else None for x in l]


def extract_numerical_error_horner(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return abs(p_x_horner - p_x_expected)


def extract_numerical_error_naive(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return abs(p_x_horner - p_x)


def has_nonzero_err_horner(result):
    return extract_numerical_error_horner(result) != 0.0


def has_nonzero_err_naive(result):
    return extract_numerical_error_naive(result) != 0.0


def extract_numerical_difference(result):
    err_naive = extract_numerical_error_naive(result)
    err_horner = extract_numerical_error_horner(result)
    return err_horner - err_naive


def extract_poly_properties(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return poly.dim, poly.max_degree, poly.num_ops, poly_horner.num_ops


def extract_num_ops_naive(result):
    poly_dim, poly_degree, num_ops_naive, num_ops_horner = extract_poly_properties(result)
    return num_ops_naive


def extract_num_ops_horner(result):
    poly_dim, poly_degree, num_ops_naive, num_ops_horner = extract_poly_properties(result)
    return num_ops_horner


def extract_num_ops_benefit(result):
    poly_dim, poly_degree, num_ops_naive, num_ops_horner = extract_poly_properties(result)
    return num_ops_horner - num_ops_naive


def has_nonzero_num_ops_horner(result):
    return extract_num_ops_horner(result) != 0


def compute_log(entry):
    if entry is None:
        return None
    return log10(entry)


def convert2log(l):
    return list(map(compute_log, l))


def plot_numerical_error():
    try:
        with open(TEST_RESULTS_PICKLE, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print('no data of numerical errors found. run the numerical tests in test_it.py first')
        return

    def filter_results(filter_fct, results):
        results_filtered = list(filter(filter_fct, results))
        print(f'filtering data: out of {len(results)} entries {len(results_filtered)} remain')
        return results_filtered

    # TODO plot
    # histogram numerical benefit
    # numerical benefit 2 num_ops_benefit
    # numerical benefit 2 dim, degree als color

    results = filter_results(has_nonzero_err_horner, results)
    results = filter_results(has_nonzero_err_naive, results)
    results = filter_results(has_nonzero_num_ops_horner, results)

    nr_datapoints = len(results)

    numerical_difference = list(map(extract_numerical_difference, results))
    numerical_err_naive = list(map(extract_numerical_error_naive, results))
    numerical_err_horner = list(map(extract_numerical_error_horner, results))
    num_ops = list(map(extract_num_ops_naive, results))
    # num_ops_log = convert2log(num_ops)

    num_ops_horner = list(map(extract_num_ops_horner, results))

    # num_ops_horner_log = convert2log(num_ops_horner)
    # numerical_err_naive = convert2log(numerical_err_naive)
    # numerical_err_horner = convert2log(numerical_err_horner)

    plot = sns.distplot(numerical_difference, kde=False, norm_hist=False)
    print('max numerical errors:', max(numerical_difference), max(numerical_err_naive), max(numerical_err_horner))
    # plot = sns.distplot([numerical_err_horner, numerical_err_naive], kde=False, norm_hist=False)
    plot.set_xlabel('numerical error')
    title = 'difference in numerical error'  # TODO
    plot.set_title(title)
    export_plot(plt.gcf(), plot_title=title)

    # TODO title, axes
    # TODO lin plot
    # TODO exponents extract growth, from ALL data
    category = len(numerical_err_naive) * ['canonical form'] + len(numerical_err_horner) * ['Horner factorisation']
    plot = sns.scatterplot(x=num_ops + num_ops, y=numerical_err_naive + numerical_err_horner,
                           hue=category, data=None, alpha=0.3)
    # .set_axis_labels('log number of operation ', 'numerical error')
    title = 'numerical error comparison'
    # plot.set_title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of operations (canonical form)')
    plt.ylabel('numerical error')
    plt.grid(True)
    # TODO ticks
    # r'$\mu=100,\ \sigma=15$'
    fig = plt.gcf()
    export_plot(fig, plot_title=title)

    # TODO evaluate saved num ops
    # TODO estimate formula, fit model


# TODO also plot max_degree over dim
# TODO plot "lucrative after"

if __name__ == '__main__':
    run_speed_benchmark()  # generate data
    generate_plots()  # create plots with data

    plot_numerical_error()
