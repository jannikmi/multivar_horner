import os
import pickle
import time
import timeit
from math import log10

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import seaborn as sns
from scipy.special import binom
import pandas as pd

from multivar_horner.multivar_horner import HornerMultivarPolynomial, MultivarPolynomial
from tests.test_helpers import rnd_settings_list, rnd_input_list
from tests.test_settings import TEST_RESULTS_PICKLE, DIM_RANGE, DEGREE_RANGE, MAX_DIMENSION

# SPEED TESTS
# TODO allow other setting ranges

NR_SAMPLES = 100

EXPORT_RESOLUTION = 300  # dpi
EXPORT_SIZE_X = 19.0  # inch
EXPORT_SIZE_Y = 11.0  # inch
SHOW_PLOTS = False
PLOTTING_DIR = os.path.abspath(os.path.join(os.path.pardir, 'plots'))
# plt.rcParams.update({'font.size': 35})
SPEED_RUN_PICKLE = 'speed_results.pickle'


def get_plot_name(file_name='plot'):
    file_name = file_name.replace(' ', '_')
    return os.path.abspath(os.path.join(PLOTTING_DIR, file_name + '_' + str(time.time())[:-7] + '.png'))


def export_plot(fig, plot_title):
    # fig.set_size_inches(EXPORT_SIZE_X, EXPORT_SIZE_Y, forward=True)
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
    color_idx = np.linspace(0, 1, MAX_DIMENSION)
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
        plt.xlabel('polynomial maximal_degree')
        plt.ylabel(label)
        # plt.title(label)
        plt.legend()
        plt.grid(True)

        export_plot(fig, file_names[run_idx])
        if SHOW_PLOTS:
            plt.show()

    # TODO plot relative time improvement, but relative to naive not horner!

    print('...done.')


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
    return poly.dim, poly.total_degree, poly.num_ops, poly_horner.num_ops


def extract_dimensionality(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return poly.dim


def extract_total_degree(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return poly.total_degree


def extract_maximal_degree(result):
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return poly.maximal_degree


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


def extract_nr_coeffs(result):  # identical in both polynomial representations!
    poly, poly_horner, p_x_expected, p_x, p_x_horner = result
    return poly_horner.num_monomials


def extract_sparsity(result):
    nr_coeffs = extract_nr_coeffs(result)
    dim = extract_dimensionality(result)
    deg = extract_total_degree(result)
    nr_coeffs_max = binom(dim + deg, deg)
    return nr_coeffs / nr_coeffs_max


def remove_zeros(l):
    return [x if x != 0.0 else None for x in l]


def rmv_zeros(*args):
    out = []
    for entries in zip(*args):
        if not 0.0 in entries:
            out.append(entries)

    return list(zip(*out))


def average_discrete(l1, l2):
    """
    average the entries of l1 where l2 is identical
    """
    assert len(l1) == len(l2)
    l1 = np.array(l1)
    l2 = np.array(l2)
    unique_entries = np.unique(l2)
    avgs = np.array([np.average(l1[l2 == e]) for e in unique_entries])
    assert len(avgs) == len(unique_entries)
    return avgs, unique_entries


def make_relative(a, b):
    return b / a


# def compute_log(entry):
#     if entry is None:
#         return None
#     return log10(entry)
#
#
# def convert2log(l):
#     return list(map(compute_log, l))


def plot_num_err_heatmap(results):
    dims = list(map(extract_dimensionality, results))
    # degs = list(map(extract_total_degree, results))
    degs = list(map(extract_maximal_degree, results))
    numerical_err_naive = list(map(extract_numerical_error_naive, results))
    numerical_err_horner = list(map(extract_numerical_error_horner, results))

    attr_name_numerical_err_naive = 'numerical error naive'
    attr_name_numerical_err_horner = 'numerical error horner'
    attr_name_numerical_err_rel = 'relative numerical error'
    attr_name_dim = 'dimension'
    attr_name_deg = 'degree'

    df = pd.DataFrame({
        attr_name_numerical_err_naive: numerical_err_naive,
        attr_name_numerical_err_horner: numerical_err_horner,
        attr_name_deg: degs,
        attr_name_dim: dims,
    })

    df_avg = pd.DataFrame()

    # average for dim and degree
    for dim in DIM_RANGE:
        for deg in DEGREE_RANGE:
            selection_idxs = (df[attr_name_dim] == dim) & (df[attr_name_deg] == deg)
            if not selection_idxs.any():
                continue
            df_selected = df[selection_idxs]
            avg_num_err_naive = df_selected[attr_name_numerical_err_naive].mean()
            avg_num_err_horner = df_selected[attr_name_numerical_err_horner].mean()
            if avg_num_err_naive == avg_num_err_horner:
                avg_num_err_naive_rel = 1.0
            else:
                avg_num_err_naive_rel = avg_num_err_naive / avg_num_err_horner

            entry = pd.Series({
                attr_name_numerical_err_rel: avg_num_err_naive_rel,
                attr_name_numerical_err_naive: avg_num_err_naive,
                attr_name_numerical_err_horner: avg_num_err_horner,
                attr_name_deg: deg,
                attr_name_dim: dim,
            })
            df_avg = df_avg.append(entry, ignore_index=True)

    # sns.relplot(x=attr_name_dim, y=attr_name_deg, data=df_avg,
    #             hue=attr_name_numerical_err_rel,
    #             size=attr_name_numerical_err_rel,
    #             sizes=(100,300),
    #             )
    heatmap_data = df_avg.pivot(attr_name_deg, attr_name_dim, attr_name_numerical_err_rel)
    heatmap_data = heatmap_data.iloc[::-1]  # reverse

    ax = sns.heatmap(heatmap_data, annot=True,
                     # fmt="d",
                     linewidths=.5,
                     cmap="YlGnBu",
                     )
    # plt.xlabel()
    title = 'numerical error heatmap'
    fig = plt.gcf()
    export_plot(fig, plot_title=title)


def plot_num_error_growth_comparison(results):
    numerical_err_naive = list(map(extract_numerical_error_naive, results))
    numerical_err_horner = list(map(extract_numerical_error_horner, results))
    num_coeffs = list(map(extract_nr_coeffs, results))

    # average first! before removing zeros
    numerical_err_naive_avg, num_coeffs_unique_naive = average_discrete(numerical_err_naive, num_coeffs)
    numerical_err_horner_avg, num_coeffs_unique_horner = average_discrete(numerical_err_horner, num_coeffs)

    numerical_err_naive_avg, num_coeffs_unique_naive = rmv_zeros(numerical_err_naive_avg, num_coeffs_unique_naive)
    numerical_err_horner_avg, num_coeffs_unique_horner = rmv_zeros(numerical_err_horner_avg, num_coeffs_unique_horner)

    category = len(num_coeffs_unique_naive) * ['canonical form'] + \
               len(num_coeffs_unique_horner) * ['Horner factorisation']

    attr_name_representation = 'representation'
    attr_name_num_coeff = 'number of coefficients'
    attr_name_numerical_err = 'average numerical error'
    repr_name_horner = 'Horner factorisation'
    repr_name_naive = 'canonical form'

    df = pd.DataFrame()

    df_naive = pd.DataFrame({attr_name_num_coeff: num_coeffs_unique_naive,
                             attr_name_numerical_err: numerical_err_naive_avg,
                             attr_name_representation: repr_name_naive})
    df_naive[attr_name_representation] = repr_name_naive

    df_horner = pd.DataFrame({attr_name_num_coeff: num_coeffs_unique_horner,
                              attr_name_numerical_err: numerical_err_horner_avg, })
    df_horner[attr_name_representation] = repr_name_horner

    df = df.append(df_naive, ignore_index=True)
    df = df.append(df_horner, ignore_index=True)

    plot = sns.scatterplot(x=attr_name_num_coeff, y=attr_name_numerical_err, hue=attr_name_representation, data=df,
                           alpha=0.8)
    # plot = sns.relplot(x=attr_name_num_coeff, y=attr_name_numerical_err, hue=attr_name_representation,
    #                    style=attr_name_representation,
    #                    kind="line", data=df,)
    # plt.legend()
    # plot.add_legend(bbox_to_anchor=(1.05, 0), loc=2, borderaxespad=0.)
    title = 'avg numerical error VS nr coeff'
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    fig = plt.gcf()
    export_plot(fig, plot_title=title)

def plot_num_coeffs2num_ops(results):

    # compare the number of operations to the number of coefficients
    # color by dimensions
    dim = list(map(extract_dimensionality, results))
    num_ops_naive = list(map(extract_num_ops_naive, results))
    num_ops_horner = list(map(extract_num_ops_horner, results))
    num_coeffs = list(map(extract_nr_coeffs, results))

    # TODO rainbow
    # equally "spaced" colors
    color_idx = np.linspace(0, 1, MAX_DIMENSION)
    cm = plt.cm.gist_rainbow
    # c = cm(color_idx[dim - 1])
    plot = sns.scatterplot(x=num_ops_naive, y=num_coeffs, hue=dim, data=None)
    title = 'ops naive VS coeffs'
    # plot.set_title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of operations (canonical form)')
    plt.ylabel('number of coefficients')
    plt.grid(True)
    # TODO ticks
    # r'$\mu=100,\ \sigma=15$'
    fig = plt.gcf()
    export_plot(fig, plot_title=title)

    # compare the number of operations to the number of coefficients
    # color by dimensions
    num_ops_horner, num_coeffs1, dim1 = rmv_zeros(num_ops_horner, num_coeffs, dim)
    plot = sns.scatterplot(x=num_ops_horner, y=num_coeffs1, hue=dim1, data=None)
    title = 'ops horner VS coeffs'
    # plot.set_title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of operations (Horner factorisation)')
    plt.ylabel('number of coefficients')
    plt.grid(True)
    fig = plt.gcf()
    export_plot(fig, plot_title=title)

def plot_numerical_error():
    sns.set_context("paper")

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


    # results_filtered_naive = filter_results(has_nonzero_err_naive, results)
    # results_filtered_horner = filter_results(has_nonzero_err_horner, results)

    plot_num_err_heatmap(results)
    plot_num_error_growth_comparison(results)
    plot_num_coeffs2num_ops(results)
    


# TODO evaluate saved num ops
# TODO plot "lucrative after"


if __name__ == '__main__':
    # TODO
    # run_speed_benchmark()  # generate data
    # generate_plots()  # create plots with data

    plot_numerical_error()
