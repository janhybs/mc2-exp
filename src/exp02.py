#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs
import tul.flow123d.data.loader as loader
from sklearn import preprocessing as sk
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import scipy.optimize as op
import numpy as np
np.set_printoptions(precision=4)

np.set_printoptions(precision=4, suppress=False)
np.random.seed(1234)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def apply_to_panda(method, data, *args, **kwargs):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        cols = data.columns.values
        rows = data.index.values
        norm = method(data, *args, **kwargs)
        return pd.DataFrame(norm, columns=cols, index=rows)
    return method(data, *args, **kwargs)


def norm(data, a=0, n='l1'):
    """
    :type data: pandas.core.frame.DataFrame
    """
    return apply_to_panda(sk.normalize, data, norm=n, axis=a)


def scale(data):
    # sk.scale()
    return apply_to_panda(sk.scale, data, with_mean=False, with_std=True)


def scale_min_max(data):
    # sk.scale()
    scaler = sk.MinMaxScaler()
    return apply_to_panda(scaler.fit_transform, data)


def load_data(filename='data_new.csv'):
    if os.path.exists(filename):
        print('data loaded from file')
        return pd.DataFrame.from_csv(filename)
    else:
        def match_func(x):
            match_duration = re.compile(r'.*\.duration')
            return match_duration.match(x)

        def rename_func(x):
            rename_field = re.compile(r'(.+)\.(.+)')
            return rename_field.sub(r'\1', x)

        # create project structure based on one sample
        sample = list(loader.find_bench(limit=1))[0]
        project = loader.create_project(loader.flatten_dict(sample).keys(), match_func, rename_func)
        # project['hostname'] = '$hostname'
        project['machine'] = '$machine'
        project['_id'] = 0

        result = list(loader.aggregate_bench(project=project))
        df = pd.DataFrame(result)
        df.to_csv(filename)
        return df


def unpack(source, *sizes):
    index = 0
    for size in sizes:
        yield np.array([source[index:index+size]])
        index = index+size


def pack(*items):
    result = list()
    for size, value in items:
        result.extend([value] * size)
    return np.array([result])


def get_error(reference, estimate):
    """Computes absolute and relative error given reference and estimate
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)
    """
    abs_error = estimate - reference
    rel_error = abs(np.divide(abs_error, reference))
    return abs_error, rel_error


def reveal_data(data_orig):
    data = data_orig.copy()
    machine = np.array([data['machine'].values])
    a = machine_set = np.array([list(set(data['machine']))])
    ti = [np.where(a == x)[1][0] for x in machine.flatten()]

    data = data.drop('machine', axis=1)
    t = np.array([data.columns.values])
    di = data.values
    sizes = a.size, a.size, t.size, t.size

    d = []
    for mach in machine_set.flatten():
        submatrix = data[data_orig['machine'] == mach]
        submatrix = submatrix
        d.append(np.mean(submatrix.values, axis=0))
        # print(mach)
        # m = pd.DataFrame(np.array([
        #     np.mean(submatrix.values, axis=0),
        #     np.median(submatrix.values, axis=0),
        #     np.min(submatrix.values, axis=0),
        #     np.max(submatrix.values, axis=0),
        #     np.std(submatrix.values, axis=0),
        # ]).T, columns=['mean', 'median', 'min', 'max', 'std'], index=t.flatten())
        # print(m)
        # break
    d = np.array(d)
    return d, di, ti, a, t, sizes


def drop_machine(data, eq=False, **kwargs):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    for key, value in kwargs.items():
        if eq:
            data = data[data[key] == value]
        else:
            data = data[data[key] != value]
    return data


def drop_test(data, **kwargs):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    for key, value in kwargs.items():
        data = data.drop(key, axis=value)
    return data


def drop_results(data, threshold):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    print(data.shape)
    columns = data.columns.values
    lower_bounds = dict()
    upper_bounds = dict()
    for mach in set(data['machine'].values):
        lower_bounds[mach] = data[data['machine'] == mach].quantile(threshold)
        upper_bounds[mach] = data[data['machine'] == mach].quantile(1 - threshold)

    new_data = []
    for lid, line_orig in data.iterrows():
        line = line_orig.copy()
        mach = line['machine']
        del line['machine']

        lower_all = np.greater(line, lower_bounds[mach])
        upper_all = np.greater(upper_bounds[mach], line)

        if np.all(lower_all) and np.all(upper_all):
            new_data.append(line_orig.values)

    new_df = pd.DataFrame(new_data, columns=columns)
    print(new_df.shape)
    return new_df


def inspect_machine(data, machine):
    values = data[data['machine'] == machine]
    del values['machine']
    # plt.plot(np.sort(values.values, axis=0)[::-1])
    plt.plot(values.values)
    plt.legend(values.columns.values)
    plt.show()

# load data
data_orig = load_data('data_new.csv')
machines = data_orig['machine']
del data_orig['machine']
data_orig = norm(data_orig)
data_orig['machine'] = machines

data_orig = drop_machine(
    data_orig,
    machine='ajaxx')

data_orig = drop_test(
    data_orig,
    mem_l1=1, mem_l2=1, mem_l3=1, cpu_simple=1,
    mmn_s4=1, mem_ll=1)

# inspect_machine(data_orig, 'tarkil')
data_orig = drop_results(data_orig, 5.0/100)  # cut 5 % outliers
# inspect_machine(data_orig, 'tarkil')


# data_orig = data_orig.drop('mem_l1', axis=1)
# data_orig = data_orig.drop('mem_l2', axis=1)
# data_orig = data_orig.drop('mem_l3', axis=1)
# data_orig = data_orig.drop('cpu_simple', axis=1)
# data_orig = data_orig[['machine', 'cpu_hash', 'cpu_simple', 'cpu_md5', 'mem_l1', 'mem_l2', 'mem_l3', 'mem_ll']]
# data_orig = data_orig[['machine', 'mmn_s1', 'mmn_s2', 'mmn_s3', 'mmn_s4', 'mms_s1', 'mms_s2', 'mms_s3', 'mms_s4', 'mvs_s1', 'mvs_s2', 'mvs_s3', 'mvs_s4']]
# data_orig = data_orig[(data_orig['machine'] != 'ajax')]
# data_orig = data_orig[(data_orig['machine'] != 'mudrc')]
# data_orig = data_orig[(data_orig['machine'] == 'hildor') | (data_orig['machine'] == 'tarkil')]
#
# columns = data_orig.columns.values
# data = []
# for m in set(data_orig['machine']):
#     values = data_orig[data_orig['machine'] == m]
#     row = dict()
#     for c in columns:
#         if c == 'machine':
#             row[c] = m
#             continue
#         lower = values[c].quantile(.2)
#         upper = values[c].quantile(.8)
#         trimmed = values[(values[c] > lower) & (values[c] < upper)][c]
#         row[c] = np.mean(trimmed.values.flatten())
#         print(lower, upper, c, m, trimmed)
#     data.append(row)
#     # print(values.shape)
# # print(columns)
#
# print(data_orig.quantile(.1))
# print(data_orig.quantile(.9))
# exit(0)
d, di, ti, a, t, sizes = reveal_data(data_orig)


# for tt in t.flatten():
#     print('    1e6,  # %s' % tt)
# print(t)
# data = data[['machine', 'mmn_s1', 'mmn_s2', 'mmn_s3', 'mmn_s4']]
# data = data[['machine', 'mem_l1', 'mem_l2', 'mem_l3', 'mem_ll']]


alpha_estimate = 1e-6
beta_estimate = 1e-5
cpu_estimate = 1e6
memory_estimate = 1e6

x0 = pack(
    (a.size, alpha_estimate),
    (a.size, beta_estimate),
    (t.size, cpu_estimate),
    (t.size, memory_estimate),
).flatten()

bounds_l = pack(
    (a.size, alpha_estimate/2),
    (a.size, beta_estimate/2),
    (t.size, cpu_estimate/2),
    (t.size, memory_estimate/200),
).flatten()
bounds_r = pack(
    (a.size, alpha_estimate*2),
    (a.size, beta_estimate*2),
    (t.size, cpu_estimate*2),
    (t.size, memory_estimate*200),
).flatten()

# bounds = [
# ]
# bounds_l = [x/10 for x in bounds]
# bounds_r = [x*10 for x in bounds]


def fit_f(x0, *args):
    _a, _b, _c, _m = unpack(args, *sizes)
    _di = _a[:, x0].T * _c + _b[:, x0].T * _m
    return _di.flatten()

xtol = 3e-16
ftol = 3e-16
gtol = 3e-16

for i in range(1):
    # popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=bounds, bounds=[bounds_l, bounds_r])
    popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=x0, bounds=(1e-6, 1e8), xtol=xtol, ftol=ftol, gtol=gtol, verbose=0)
    # popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=bounds, bounds=[bounds_l, bounds_r], xtol=xtol, ftol=ftol, gtol=gtol, verbose=2)
    # popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=x0,  method='lm', maxfev=1000)
    # print(pd.DataFrame(popt))
    # for b in popt:
    #     print(b, ',')

    ea, eb, ec, em = unpack(popt, *sizes)
    print(pd.DataFrame([ea.flatten(), eb.flatten()], index=['alpha', 'beta'], columns=a.flatten()))
    print(pd.DataFrame([ec.flatten(), em.flatten()], index=['cpus', 'mems'], columns=t.flatten()))

    ed = ea.T * ec + eb.T * em
    edi = ea[:, ti].T * ec + eb[:, ti].T * em

    abs_error, rel_error = get_error(d, ed)
    D = abs(rel_error) + 1e-16
    print(pd.DataFrame([D.flatten().min(), D.flatten().max(), D.flatten().mean()], index=['min', 'max', 'mean']))

    plt.colorbar(plt.matshow(D, norm=LogNorm(vmin=1e-4, vmax=1e-1)))
    plt.xticks(np.arange(t.size), t.flatten(), rotation='vertical')
    plt.yticks(np.arange(a.size), a.flatten())
    # plt.colorbar(plt.matshow(D))
    plt.show()

# # x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
#
# #
#
# a_size = 6
# t_size = 12
# n_size = 5000

# a = (np.random.rand(1, a_size) / 2 + 0.5) * 1e6
# t = (np.random.rand(1, t_size) / 2 + 0.5) * 1e-3
#
# # result
# d = a.T * t
# print(d)
#
#
# ti = np.random.choice(np.arange(a.size), n_size)
# ai = a[:, ti]
# di = ai.T * t
# print(di)
#
# v = np.hstack((a, t))
# x0 = np.array(a.size * [1e6] + t.size * [1e-3])
#
#
# def fit_f(x0, *args):
#     _a = np.array([args[:a.size]])
#     _t = np.array([args[a.size:]])
#     _di = _a[:, x0].T * _t
#     return _di.flatten()
#
#
# bounds_l = a.size * [1e5] + t.size * [1e-4]
# bounds_r = a.size * [1e7] + t.size * [1e-2]
#
# popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=x0, bounds=[bounds_l, bounds_r], sigma=1e-6)
# print(pd.DataFrame(popt))
# print(pd.DataFrame(v.flatten()))
#
# ea = np.array([popt[:a.size]])
# et = np.array([popt[a.size:]])
# ed = ea.T * et
# edi = ea[:, ti].T * et
#
#
# error_d = sum(abs(ed - d).flatten())
# error_di = sum(abs(edi - di).flatten())
# print(ed - d, error_d)
# print(edi - di, error_di)
#
# # alphas = np.floor(np.random.random((5, 1)) * 1000) - 500
# betas = np.floor(np.random.random((5, 1)) * -1000) + 500
# data = alphas * betas.T
# print(data.flatten())
# print(data)
# print(alphas)
# print(betas)
#
#
# def min_f(x):
#     return x[0]**2 + x[0] + 2
#
# r = op.minimize(min_f, [2.])
# print(r)
#
# x = np.arange(-1000, 1000) / 100.
# y = [min_f([xi]) for xi in x]
# plt.plot(x, y)
# plt.show()

# def min_f(x0, X):
#     return x0
#
#
# r = op.minimize(min_f, 10 * [0.1], data.flatten())
# op.curve_fit()
# print(r)
# def rosen(x):
#     """The Rosenbrock function"""
#     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
#
# x0 = np.array([1.3, 0.7, 0.8, 1000.9, 1.2])
# res = op.minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
#
# print(res)
# print(res.x)