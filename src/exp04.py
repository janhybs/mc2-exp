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
    sizes = a.size, a.size, a.size, t.size, t.size, t.size

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


def drop_outliers(data, threshold):
    """
    :param threshold: float quantile from 0 to 0.5
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
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

        lower_all = np.greater_equal(line, lower_bounds[mach])
        upper_all = np.greater_equal(upper_bounds[mach], line)

        if np.all(lower_all) and np.all(upper_all):
            new_data.append(line_orig.values)

    new_df = pd.DataFrame(new_data, columns=columns)
    print(data.shape[0], new_df.shape[0] / data.shape[0])
    return new_df


def inspect_machine(data, machine):
    values = data[data['machine'] == machine]
    del values['machine']
    # plt.plot(np.sort(values.values, axis=0)[::-1])
    plt.plot(values.values)
    plt.legend(values.columns.values)
    plt.show()


def calculate_eq(a, b, y, c, m, n):
    """
    :type a: numpy.core.multiarray.ndarray
    :type b: numpy.core.multiarray.ndarray
    :type c: numpy.core.multiarray.ndarray
    :type y: numpy.core.multiarray.ndarray
    :type m: numpy.core.multiarray.ndarray
    :type n: numpy.core.multiarray.ndarray
    :rtype: numpy.core.multiarray.ndarray
    """
    return a.T * c + b.T * m + y.T * n

# load data
data_orig = load_data('data_new.csv')
machines = data_orig['machine']
del data_orig['machine']
data_orig = norm(data_orig, a=0)
data_orig = norm(data_orig, a=1)
data_orig['machine'] = machines

# drop problematic architecture
data_orig = drop_machine(
    data_orig,
    machine='ajaxx')

# drop sensitive tests
data_orig = drop_test(
    data_orig,
    mem_l1=1, mem_l2=1, mem_l3=1, cpu_simple=1,
    mmn_s4=1, mem_ll=1)

# cut 5 % outliers
data_orig = drop_outliers(
    data_orig, 3.0/100)


d, di, ti, a, t, sizes = reveal_data(data_orig)

alpha_estimate = 1e-5
beta_estimate = 1e-4
gamma_estimate = 1e-3

cpu_estimate = 1e6
mem1_estimate = 1e6
mem2_estimate = 1e3

x0 = pack(
    (a.size, alpha_estimate),
    (a.size, beta_estimate),
    (a.size, gamma_estimate),
    (t.size, cpu_estimate),
    (t.size, mem1_estimate),
    (t.size, mem2_estimate),
).flatten()


def minimalize_function(x0, *args):
    _a, _b, _y, _c, _m, _n = unpack(args, *sizes)
    _di = calculate_eq(_a[:, x0], _b[:, x0], _y[:, x0], _c, _m, _n)
    return _di.flatten()

xtol = 3e-16
ftol = 3e-16
gtol = 3e-16

for i in range(1):
    popt, pcov = op.curve_fit(
        minimalize_function,
        xdata=ti,
        ydata=di.flatten(),
        p0=x0,
        bounds=(1e-6, 1e9),
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=2
    )

    ea, eb, ey, ec, em, en = unpack(popt, *sizes)
    print(pd.DataFrame([ea.flatten(), eb.flatten(), ey.flatten()], index=['alpha', 'beta', 'gamma'], columns=a.flatten()))
    print(pd.DataFrame([ec.flatten(), em.flatten(), en.flatten()], index=['cpus', 'mem1', 'mem2'], columns=t.flatten()))

    ed = calculate_eq(ea, eb, ey, ec, em, en)
    edi = calculate_eq(ea[:, ti], eb[:, ti], ey[:, ti], ec, em, en)

    abs_error, rel_error = get_error(d, ed)
    D = abs(rel_error) + 1e-16
    print(pd.DataFrame([D.flatten().min(), D.flatten().max(), D.flatten().mean()], index=['min', 'max', 'mean']))

    plt.colorbar(plt.matshow(D, norm=LogNorm(vmin=1e-4, vmax=1e-1)))
    plt.xticks(np.arange(t.size), t.flatten(), rotation='vertical')
    plt.yticks(np.arange(a.size), a.flatten())
    # plt.colorbar(plt.matshow(D))
    plt.savefig("l1_01.png")
    plt.show()