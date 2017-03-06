#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

import re
import numpy as np
import pandas as pd
import scipy.optimize as op
from sklearn import preprocessing as sk
import tul.flow123d.data.loader as loader

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def apply_to_panda(method, data, *args, **kwargs):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        cols = data.columns.values
        rows = data.index.values
        normed = method(data, *args, **kwargs)
        return pd.DataFrame(normed, columns=cols, index=rows)
    return method(data, *args, **kwargs)


def norm(data, a=0, n='l1', return_norm=False):
    """
    :type data: pandas.core.frame.DataFrame
    """
    if return_norm:
        try:
            _, calculated_norms = sk.normalize(data.values, norm=n, axis=a, return_norm=True)
            return apply_to_panda(sk.normalize, data, norm=n, axis=a), calculated_norms
        except:
            _, calculated_norms = sk.normalize(data, norm=n, axis=a, return_norm=True)
            return apply_to_panda(sk.normalize, data, norm=n, axis=a), calculated_norms
    return apply_to_panda(sk.normalize, data, norm=n, axis=a)


def scale(data):
    # sk.scale()
    return apply_to_panda(sk.scale, data, with_mean=False, with_std=True)


def scale_min_max(data):
    # sk.scale()
    scaler = sk.MinMaxScaler()
    return apply_to_panda(scaler.fit_transform, data)


def unpack(source, sizes, as_matrix=True):
    index = 0
    for size in sizes:
        yield np.array([source[index:index+size]]) if as_matrix else source[index:index+size]
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


def drop_row(data, eq=False, **kwargs):
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


def drop_col(data, *args, **kwargs):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    for arg in args:
        data = data.drop(arg, axis=1)

    for key, value in kwargs.items():
        data = data.drop(key, axis=value)
    return data


def calculate_eq(a, b, c, m):
    """
    :type a: numpy.core.multiarray.ndarray
    :type b: numpy.core.multiarray.ndarray
    :type c: numpy.core.multiarray.ndarray
    :type m: numpy.core.multiarray.ndarray
    :rtype: numpy.core.multiarray.ndarray
    """
    return a.T * c + b.T * m


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


def drop_outliers2(data, threshold, threshold2=1):
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

    stat = pd.DataFrame()
    new_data = []
    for lid, line_orig in data.iterrows():
        line = line_orig.copy()
        mach = line['machine']
        del line['machine']

        lower_all = np.greater(line, lower_bounds[mach])
        upper_all = np.greater(upper_bounds[mach], line)

        oks = np.sum(upper_all.values.astype(int) & lower_all.values.astype(int))

        if np.all(lower_all) and np.all(upper_all):
            new_data.append(line_orig.values)
        elif oks/lower_all.size > threshold2:
            new_data.append(line_orig.values)
        else:
            if mach not in stat:
                stat[mach] = np.zeros(lower_all.shape)
            stat[mach] = stat[mach].values + lower_all.values.astype(int) + upper_all.values.astype(int)
            # print(upper_all.values.astype(int) & lower_all.values.astype(int))
            pass
        # print(upper_all.values.astype(int) & lower_all.values.astype(int))

    new_df = pd.DataFrame(new_data, columns=columns)
    print(data.shape[0], new_df.shape[0], new_df.shape[0] / data.shape[0])
    return new_df


def load_data(version=None):
    def match_func(x):
        match_duration = re.compile(r'.*\.duration')
        return match_duration.match(x)

    def rename_func(x):
        rename_field = re.compile(r'(.+)\.(.+)')
        return rename_field.sub(r'\1', x)

    sample = list(loader.find_bench(limit=1))[0]
    project = loader.create_project(loader.flatten_dict(sample).keys(), match_func, rename_func)
    # project['hostname'] = '$hostname'
    project['machine'] = '$machine'
    project['_id'] = 0

    match = None if not version else dict(version=version)

    result = list(loader.aggregate_bench(project=project, match=match))
    return pd.DataFrame(result)


def construct_mean_matrix(data):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    d = []
    a = sorted(list(set(data['machine'].values)))

    for mach in a:
        submatrix = data[data['machine'] == mach]
        submatrix = submatrix.drop('machine', axis=1)
        d.append(np.mean(submatrix.values, axis=0))
    d = np.array(d)
    return d


def reveal_data(data):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    data = data.sort_values(by=('machine'))

    # normalize
    machines = data['machine']
    del data['machine']
    data = norm(data, a=0)
    data = norm(data, a=1)
    data['machine'] = machines

    machine = np.array([data['machine'].values])
    a = np.array([sorted(list(set(data['machine'])))])
    ti = [np.where(a == x)[1][0] for x in machine.flatten()]
    d = construct_mean_matrix(data)

    data = data.drop('machine', axis=1)
    t = np.array([sorted(data.columns.values)])
    di = data.values

    return d, di, a, t, ti


def plot_error(E, a=None, t=None, vmin=1e-3, vmax=1e-1):
    plt.colorbar(plt.matshow(E, norm=LogNorm(vmin=vmin, vmax=vmax)))
    if t is not None:
        if type(t) is list:
            plt.xticks(np.arange(len(t)), t, rotation='vertical')
        else:
            plt.xticks(np.arange(t.size), t.flatten(), rotation='vertical')

    if a is not None:
        if type(a) is list:
            plt.yticks(np.arange(len(a)), a)
        else:
            plt.yticks(np.arange(a.size), a.flatten())

    plt.show()
