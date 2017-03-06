#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs
from collections import namedtuple

import tul.flow123d.data.loader as loader
from sklearn import preprocessing as sk
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import scipy.optimize as op
import math
import numpy as np
np.set_printoptions(precision=4)

np.set_printoptions(precision=4, suppress=False)
np.random.seed(1234)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


from tul.flow123d.utils.stats import norm, drop_row, drop_col, drop_outliers, load_data, reveal_data, plot_error, get_error, drop_outliers2
from tul.flow123d.minimize.min2term import estimate_all, estimate_cm


drops = 'ajax',
cols = ['mem_l1', 'mem_l2', 'mem_l3', 'cpu_simple', 'mmn_s4', 'mem_ll']
# cols = [
#     'mem_l1', 'mem_l2', 'mem_l3', 'mem_ll',
#     'mms_s1', 'mvs_s1'
#     # 'cpu_simple',
#     # 'mmn_s1', 'mmn_s2', 'mmn_s3', 'mmn_s4',
#     ]
outliers = 5


def inspect_test_on_mach(data, mach, test=None):
    sub = data[data['machine'] == mach][test]
    plt.title(test + ' on ' + mach)
    plt.plot(sub)
    plt.show()


def inspect_mach(data, mach):
    sub = data[data['machine'] == mach]
    sub = sub.drop('machine', axis=1)
    sub = norm(sub)
    cols = sub.columns.values

    x, y = math.ceil(math.sqrt(len(cols))), math.floor(math.sqrt(len(cols)))
    if x*y < cols.size:
        x += 1
    le = 1./sub.shape[0]

    for col in cols:
        plt.subplot(x, y, 1 + np.where(cols == col)[0][0])
        plt.ylim((le*0.5, le/0.5))
        plt.title(col)
        plt.plot(np.arange(sub[col].size), sub[col])
    plt.show()


def estimate(version, inspect=()):
    data = load_data(version=version)

    if inspect:
        if len(inspect) == 2:
            inspect_test_on_mach(data, *inspect)
        else:
            inspect_mach(data, *inspect)

    # drop problematic architecture
    for d in drops:
        data = drop_row(
            data,
            machine=d)

    # drop sensitive tests
    data = drop_col(data, *cols)

    if inspect:
        if len(inspect) == 2:
            inspect_test_on_mach(data, *inspect)
        else:
            inspect_mach(data, *inspect)

    # cut 5 % outliers
    data = drop_outliers(
        data, outliers/100)

    if inspect:
        if len(inspect) == 2:
            inspect_test_on_mach(data, *inspect)
        else:
            inspect_mach(data, *inspect)

    # data = data.sort_values(by=['mvs_s3'])
    # data = drop_col(
    #     data,
    #     machine=1)
    #
    # data = norm(data, a=0)
    # data = norm(data, a=1)
    #
    # plt.colorbar(plt.matshow(data.values.T))
    # plt.show()
    # exit(0)

    d, di, a, t, ti = reveal_data(data)
    ea, eb, ec, em, ed, E, coef_ab, coef_cm = estimate_all(d, di, a, t, ti)

    Result = namedtuple('Estimate', ['d', 'di', 'a', 't', 'ti', 'ea', 'eb', 'ec', 'em', 'ed', 'E', 'coef_ab', 'coef_cm'])
    return Result(d, di, a, t, ti, ea, eb, ec, em, ed, E, coef_ab, coef_cm)


def estimate_from(version, alpha, beta, inspect=()):
    data = load_data(version=version)

    if inspect:
        if len(inspect) == 2:
            inspect_test_on_mach(data, *inspect)
        else:
            inspect_mach(data, *inspect)

    # drop problematic architecture
    for d in drops:
        data = drop_row(data, machine=d)

    # drop sensitive tests
    data = drop_col(data, *cols)

    # cut 5 % outliers
    data = drop_outliers2(data, outliers/100, .7)

    d, di, a, t, ti = reveal_data(data)
    ea, eb, ec, em, ed, E, coef_ab, coef_cm = estimate_cm(d, di, a, t, ti, alpha, beta)

    Result = namedtuple('Estimate', ['d', 'di', 'a', 't', 'ti', 'ea', 'eb', 'ec', 'em', 'ed', 'E', 'coef_ab', 'coef_cm'])
    return Result(d, di, a, t, ti, ea, eb, ec, em, ed, E, coef_ab, coef_cm)

baseline = estimate(version='1.1.1', inspect=('luna', 'mmn_s3'))
# baseline = estimate(version='1.1.1')
estimates = [
    baseline
]

# plot_error(baseline.E, baseline.a, baseline.t)

# exit(0)

estimates.extend([
    estimate_from(version='1.1.2', alpha=baseline.ea, beta=baseline.eb),
    estimate_from(version='1.1.3', alpha=baseline.ea, beta=baseline.eb),
    estimate_from(version='1.1.4', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.0.3', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.0.4', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.0.5', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.0.6', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.0.7', alpha=baseline.ea, beta=baseline.eb),
    # estimate_from(version='1.1.1', alpha=baseline.ea, beta=baseline.eb),
])
le = 1/len(estimates)
# plot_error(estimates[3].E, estimates[3].a, estimates[3].t)

# plt.subplot(211)
# inspect_test(ba, 'mvs_s3')
# plt.subplot(212)
# inspect_test(version_105, 'mvs_s3')
# plt.show()
for e in estimates:
    print(e.coef_cm)

# e = get_error(version_102.ec, version_105.ec)
# plot_error(e[1], version_102.a, version_102.t)
# plot_error(e[1])


i = 1
for test in baseline.t.flatten():
    plt.subplot(5, 5, i)

    coefs = np.array([e.coef_cm[test] for e in estimates])
    coefs = norm(coefs, a=0)

    print(test)
    a = plt.plot(coefs, lw=2, marker='.', ms=15)
    a[0].set_color('#5DA5DA')
    a[1].set_color('#FAA43A')
    plt.ylim((le*0.5, le/0.5))
    plt.xticks(np.arange(coefs.shape[0]), ['v'+str(i+1) for i in range(coefs.shape[0])])
    plt.title(test)
    # plt.legend(['alpha', 'beta'], loc=7)
    i += 1
plt.savefig('regression.png')
plt.show()

