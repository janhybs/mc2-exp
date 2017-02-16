#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs
import tul.flow123d.data.loader as loader
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import scipy.optimize as op
import numpy as np
np.set_printoptions(precision=4)


def load_data(filename='data.csv'):
    if os.path.exists(filename):
        print('data loaded from file')
        return pd.DataFrame.from_csv(filename)
    else:
        def match_func(x):
            match_duration = re.compile(r'.*\.duration')
            return match_duration.match(x)

        def rename_func(x):
            rename_field = re.compile(r'(.+)\.(.+)')
            return  rename_field.sub(r'\1', x)

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


#
# data = load_data()
# print(data)

# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

#

a_size = 6
t_size = 12
n_size = 500
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1234)

a = (np.random.rand(1, a_size) / 2 + 0.5) * 1e6
t = (np.random.rand(1, t_size) / 2 + 0.5) * 1e-3

# result
d = a.T * t
# print(d)


ti = np.random.choice(np.arange(a.size), n_size)
ai = a[:, ti]
di = ai.T * t
# print(di)

v = np.hstack((a, t))
x0 = np.array(a.size * [1e6] + t.size * [1e-3])


def fit_f(x0, *args):
    _a = np.array([args[:a.size]])
    _t = np.array([args[a.size:]])
    _di = _a[:, x0].T * _t
    return _di.flatten()


def get_error(reference, estimate):
    """Computes absolute and relative error given reference and estimate
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)
    """
    abs_error = estimate - reference
    rel_error = abs(np.divide(abs_error, reference))
    return abs_error, rel_error


bounds_l = a.size * [1e5] + t.size * [1e-4]
bounds_r = a.size * [1e7] + t.size * [1e-2]

popt, pcov = op.curve_fit(fit_f, xdata=ti, ydata=di.flatten(), p0=x0, bounds=[bounds_l, bounds_r], sigma=1e-6, absolute_sigma=1e-6)
print(pd.DataFrame(popt))
print(pd.DataFrame(v.flatten()))

ea = np.array([popt[:a.size]])
et = np.array([popt[a.size:]])
ed = ea.T * et
edi = ea[:, ti].T * et


abs_error, rel_error = get_error(d, ed)
D = abs(rel_error) + 1e-16
print(pd.DataFrame([D.flatten().min(), D.flatten().max()], index=['min', 'max']))

# plt.colorbar(plt.matshow(D, norm=LogNorm(vmin=D.flatten().min(), vmax=D.flatten().max())))
plt.colorbar(plt.matshow(D, norm=LogNorm(vmin=1e-16, vmax=1e-1)))
# plt.matshow(ed - d)
plt.show()


# alphas = np.floor(np.random.random((5, 1)) * 1000) - 500
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