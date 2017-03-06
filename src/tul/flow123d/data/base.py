#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pandas as pd
import math
from tul.flow123d.utils.stats import unpack
from tul.flow123d.utils.stats import norm
from tul.flow123d.utils.stats import plot_error
from tul.flow123d.utils.stats import get_error
from tul.flow123d.utils.stats import drop_col
from tul.flow123d.utils.stats import drop_row

def estimate_eq(a, b, c, m):
    # return a * c + b * m
    # return abs(a * math.exp(c)) + abs(b * math.exp(m))
    return a * math.exp(c) + b * math.exp(m)
    # return math.exp(a + c) + math.exp(b + m)


class Norm(object):
    NONE = 0
    MACH = 1
    TEST = 2
    BOTH = 8

    MACH_TEST = MACH | TEST
    TEST_MACH = TEST | MACH*2*2

    @classmethod
    def find(cls, value):
        return {
            cls.MACH: 'MACH',
            cls.TEST: 'TEST',
            cls.MACH_TEST: 'MACH_TEST',
            cls.TEST_MACH: 'TEST_MACH',
            cls.NONE: 'NONE',
            cls.BOTH: 'BOTH',
        }.get(value)


class D(object):

    xtol = 1e-7
    ftol = 1e-7
    test_drop = ['mem_l1', 'mem_l2', 'mem_l3', 'cpu_simple', 'mmn_s4', 'mem_ll', 'mvs_s1', 'mms_s1', 'mmn_s1']
    mach_drop = ['ajax']
    ProgressMonitor = namedtuple('ProgressMonitor', ['min', 'max', 'cur', 'k'])
    estimate_eq = estimate_eq
    penalty_mult = 1

    def __init__(self, data, normalize=Norm.MACH_TEST, cut_outliers=True):
        """
        :type data: pandas.core.frame.DataFrame
        """

        # normalize
        # if normalize:
        #     machines = data['machine']
        #     del data['machine']
        #     data, norm_0 = norm(data, a=0, return_norm=True)
        #     data, norm_1 = norm(data, a=1, return_norm=True)
        #     data['machine'] = machines

        self.norm = normalize
        data = drop_col(data, self.test_drop)
        for m in self.mach_drop:
            data = drop_row(data, machine=m)

        self.data = data
        machines = data['machine'].values
        tests = set(data.columns.values) - {'machine'}
        self.all = dict()

        for machine in machines:
            if machine not in self.all:
                self.all[machine] = dict()

            for test in tests:
                if test not in self.all[machine]:
                    self.all[machine][test] = self.data[data['machine'] == machine][test].values

        if cut_outliers:
            self.remove_outliers()

        self.y_bar = np.mean(np.hstack(self[:, :]))
        self.y_bar_i = {i: np.mean(np.hstack(self[i, :])) for i in self.mach_list}
        self.y_bar_j = {j: np.mean(np.hstack(self[:, j])) for j in self.test_list}
        self.y_norm_i_j = lambda i,j: (self.y_bar_i[i] * self.y_bar_j[j]) / self.y_bar

        self.norm_test, self.norm_mach = dict(), dict()
        if normalize:
            if normalize & Norm.BOTH:
                self.normalize_all()

            if normalize & Norm.MACH:
                self.normalize_mach()

            if normalize & Norm.TEST:
                self.normalize_test()

            if normalize & Norm.MACH*2*2:
                self.normalize_mach()

        self.progress_monitor = self.ProgressMonitor(np.inf, -np.inf, 0, 1)

    def as_df(self):
        return self.dict2df(self.all)

    @property
    def mach_list(self):
        return sorted(list(self.all.keys()))

    @property
    def test_list(self):
        return sorted(list(self.all[self.mach_list[0]].keys()))

    def normalize_test(self):
        self.norm_test = dict()
        for m in self.all:
            for t in self.all[m]:
                self.norm_test[t] = np.sum(np.hstack(self[:, t]))
            break

        for m in self.all:
            for t in self.all[m]:
                self[m, t] = self[m, t] / self.norm_test[t]

    def normalize_mach(self):
        self.norm_mach = dict()
        for m in self.all:
            self.norm_mach[m] = np.sum(np.hstack(self[m, :]))

        for m in self.all:
            for t in self.all[m]:
                self[m, t] = self[m, t] / self.norm_mach[m]

    def normalize_all(self):
        for i in self.all:
            for j in self.all[i]:
                y_norm_i_j = (self.y_bar_i[i] * self.y_bar_j[j]) / self.y_bar
                self[i, j] = self[i, j] / y_norm_i_j

    def __getitem__(self, k):
        if type(k) is tuple:
            if len(k) == 2:
                if type(k[0]) is not slice and type(k[1]) is not slice:
                    return self.all[k[0]][k[1]]

                # return k[1] for all machines
                elif type(k[0]) is slice and type(k[1]) is not slice:
                    result = []
                    for m in self.all:
                        if k[1] in self.all[m]:
                            result.append(self.all[m][k[1]])
                    return result

                # return k[0] for all tests
                elif type(k[1]) is slice and type(k[0]) is not slice:
                    result = []
                    if k[0] in self.all:
                        for t in self.all[k[0]]:
                            result.append(self.all[k[0]][t])
                    return result
                else:
                    result = []
                    for m in self.all:
                        for t in self.all[m]:
                            result.append(self.all[m][t])
                    return result

        return self.all[k]

    def __setitem__(self, k, v):
        if type(k) is tuple:
            self.all[k[0]][k[1]] = v
            return
        self.all[k] = v

    def remove_outliers(self, qa=25, qb=75, tol=1.5):
        for m in self.all:
            for t in self.all[m]:
                values = self[m, t]
                length = values.size
                Q1, Q3 = np.percentile(values, [qa, qb])
                IQR = Q3 - Q1

                a = Q1 - IQR * tol
                b = Q3 + IQR * tol
                indices = (values > a) & (values < b)
                values = values[indices]

                self[m, t] = values
                # print(length, length - values.size, m, t)

    def minimize_opts(self, min_func, x0, **opts):
        return dict(
            fun=min_func,
            x0=x0,
            method='Powell',
            # callback=self.show_minimize_progress,
            options=dict(
                xtol=self.xtol,
                ftol=self.ftol,
                maxfev=len(x0) * 2000,
                # bounds=[[0, 1e10]] * len(p0),
                disp=0
            )
        )

    def show_minimize_progress(self, x0, *args, **kwargs):
        d = dict(self.progress_monitor.__dict__)
        print('\r{k:5d}| {cur:18.10f} {min:18.10f} {max:18.10f}'.format(**d), end='')

    def make_square(self):
        maximum = 0
        for m in self.all:
            for t in self.all[m]:
                values = self[m, t]
                maximum = len(values) if len(values) > maximum else maximum

        for m in self.all:
            for t in self.all[m]:
                values = self[m, t]
                new_values = maximum - len(values)
                if new_values:
                    self[m, t] = np.hstack((values, [values.mean()] * new_values))

    def make_reference(self):
        result = self.all.copy()
        for m in self.all:
            for t in self.all[m]:
                result[m][t] = np.mean(self[m, t])
        return result

    @classmethod
    def dict2df(cls, d):
        a = sorted(list(d.keys()))
        t = sorted(list(d[a[0]].keys()))
        result = list()

        for mach in a:
            row = list()
            for test in t:
                row.append(d[mach][test])
            result.append(row)
        return pd.DataFrame(result, columns=t, index=a)

    @classmethod
    def construct_from_abcm(cls, mach_list, test_list, a, b, c, m):
        result = dict()
        for mach in mach_list:
            i = mach_list.index(mach)
            ai, bi = a[i], b[i]

            result[mach] = dict()
            for test in test_list:
                j = test_list.index(test)
                cj, mj = c[j], m[j]

                result[mach][test] = cls.estimate_eq(ai, bi, cj, mj)
        return result

    def estimate_abcm(self):
        reference = self.make_reference()
        mach_list = sorted(list(self.all.keys()))
        test_list = sorted(list(self.all[mach_list[0]].keys()))

        sizes = len(mach_list), len(mach_list), len(test_list), len(test_list)
        estimates = 1e-8, 1e-6, 1e1, 1e1
        x0 = [estimates[i] for i in range(len(sizes)) for s in range(sizes[i])]

        test_data = self.all.copy()

        self.progress_monitor = self.ProgressMonitor(np.inf, -np.inf, 0, 1)
        errors = []

        def minimize_func(x0):
            a, b, c, m = unpack(x0, sizes, as_matrix=False)
            error = 0

            for mach in test_data:
                i = mach_list.index(mach)
                ai, bi = a[i], b[i]

                for test in test_data[mach]:
                    j = test_list.index(test)
                    cj, mj = c[j], m[j]

                    try:
                        diff = (D.estimate_eq(ai, bi, cj, mj) - reference[mach][test])
                        rel_diff = diff #/ reference[mach][test]
                        rel_diff += penalty(ai, bi, cj, mj) * D.penalty_mult
                        diff_error = (rel_diff ** 2) / self.y_norm_i_j(mach, test)
                    except:
                        diff_error = np.inf

                    error += diff_error
                    errors.append(diff_error)

            self.progress_monitor = self.ProgressMonitor(
                min(self.progress_monitor.min, error),
                max(self.progress_monitor.max, error),
                error,
                self.progress_monitor.k+1
            )
            return error + penalty_all(a, b, c, m) * D.penalty_mult

        # r = op.minimize(
        #     minimize_func,
        #     x0=p0,
        #     # method='COBYLA',
        #     # options=dict(
        #     #     tol=1e-5,
        #     #     disp=True,
        #     #     maxiter=10000
        #     # )
        #     method='Powell',
        #     options=dict(
        #         xtol=self.xtol,
        #         ftol=self.ftol,
        #         maxfev=len(p0)*2000,
        #         # bounds=[[0, 1e10]] * len(p0),
        #         disp=3
        #     )
        #     # bounds=[[0, 1e10]] * len(p0),
        #     # method='L-BFGS-B',
        #     # options=dict(
        #     #     ftol=1e-18,
        #     #     gtol=1e-18,
        #     #     disp=1,
        #     #     eps=1
        #     # )
        # )

        # r = op.least_squares(
        #     minimize_func,
        #     x0=p0,
        #     ftol=3e-18,
        #     gtol=3e-18,
        #     xtol=3e-18,
        #     verbose=1,
        #     bounds=[1e-6, 1e7]
        # )

        # r = op.fmin_l_bfgs_b(
        #     minimize_func,
        #     pgtol=1e-16,
        #     disp=True,
        #     approx_grad=True,
        #     x0=p0
        # )

        r = op.minimize(
            **self.minimize_opts(minimize_func, x0)
        )

        a, b, c, m = unpack(r.x, sizes, False)
        ab = pd.DataFrame([a, b], index=('alpha', 'beta'), columns=mach_list)
        cm = pd.DataFrame([c, m], index=('cpu', 'mem'), columns=test_list)

        data_e = self.construct_from_abcm(mach_list, test_list, a, b, c, m)
        data_e_df = self.dict2df(data_e)
        reference_df = self.dict2df(reference)

        errors = get_error(data_e_df, reference_df)
        return Estimate(ab, cm, errors)

    def estimate_cm(self, ab):
        reference = self.make_reference()
        mach_list = sorted(list(self.all.keys()))
        test_list = sorted(list(self.all[mach_list[0]].keys()))

        sizes = len(test_list), len(test_list)
        estimates = 1e1, 1e1
        x0 = [estimates[i] for i in range(len(sizes)) for s in range(sizes[i])]

        a = ab[mach_list].values[0]
        b = ab[mach_list].values[1]

        test_data = self.all.copy()

        k, error, max_error, min_error = 0, None, -np.inf, np.inf
        errors = []

        def minimize_func(x0):
            nonlocal max_error, min_error, error, k
            c, m = unpack(x0, sizes, as_matrix=False)
            error = 0

            for mach in test_data:
                i = mach_list.index(mach)
                ai, bi = a[i], b[i]

                for test in test_data[mach]:
                    j = test_list.index(test)
                    cj, mj = c[j], m[j]


                    try:
                        diff = (D.estimate_eq(ai, bi, cj, mj) - reference[mach][test])
                        rel_diff = diff / reference[mach][test]
                        rel_diff += penalty(ai, bi, cj, mj) * D.penalty_mult
                        diff_error = rel_diff ** 2
                    except:
                        diff_error = np.inf

                    error += diff_error
                    errors.append(diff_error)

            self.progress_monitor = self.ProgressMonitor(
                min(self.progress_monitor.min, error),
                max(self.progress_monitor.max, error),
                error,
                self.progress_monitor.k+1
            )
            return error + penalty_all(a, b, c, m) * D.penalty_mult

        # r = op.minimize(
        #     minimize_func,
        #     x0=p0,
        #     method='Powell',
        #     options=dict(
        #         xtol=self.xtol,
        #         ftol=self.ftol,
        #         maxiter=len(p0) * 2000,
        #         disp=3
        #     )
        # )

        r = op.minimize(
            **self.minimize_opts(minimize_func, x0)
        )

        c, m = unpack(r.x, sizes, False)
        ab = pd.DataFrame([a, b], index=('alpha', 'beta'), columns=mach_list)
        cm = pd.DataFrame([c, m], index=('cpu', 'mem'), columns=test_list)

        data_e = self.construct_from_abcm(mach_list, test_list, a, b, c, m)
        data_e_df = self.dict2df(data_e)
        reference_df = self.dict2df(reference)

        errors = get_error(data_e_df, reference_df)
        return Estimate(ab, cm, errors)

    def cut_data(self, count=10):
        mach_list = list(self.all.keys())
        # rnd = np.random.random_integers(0, len(mach_list)-1, count)
        # print(rnd, len(mach_list))

        random_mach = np.random.choice(mach_list, count)
        result = dict()
        i = 0
        for m in set(random_mach):
            result[m] = dict()

            for t in self.all[m]:
                size = sum(random_mach == m)
                result[m][t] = np.random.choice(self[m, t], size)
                i+=size

        self.all = result
        return result


def penalty(a, b, c, m):
    """
    This function produces penalty for single a,b,c,m estimate
    :param a: \pi value for single machine
    :param b: \mu value for single machine
    :param c: p value for single test
    :param m: m value for single test

    :type a: float
    :type b: float
    :type c: float
    :type m: float
    :return:
    """
    p = 0
    if a < 1e-8:
        p += 1000
    if b < 1e-8:
        p += 1000
    if c < 8:
        p += 1000
    if m < 8:
        p += 1000
    return p


def penalty_all(a, b, c, m):
    p = 0

    diff = math.log10(abs(max(a) / min(a)))
    p += 0 if diff < 2 else diff*1

    diff = math.log10(abs(max(b) / min(b)))
    p += 0 if diff < 2 else diff*1

    diff = abs(max(c) - min(m))
    p += 0 if diff < 1.5 else diff*10

    diff = abs(max(m) - min(m))
    p += 0 if diff < 1.5 else diff*10
    return p


def iqr(values, qa=25, qb=75, tol=1.5):
    values_np = np.array(values)
    Q1, Q3 = np.percentile(values_np, [qa, qb])
    IQR = Q3 - Q1

    a = Q1 - IQR * tol
    b = Q3 + IQR * tol
    
    indices = (values_np > a) & (values_np < b)
    return values_np[indices]


def smooth(x, window_len=11, window='hamming'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    """
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else: 
        w = eval('np.' + window + '(window_len)')
    
    y = np.convolve(w / w.sum(), s,mode='valid')
    return y   


def outside_bounds(x, y, ucl, lcl):
    x_over = list()
    y_over = list()

    i = 0
    for i in range(len(y)):
        yi = y[i]
        xi = x[i]
        if (yi > ucl[i] or yi < lcl[i]):
            y_over.append(yi)
            x_over.append(xi)
    return x_over, y_over


class Estimate(object):
    def __init__(self, ab, cm, errors):
        self.abs_error, self.rel_error = errors
        self.ab = ab
        self.cm = cm
        self.mach_list = self.ab.columns.values
        self.test_list = self.cm.columns.values

        self.a, self.b = self.ab.T['alpha'].values, self.ab.T['beta'].values
        self.c, self.m = self.cm.T['cpu'].values, self.cm.T['mem'].values

        stats = 'min', 'mean', 'max', 'std'
        self.stats = pd.DataFrame(
            [getattr(np, f)(self.rel_error.values.flatten()) for f in stats],
            index=stats,
            columns=['error']
        ).T
        self.ref_ab = pd.DataFrame(np.mean(self.ab.T), columns=['ref']).T

    def plot_error(self, relative=True):
        if relative:
            plot_error(self.rel_error, self.mach_list, self.test_list)
            return
        plot_error(self.abs_error, self.mach_list, self.test_list)

    def __repr__(self):
        r = 'Estimate: \n'
        r += repr(self.ab) + '\n'
        r += repr(self.cm)
        return r
