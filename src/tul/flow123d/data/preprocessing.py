#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

import numpy as np
import pandas as pd


def remove_outliers(values, qa=25, qb=75, tol=1.5, return_indices=False):
    """
    :type values: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """
    values_np = np.array(values)
    Q1, Q3 = np.percentile(values_np, [qa, qb])
    IQR = Q3 - Q1

    a = Q1 - IQR * tol
    b = Q3 + IQR * tol

    indices = (values_np > a) & (values_np < b)
    return indices if return_indices else values_np[indices]

iqr_filter = remove_outliers


class ShiftDetection(object):

    def __init__(self, h=6, k=6, a=None):
        self.h = h
        self.k = k
        self.n = h + k
        self.a = a or self.a_wilcoxon

    def a_wilcoxon(self, i):
        return i

    def a_median(self, i):
        return 1 if i < self.n/2 else 0

    def find_shift(self, data, step=1, h=None, k=None, a=None):
        """
        :rtype: pandas.core.frame.DataFrame
        :param data:
        :param int h: negative look
        :param int k: positive look
        :param _collections_abc.Callable a: score function
        """
        h = h or self.h
        k = k or self.k
        a = a or self.a
        self.n = h + k

        stats = []
        for i in range(h, len(data)-k, step):
            y, t = data[i - h:i + k], i
            L = self.rank_stat(data, t, h, k, a)
            hmu, kmu = np.mean(data[i - h:i]), np.mean(data[i:i + k])
            stats.append((y, t, L, hmu-kmu, hmu, kmu))
        return pd.DataFrame(stats, columns=('y', 't', 'L', 'delta', 'hmu', 'kmu'))

    @classmethod
    def plot_find_result(cls, shifts):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.subplot(211)
        plt.title('Value of $L$ in time $t$')
        plt.plot(shifts['t'].values, shifts['L'].values)

        plt.subplot(212)
        plt.title('Mean difference in time $t$')
        plt.plot(shifts['t'].values, shifts['delta'].values)

    def rank_stat(self, data, t, h=None, k=None, a=None):
        """

        :param data:
        :param int t: point in time
        :param int h: negative look
        :param int k: positive look
        :param _collections_abc.Callable a: score function
        :return:
        """
        h = h or self.h
        k = k or self.k
        a = a or self.a
        self.n = h + k

        n = h + k                   # n - number of observations
        y = data[t-h:t+k]           # y - values of y

        r = np.argsort(y)           # indices of the sorted array, normalized by n
        abar = np.sum(a(r)) / n  # mean a

        s_p = np.sum(a(r[h:n])) / k  # normalize by k
        s_m = np.sum(a(r[0:h])) / h  # normalize by h

        hsa = h * (s_m - abar) ** 2
        ksa = k * (s_p - abar) ** 2
        aia = sum((a(r) - abar) ** 2)

        L = (n * (hsa + ksa)) / aia
        return L