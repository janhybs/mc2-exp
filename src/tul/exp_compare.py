#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

import pandas as pd
import numpy as np
import math
import scipy.stats as sc
from sklearn import preprocessing as sk
import matplotlib.pyplot as plt

#
# def apply_to_panda(method, data, *args, **kwargs):
#     """
#     :type data: pandas.core.frame.DataFrame
#     :rtype: pandas.core.frame.DataFrame
#     """
#     if isinstance(data, pd.DataFrame):
#         cols = data.columns.values
#         rows = data.index.values
#         norm = method(data, *args, **kwargs)
#         return pd.DataFrame(norm, columns=cols, index=rows)
#     return method(data, *args, **kwargs)
#
#
# def norm(data, a=0, n='l1'):
#     """
#     :type data: pandas.core.frame.DataFrame
#     """
#     return apply_to_panda(sk.normalize, data, norm=n, axis=a)
#
# bench = pd.DataFrame(
#     [[1.0345478849120843e-06, 1.1657706118227633e-06, 1.1847867802180826e-06, 1.1417134958417643e-06,
#       2.029823339389075e-06, 1.0031145410087758e-06],
#      [1.9040130547694674e-06, 1.8579975801047394e-06, 1.8774337295605512e-06, 1.9480466680956306e-06,
#       1.008857309703421e-06, 1.8988039370690335e-06]],
#     columns=['exmag', 'gram', 'hildor', 'luna', 'mudrc', 'tarkil'],
#     index=['alpha', 'beta']
# )
# bench = bench.drop('gram', axis=1)
# flow = pd.DataFrame(
#     [[1.2821992315609868e-05, 1.53846719200773e-05, 1.168599865373086e-05, 2.9527680976424278e-05,
#       1.08164569913388e-05],
#      [1.5301521244537172e-05, 1.4706613928600705e-05, 1.557264776998122e-05, 1.1382846347384149e-05,
#       1.5782392408247202e-05]],
#     columns=['exmag', 'hildor', 'luna', 'mudrc', 'tarkil'],
#     index=['alpha', 'beta']
# )
#
# a = 1
# bench = norm(bench, a=a)
# flow = norm(flow, a=a)
# print(bench)
# print(flow)
#
# plt.subplot(221)
# plt.title(r'$\alpha$ correlation')
# plt.bar(np.arange(5)+0.0, bench.T['alpha'].values, width=0.3, color='#5DA5DA', label='bench')
# plt.bar(np.arange(5)+0.3, flow.T['alpha'].values, width=0.3, color='#FAA43A', label='flow')
# plt.xticks(np.arange(5)+0.3, bench.columns.values)
# plt.legend()
#
# plt.subplot(222)
# plt.title(r'$\beta$ correlation')
# plt.bar(np.arange(5)+0.0, bench.T['beta'].values, width=0.3, color='#5DA5DA', label='bench')
# plt.bar(np.arange(5)+0.3, flow.T['beta'].values, width=0.3, color='#FAA43A', label='flow')
# plt.xticks(np.arange(5)+0.3, bench.columns.values)
# plt.legend()
#
# plt.subplot(223)
# plt.title(r'$\alpha$ correlation')
# plt.plot(bench.T['alpha'].values, color='#5DA5DA', label='bench')
# plt.plot(flow.T['alpha'].values, color='#FAA43A', label='flow')
# plt.xticks(np.arange(5)+0.3, bench.columns.values)
# plt.legend()
#
# plt.subplot(224)
# plt.title(r'$\beta$ correlation')
# plt.plot(bench.T['beta'].values, color='#5DA5DA', label='bench')
# plt.plot(flow.T['beta'].values, color='#FAA43A', label='flow')
# plt.xticks(np.arange(5)+0.3, bench.columns.values)
# plt.legend()
# plt.show()
#
#
# pearson = sc.pearsonr(bench.T['alpha'].values, flow.T['alpha'].values)
# spearman = sc.spearmanr(bench.T['alpha'].values, flow.T['alpha'].values)
#
# stderr = 1.0 / math.sqrt(len(bench.T['alpha'].values) - 3)
# delta = 1.96 * stderr
#
# print('Correlation coeficients:')
# print('  - pearson  =', pearson[0])
# print('  - spearman =', spearman[0])#, math.tanh(math.atanh(pearson[0]) + delta))
# # plt.scatter(bench.T['alpha'].values, flow.T['alpha'].values)
# # plt.show()


def create(m, n, per_line, strip):
    matrix = np.zeros((m, n))
    half = int((strip - 1)/2)
    for i in range(m):
        middle = int(i/m * n)
        left = int(middle - half)
        right = int(middle + half)

        if left < 0:
            right += abs(left)
            left = 0

        if right >= n:
            left += n - right
            right = n

        index = np.random.randint(left, right, per_line)
        matrix[i, index] = 1
        matrix[i, middle] = .5
    return matrix

fig = plt.figure()
ax = fig.add_subplot(121)
ax.matshow(create(500, 300, 20, 15))
ax = fig.add_subplot(122)
ax.matshow(create(500, 300, 20, 50))
plt.show()