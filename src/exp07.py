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
import progressbar

np.set_printoptions(precision=4)
np.set_printoptions(precision=4, suppress=False)
np.random.seed(1234)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from tul.flow123d.utils.stats import norm, drop_row, drop_col, drop_outliers, load_data, reveal_data, plot_error, get_error, drop_outliers2
from tul.flow123d.minimize.min2term import estimate_all, estimate_cm
from tul.flow123d.data import preprocessing as pp
from tul.flow123d.data.base import D


# calculate baseline
D.ftol     = 1e-5
D.xtol     = D.ftol
cut_count  = 15
repetition = 100

d = D(load_data(version='1.1.1'))
baseline = d.estimate_abcm()
baseline.plot_error()




# final_cpu = []
# final_mem = []
# tests = 50
# for i in range(tests):
#     print(i+1, 'of', tests)
#
#     d = D(load_data(version='1.1.1'))
#     d.cut_data(count=5)
#     e = d.estimate_cm(baseline.ab)
#
#     final_cpu.append(e.cm.values[0])
#     final_mem.append(e.cm.values[1])
#     print(e.cm[])

#
# # calculate estimates for random subset
# final_cpu = []
# final_mem = []
# tests = 50
# for i in range(tests):
#     print(i+1, 'of', tests)
#
#     d = D(load_data(version='1.1.1'))
#     d.cut_data(count=5)
#     e = d.estimate_cm(baseline.ab)
#
#     final_cpu.append(e.cm.values[0])
#     final_mem.append(e.cm.values[1])
#
#
# final_cpu = np.array(final_cpu)
# final_mem = np.array(final_mem)
#
# # plot cpu values
# plt.figure(figsize=(20, 15))
# for i in range(final_cpu.shape[1]):
#     test = baseline.test_list[i]
#     data = final_cpu[:, i]
#     std = np.std(data)
#
#     plt.subplot(3, 4, i+1)
#     plt.title('{:} ($\sigma$ = {:1.4f})'.format(test, std))
#     plt.hist(data)
#     print(test, std)
#
# # plt.savefig('spread-cpu.pdf')
# plt.show()
#
#
# # plot mem values
# plt.figure(figsize=(20, 15))
# for i in range(final_cpu.shape[1]):
#     test = baseline.test_list[i]
#     data = final_mem[:, i]
#     std = np.std(data)
#
#     plt.subplot(3, 4, i+1)
#     plt.title('{:} ($\sigma$ = {:1.4f})'.format(test, std))
#     plt.hist(data)
#     print(test, std)
#
# # plt.savefig('spread-mem.pdf')
# plt.show()
#
#
# # d = D(load_data(version='1.1.4'))
# # estimate2 = d.estimate_cm(estimate.a, estimate.b)
# #
# # print(estimate2)
# # estimate2.plot_error()
# #
# #
# # c1 = norm(estimate.c.reshape(-1, 1)).flatten()
# # c2 = norm(estimate2.c.reshape(-1, 1)).flatten()
# # plt.subplot(211)
# # plt.plot(np.arange(c1.size), c1)
# # plt.plot(np.arange(c2.size), c2)
# # plt.xticks(np.arange(estimate.c.size), estimate.test_list)
# #
# # m1 = norm(estimate.m.reshape(-1, 1)).flatten()
# # m2 = norm(estimate2.m.reshape(-1, 1)).flatten()
# # plt.subplot(212)
# # plt.plot(np.arange(m1.size), m1)
# # plt.plot(np.arange(m2.size), m2)
# # plt.xticks(np.arange(estimate.m.size), estimate.test_list)
# # plt.show()