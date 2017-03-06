#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

import sys
import math

main = sys.modules['__main__']
import warnings
warnings.filterwarnings('ignore')


from tul.flow123d.experiments import Experiment as Exp
from tul.flow123d.data.base import D as Data
from tul.flow123d.data.base import Norm
from tul.flow123d.utils.stats import norm, drop_row, drop_col, drop_outliers, load_data

from IPython.display import display, HTML
from pluck import pluck

import pandas as pd
import scipy.optimize as op
import scipy.stats as st
import numpy as np


np.set_printoptions(precision=4, suppress=False)
np.random.seed(1234)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


Data.ftol       = 1e-3
Data.xtol       = Data.ftol
Data.test_drop = [
    'mem_l1', 'mem_l2', 'mem_l3',
    'cpu_simple', 'mmn_s4', 'mem_ll',
    'mvs_s1', 'mms_s1', 'mmn_s1',
    #'cpu_hash', 'cpu_md5'
]
Data.mach_drop = [
    'ajax',# 'luna'
]


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


versions   =  '1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5'
version    = versions[0]
cut_count  = 15
normalize  = Norm.MACH_TEST
repetition = 25

exp = Exp(
    (main, 'cut_count'),
    (main, 'version'),
    (main,  'repetition'),
    (main, 'normalize', Norm.find),
    (Data, 'xtol'),
)

exp.data     = Data(load_data(version=version), normalize=normalize)
exp.baseline = exp.data.estimate_abcm()


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


display(exp)
exp.baseline.plot_error()
display(exp.baseline.stats)
display(exp.baseline.ref_ab)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# estimate p and m value when \pi and \mu are known

import progressbar

bar = progressbar.ProgressBar(
    fd=sys.stdout,
    term_width=80,
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(marker='='),
        ' (', progressbar.ETA(), ') ',
    ]
)

exp.estimates = dict()


def estimate_pm(version, repetition, test_data_size):
    exp.estimates[version] = list()
    bar.max_value = repetition
    with bar:
        for i in range(repetition):
            bar.update(i)

            d = Data(load_data(version=version), normalize=normalize)

            if test_data_size:
                d.cut_data(count=test_data_size)
            e = d.estimate_cm(exp.baseline.ab)

            exp.estimates[version].append(e)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


estimate_pm('1.1.2', repetition, 10)
cms = pluck(exp.estimates['1.1.2'], 'cm.values')
cms = np.vstack(cms)
display(cms.shape)
#display(exp.baseline.cm)
