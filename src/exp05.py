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



cursor = loader.load_flat()
data_raw = pd.DataFrame(list(cursor))
t = np.array([list(set(data_raw['testname']))])
a = np.array([list(set(data_raw['machine']))])
print(t)
print(a)