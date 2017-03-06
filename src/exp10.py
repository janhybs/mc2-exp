#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs


import pandas as pd
import scipy.optimize as op
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from tul.flow123d.data.preprocessing import ShiftDetection

y = np.hstack((
    np.random.rand(40)*2,
    np.random.rand(40)*2 + 5,
    np.random.rand(40)*2 + 10
))

shift_detect = ShiftDetection(h=15, k=5)
shifts = shift_detect.find_shift(y)

fig = plt.figure(figsize=(10, 5))
shift_detect.plot_find_result(shifts)
plt.tight_layout()
plt.show()