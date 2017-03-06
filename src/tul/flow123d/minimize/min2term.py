#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

import numpy as np
import pandas as pd
import scipy.optimize as op
from sklearn import preprocessing as sk
from tul.flow123d.utils.stats import pack, unpack, calculate_eq, get_error


def estimate_all(d, di, a, t, ti):
    """
    :type data: pandas.core.frame.DataFrame
    :rtype: pandas.core.frame.DataFrame
    """

    sizes = a.size, a.size, t.size, t.size

    xtol = 3e-16
    ftol = 3e-16
    gtol = 3e-16

    alpha_estimate = 1e-5
    beta_estimate = 1e-4
    cpu_estimate = 1e6
    memory_estimate = 1e6

    x0 = pack(
        (a.size, alpha_estimate),
        (a.size, beta_estimate),
        (t.size, cpu_estimate),
        (t.size, memory_estimate),
    ).flatten()

    def minimize_function(x0, *args):
        _a, _b, _c, _m = unpack(args, *sizes)
        _di = calculate_eq(_a[:, x0], _b[:, x0], _c, _m)
        return _di.flatten()

    popt, pcov = op.curve_fit(
        minimize_function,
        xdata=ti,
        ydata=di.flatten(),
        p0=x0,
        bounds=(1e-6, 1e9),
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=2
    )

    ea, eb, ec, em = unpack(popt, *sizes)
    ed = calculate_eq(ea, eb, ec, em)

    coef_ab = pd.DataFrame([ea.flatten(), eb.flatten()], index=['alpha', 'beta'], columns=a.flatten())
    coef_ab = coef_ab.sort_index(axis=1)

    coef_cm = pd.DataFrame([ec.flatten(), em.flatten()], index=['cpu', 'mem'], columns=t.flatten())
    coef_cm = coef_cm.sort_index(axis=1)

    abs_error, rel_error = get_error(d, ed)
    E = abs(rel_error) + 1e-16
    print(pd.DataFrame([E.flatten().min(), E.flatten().max(), E.flatten().mean()], index=['min', 'max', 'mean']))

    return ea, eb, ec, em, ed, E, coef_ab, coef_cm


def estimate_cm(d, di, a, t, ti, alpha, beta):

    sizes = t.size, t.size

    xtol = 3e-16
    ftol = 3e-16
    gtol = 3e-16

    cpu_estimate = 1e6
    memory_estimate = 1e6

    x0 = pack(
        (t.size, cpu_estimate),
        (t.size, memory_estimate),
    ).flatten()

    def minimize_function(x0, *args):
        _c, _m = unpack(args, *sizes)
        _di = calculate_eq(alpha[:, x0], beta[:, x0], _c, _m)
        return _di.flatten()

    popt, pcov = op.curve_fit(
        minimize_function,
        xdata=ti,
        ydata=di.flatten(),
        p0=x0,
        bounds=(1e-6, 1e9),
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=1
    )

    ec, em = unpack(popt, *sizes)
    ea, eb = alpha, beta
    ed = calculate_eq(ea, eb, ec, em)

    coef_ab = pd.DataFrame([ea.flatten(), eb.flatten()], index=['alpha', 'beta'], columns=a.flatten())
    coef_ab = coef_ab.sort_index(axis=1)

    coef_cm = pd.DataFrame([ec.flatten(), em.flatten()], index=['cpu', 'mem'], columns=t.flatten())
    coef_cm = coef_cm.sort_index(axis=1)

    abs_error, rel_error = get_error(d, ed)
    E = abs(rel_error) + 1e-16
    print(pd.DataFrame([E.flatten().min(), E.flatten().max(), E.flatten().mean()], index=['min', 'max', 'mean']))

    return ea, eb, ec, em, ed, E, coef_ab, coef_cm