#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs

from tul.flow123d.db.mongo import mongo as m
from operator import itemgetter
from itertools import groupby
m.init()


def find_bench(limit=1):
    if limit:
        return m.bench.find({})
    return m.bench.find({}).limit(limit)


def find_prof(limit=1):
    if limit:
        return m.flat.find({})
    return m.flat.find({}).limit(limit)


def flatten_dict(dd, separator='.', prefix=''):
    return {
        prefix + separator + k if prefix else k: v
        for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
        } if isinstance(dd, dict) else {prefix: dd}


def create_project(keys, match, rename, dollar='$'):
    """
    :type match:  callable
    :type rename: callable
    """
    result = dict()
    for k in keys:
        if match(k):
            result[rename(k)] = dollar + k

    return result


def aggregate_bench(match=None, project=None):
    query = []

    if match:
        query.append({'$match': match})

    if project:
        query.append({'$project': project})

    return m.bench.aggregate(query)


def aggregate_flat(match=None, project=None):
    query = []

    if match:
        query.append({'$match': match})

    if project:
        query.append({'$project': project})

    return m.flat.aggregate(query)


def group_by(items, field):
    items.sort(key=itemgetter(field))
    groups = groupby(items, key=itemgetter(field))
    # return [[item for item in data] for (key, data) in groups]
    return [{key: list(data)} for (key, data) in groups]


def load_flat():
    return m.flat.aggregate([
        {
            '$match': {
                'base.case-name': 'flow_fv',
                'base.returncode': 0,
                'base.run-process-count': 1,
                'base.test-name': '02_cube_123d',
                'base.task-size': 12954,
                # 'indices.path_hash': {'$in': [
                #     '83c60778ea2fead961ab5df28d080523',  # /Whole Program
                #     '0d56f027188fea10c75f4de68a9f5da6',  # /Whole Program/Application::run
                #     'dd75afbf41d0bd52afd1c0d0bc0e6d15',  # /Whole Program/Application::run/HC run simulation
                #     'a16bc554d64972ad479c22527b62630c',  # /Whole Program/Application::run/HC run simulation/TOS-output data
                #     # '697e722ce2ab45cd0ef6d7abad5ba964',  # /Whole Program/Application::run/HC run simulation/TOS-output data/OutputTime::write_time_frame
                #     # 'e975891dfe2cb09a238b6b2af83b5a2e',  # /Whole Program/Application::run/HC run simulation/TOS-output data/TOS-balance
                #     # 'f4652ac04e7b3a7ca60d0c4b1be22496',  # /Whole Program/Application::run/HC run simulation/TOS-output data/Convection balance zero time step
                # ]}
            }
        },
        {
            '$project': {
                '_id': 0,
                'machine': '$base.nodename',
                'testname': '$indices.path',
                'duration': '$cumul-time-sum'
            }
        },
    ])