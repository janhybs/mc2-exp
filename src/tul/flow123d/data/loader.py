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
                'base.test-name': '02_cube_123d'
            }
        },
        {
            '$project': {
                '_id': 0,
                'machine': '$base.nodename',
                'testname': '$indices.path_hash',
                'duration': '$cumul-time-sum'
            }
        }
    ])