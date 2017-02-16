#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs
import gridfs
from pymongo import MongoClient


class Mongo(object):
    """
    Class Mongo manages connection and queries
    :type db          : pymongo.database.Database
    :type bench       : pymongo.database.Collection
    :type nodes       : pymongo.database.Collection
    :type flat        : pymongo.database.Collection
    :type fs          : pymongo.database.Collection
    """

    _inited = False
    client = None
    db = None

    bench = None
    flat = None
    nodes = None
    fs = None

    @classmethod
    def init(cls, auto_auth=True):
        if cls._inited:
            return cls

        cls.client = MongoClient('hybs.nti.tul.cz')
        cls.db = cls.client.get_database('bench')
        cls.bench = cls.db.get_collection('bench')
        cls.nodes = cls.db.get_collection('nodes')
        cls.flat = cls.db.get_collection('flat_copy')
        cls.fs = cls.db.get_collection('fs')
        if auto_auth and cls.needs_auth():
            cls.auth()
        # cls.fs = gridfs.GridFS(cls.db)
        cls._inited = True
        return cls
    
    @classmethod
    def needs_auth(cls):
        try:
            cls.db.collection_names()
            return False
        except:
            return True
    
    @classmethod
    def auth(cls, username=None, password=None, config_file=None):
        if username is None:
            import os
            import yaml
            if config_file is None:
                root = __file__
                # recursively go down until src folder is found
                while True:
                    if os.path.basename(root) == 'src':
                        root = os.path.dirname(root)
                        break
                    root = os.path.dirname(root)
                config_file = os.path.join(root, '.config.yaml')
            # load config and extract username and password
            config = yaml.load(open(config_file, 'r').read())
            username = config.get('pymongo').get('username')
            password = config.get('pymongo').get('password')
        
        return cls.client.admin.authenticate(username, password)

mongo = Mongo