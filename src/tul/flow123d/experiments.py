#!/usr/bin/python
# -*- coding: utf-8 -*-
# author:   Jan Hybs


class Experiment(object):

    def __init__(self, *args):
        self._monitors = []
        self.name = ''
        for arg in args:
            if len(arg) == 2:
                self._monitors.append([arg[0], arg[1], ''])
            else:
                self._monitors.append(arg)

    def describe(self, key_value_sep='-', prop_sep='--', width=1):
        result = []

        if width == 'auto':
            width = max([len(m[1]) for m in self._monitors])

        for context, prop, fmt in self._monitors:
            value = getattr(context, prop)
            if callable(fmt):
                value = fmt(value)
                result += [('{prop:'+str(width)+'s}{key_value_sep}{value}').format(**locals())]
            else:
                result += [('{prop:'+str(width)+'s}{key_value_sep}{value' + (':' + fmt if fmt else '') + '}').format(**locals())]

        return prop_sep.join(result)

    @property
    def describe_line(self):
        return self.describe()

    def __repr__(self):
        if self.name:
            return ('Experiment '+self.name+': \n    ') + self.describe(' = ', '\n    ', width='auto')
        else:
            return 'Experiment: \n    ' + self.describe(' = ', '\n    ', width='auto')

    def __hash__(self):
        return hash(self.describe_line)

    def __eq__(self, other):
        return self.describe_line == other.describe_line

    def __ne__(self, other):
        return not self.__eq__(other)


