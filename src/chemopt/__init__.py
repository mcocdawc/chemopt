from __future__ import absolute_import

import os

import chemopt.molpro


def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func
