from __future__ import absolute_import

import pkg_resources  # part of setuptools
__version__ = pkg_resources.get_distribution("chemcoord").version

def export(func):
    if callable(func) and hasattr(func, '__name__'):
        globals()[func.__name__] = func
    try:
        __all__.append(func.__name__)
    except NameError:
        __all__ = [func.__name__]
    return func


import chemopt.interface
import chemopt.configuration
import chemopt.zmat_optimisation
import chemopt.utilities
