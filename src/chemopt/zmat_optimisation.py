import chemcoord as cc
from chemopt.interface.generic import calculate


def optimise(zmat, symbols=None, max_iterations=100, **kwargs):
    results = calculate(molecule=zmat.get_cartesian(), **kwargs)
    return results.grads
