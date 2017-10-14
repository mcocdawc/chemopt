import time
import numpy as np
from scipy.optimize import minimize

from cclib.parser.utils import convertor
from chemopt import export
from chemopt.interface.generic import calculate


def convert(x):
    return convertor(x, 'hartree', 'eV') / convertor(1, 'bohr', 'Angstrom')


def optimise(zmolecule, output, symbols=None, **kwargs):
    """Optimize a molecule.

    Args:
        frame (pd.DataFrame): A Dataframe with at least the
            columns ``['atom', 'x', 'y', 'z']``.
            Where ``'atom'`` is a string for the elementsymbol.
        atoms (sequence): A list of strings. (Elementsymbols)
        coords (sequence): A ``n_atoms * 3`` array containg the positions
            of the atoms. Note that atoms and coords are mutually exclusive
            to frame. Besides atoms and coords have to be both either None
            or not None.

    Returns:
        :class:`chemcoord.Cartesian`: A new cartesian instance.
    """
    V = _create_V_function(zmolecule, output, **kwargs)
    create_output(output, zmolecule, **kwargs)
    opt = minimize(V, x0=_extract_C_rad(zmolecule), jac=True, method='BFGS')
    return opt


def _extract_C_rad(zmolecule):
    C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
    C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
    return C_rad.flatten(order='F')


def _create_V_function(zmolecule, outputfile, **kwargs):
    get_zm_from_C = _get_zm_from_C_generator(zmolecule)

    def V(C_rad, calculated=[], get_calculated=False):
        zmolecule = get_zm_from_C(C_rad)

        result = calculate(molecule=zmolecule, forces=True, **kwargs)
        energy = result.scfenergies[0]
        grad_energy_X = convert(result.grads[0])

        grad_X = zmolecule.get_grad_cartesian(
            as_function=False, drop_auto_dummies=True)
        grad_energy_C = np.sum(
            grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))

        for i in range(min(3, grad_energy_C.shape[0])):
            grad_energy_C[i, i:] = 0

        grad_energy_C = grad_energy_C.flatten()
        calculated.append((zmolecule, energy, grad_energy_C))
        if get_calculated:
            return calculated
        else:
            return energy, grad_energy_C
    return V


def _get_zm_from_C_generator(zmolecule):
    def get_zm_from_C(C_rad, previous_zm=[zmolecule]):
        C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
        C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

        new_zm = previous_zm[-1].copy()
        new_zm.safe_loc[zmolecule.index, ['bond', 'angle', 'dihedral']] = C_deg
        previous_zm.append(new_zm)
        return new_zm
    return get_zm_from_C


def create_ouput(output, zmolecule, **kwargs):
    header = """\
This is ChemOpt {version} ptimising a molecule in internal coordinates.

Structure as Zmatrix:
{zmat}
Structure in cartesian coordinates:
{cartesian}

The setup for the electronic calculation is:
Backend: {backend}
Theory: {theory}
Basis: {basis}
Charge: {charge}
Multiplicity: {multiplicity}

Starting at {time}:
n energy Δ
""".format(version=chemopt.__version__, zmat=zmolecule.to_zmat(),
           cartesian=zmolecule.get_cartesian().to_xyz(),
           time=time.asctime(), **kwargs)

    with open(output, 'x') as f:
        f.write(header)
