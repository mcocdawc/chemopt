from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from cclib.parser.utils import convertor
# from chemopt import export
from chemopt.configuration import conf_defaults, fixed_defaults
from chemopt.interface.generic import calculate
from tabulate import tabulate


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
    with open(output, 'w') as f:
        f.write(_create_header(zmolecule, **kwargs))
    minimize(V, x0=_extract_C_rad(zmolecule), jac=True, method='BFGS')
    calculated = V(_extract_C_rad(zmolecule), get_calculated=True)
    return calculated


def _extract_C_rad(zmolecule):
    C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
    C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
    return C_rad.flatten(order='F')


def _create_V_function(zmolecule, output, **kwargs):
    get_zm_from_C = _get_zm_from_C_generator(zmolecule)

    def V(C_rad, calculated=[], get_calculated=False):
        if get_calculated:
            return calculated, get_zm_from_C(get_calculated=True)
        else:
            zmolecule = get_zm_from_C(C_rad)

            result = calculate(molecule=zmolecule, forces=True, **kwargs)
            energy = convertor(result.scfenergies[0], 'eV', 'hartree')
            grad_energy_X = result.grads[0]

            grad_X = zmolecule.get_grad_cartesian(
                as_function=False, drop_auto_dummies=True)
            grad_energy_C = np.sum(
                grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))

            for i in range(min(3, grad_energy_C.shape[0])):
                grad_energy_C[i, i:] = 0
            calculated.append([energy, grad_energy_C])
            with open(output, 'a') as f:
                f.write(_get_table_row(calculated))

            return energy, grad_energy_C.flatten()
    return V


def _get_zm_from_C_generator(zmolecule):
    def get_zm_from_C(C_rad, previous_zmats=[zmolecule], get_previous=False):
        if get_previous:
            return previous_zmats
        else:
            C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
            C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

            new_zm = previous_zmats[-1].copy()
            new_zm.safe_loc[zmolecule.index, ['bond', 'angle', 'dihedral']] = C_deg
            previous_zmats.append(new_zm)
            return new_zm
    return get_zm_from_C


def _create_header(zmolecule, theory, basis,
                   backend=None,
                   charge=fixed_defaults['charge'],
                   title=fixed_defaults['title'],
                   multiplicity=fixed_defaults['multiplicity'], **kwargs):
    if backend is None:
        backend = conf_defaults['backend']
    get_header = """\
# This is ChemOpt {version} optimising a molecule in internal coordinates.

## Starting Structures
### Starting structure as Zmatrix
{zmat}

### Starting structure in cartesian coordinates
{cartesian}

## Setup for the electronic calculations
{electronic_calculation_setup}

## Iterations
Starting {time}:

{table_header}
""".format

    def _get_table_header():
        get_row = '|{:>4.4}| {:^16.16} | {:^16.16} |\n'.format
        header = get_row('n', 'energy [hartree]', 'delta [hartree]')
        header += get_row(4 * '-', 16 * '-', 16 * '-')
        return header

    def _get_calc_setup(backend, theory, charge, multiplicity):
        data = [['Theory', theory],
                ['Charge', charge],
                ['Multiplicity', multiplicity]]
        return tabulate(data, tablefmt='pipe', headers=['Backend', backend])
    calculation_setup = _get_calc_setup(backend, theory, charge, multiplicity)

    header = get_header(
        version='0.1.0', title=title, zmat=_get_markdown(zmolecule),
        cartesian=_get_markdown(zmolecule.get_cartesian()),
        electronic_calculation_setup=calculation_setup,
        time=datetime.now().replace(microsecond=0).isoformat(),
        table_header=_get_table_header())
    return header


def _get_markdown(molecule):
    data = molecule._frame
    return tabulate(data, tablefmt='pipe', headers=data.columns)


def _get_table_row(calculated):
    n = len(calculated)
    energy = calculated[-1][0]
    if n == 1:
        delta = 0.
    else:
        delta = energy - calculated[-2][0]
    return '|{:>4}| {:16.10f} | {:16.10f} |\n'.format(n, energy, delta)


def _create_footer(time, delta_time):
    get_output = """\
The calculation finished successfully {time}
and needed {delta_time}.

## Optimised Structures
### Optimised structure as Zmatrix
{zmat}

### Optimised structure in cartesian coordinates
{cartesian}
""".format
    output = get_output()
    return output
