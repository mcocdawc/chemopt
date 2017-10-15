import inspect
import os
from datetime import datetime
from os.path import basename, splitext

import numpy as np
from chemcoord.xyz_functions import to_molden
from scipy.optimize import minimize

from cclib.parser.utils import convertor
from chemopt.configuration import conf_defaults, fixed_defaults
from chemopt.interface.generic import calculate
from tabulate import tabulate


def optimise(zmolecule, symbols=None, **kwargs):
    """Optimize a molecule.

    Args:
        zmolecule (:class:`~chemcoord.Zmat`):
        symbols (sympy expressions):

    Returns:
        list: A list of dictionaries. Each dictionary has three keys:
        ``['energy', 'grad_energy', 'zmolecule']``.
        The energy is given in Hartree
        The energy gradient ('grad_energy') is given in internal coordinates.
        The units are Hartree / Angstrom for bonds and
        Hartree / radians for angles and dihedrals.
        The :class:`~chemcoord.Zmat` instance given by ``zmolecule``
        contains the keys ``['energy', 'grad_energy']`` in ``.metadata``.
    """
    base_filename = splitext(basename(inspect.stack()[-1][1]))[0]

    for f in ['{}.molden'.format, '{}.out'.format, '{}_el_calcs'.format]:
        rename_existing(f(base_filename))
    os.mkdir('{}_el_calcs'.format(base_filename))

    V = _get_V_function(zmolecule, base_filename, **kwargs)
    t1 = datetime.now()
    with open('{}.out'.format(base_filename), 'w') as f:
        f.write(_get_header(zmolecule, start_time=_get_isostring(t1), **kwargs))
    minimize(V, x0=_get_C_rad(zmolecule), jac=True, method='BFGS')
    calculated = V(get_calculated=True)

    to_molden([x['zmolecule'].get_cartesian() for x in calculated],
              buf='{}.molden'.format(base_filename))
    t2 = datetime.now()

    with open('{}.out'.format(base_filename), 'a') as f:
        footer = _get_footer(opt_zmat=calculated[-1]['zmolecule'],
                             start_time=t1, end_time=t2,
                             base_filename=base_filename)
        f.write(footer)
    return calculated


def _get_C_rad(zmolecule):
    C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
    C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
    return C_rad.flatten(order='F')


def _get_V_function(zmolecule, base_filename, **kwargs):
    get_zm_from_C = _get_zm_from_C_generator(zmolecule)

    def V(C_rad=None, calculated=[], get_calculated=False):
        if get_calculated:
            return calculated
        elif C_rad is not None:
            zmolecule = get_zm_from_C(C_rad)

            el_input = os.path.join('{}_el_calcs'.format(base_filename),
                                    '{}'.format(base_filename))
            result = calculate(molecule=zmolecule, forces=True,
                               base_filename=el_input, **kwargs)
            energy = convertor(result.scfenergies[0], 'eV', 'hartree')
            grad_energy_X = result.grads[0] / convertor(1, 'bohr', 'Angstrom')

            grad_X = zmolecule.get_grad_cartesian(
                as_function=False, drop_auto_dummies=True)
            grad_energy_C = np.sum(
                grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))

            for i in range(min(3, grad_energy_C.shape[0])):
                grad_energy_C[i, i:] = 0

            zmolecule.metadata['energy'] = energy
            zmolecule.metadata['grad_energy'] = grad_energy_C
            calculated.append({'energy': energy, 'grad_energy': grad_energy_C,
                               'zmolecule': zmolecule})
            with open('{}.out'.format(base_filename), 'a') as f:
                f.write(_get_table_row(calculated))

            return energy, grad_energy_C.flatten()
        else:
            raise ValueError
    return V


def _get_zm_from_C_generator(zmolecule):
    def get_zm_from_C(C_rad=None, previous_zmats=[zmolecule],
                      get_previous=False):
        if get_previous:
            return previous_zmats
        elif C_rad is not None:
            C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
            C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

            new_zm = previous_zmats.pop().copy()
            zmat_values = ['bond', 'angle', 'dihedral']
            new_zm.safe_loc[zmolecule.index, zmat_values] = C_deg
            previous_zmats.append(new_zm)
            return new_zm
        else:
            raise ValueError
    return get_zm_from_C


def _get_header(zmolecule, theory, basis,
                   start_time,
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
Starting {start_time}

{table_header}
""".format

    def _get_table_header():
        get_row = '|{:>4.4}| {:^16.16} | {:^16.16} |'.format
        header = (get_row('n', 'energy [hartree]', 'delta [hartree]')
                  + '\n'
                  + get_row(4 * '-', 16 * '-', 16 * '-'))
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
        start_time=start_time,
        table_header=_get_table_header())
    return header


def _get_markdown(molecule):
    data = molecule._frame
    return tabulate(data, tablefmt='pipe', headers=data.columns)


def _get_table_row(calculated):
    n = len(calculated)
    energy = calculated[-1]['energy']
    if n == 1:
        delta = 0.
    else:
        delta = calculated[-1]['energy'] - calculated[-2]['energy']
    return '|{:>4}| {:16.10f} | {:16.10f} |\n'.format(n, energy, delta)


def rename_existing(filepath):
    if os.path.exists(filepath):
        get_path = (filepath + '_{}').format
        found = False
        end = 1
        while not found:
            if not os.path.exists(get_path(end)):
                found = True
            end += 1
        for i in range(end - 1, 1, -1):
            os.rename(get_path(i - 1), get_path(i))
        os.rename(filepath, get_path(1))


def _get_footer(opt_zmat, start_time, end_time, base_filename):
    get_output = """\

## Optimised Structures
### Optimised structure as Zmatrix

{zmat}


### Optimised structure in cartesian coordinates

{cartesian}

## Closing

Structures were written to {molden}.

The calculation finished successfully at: {end_time}
and needed: {delta_time}.
""".format
    output = get_output(zmat=_get_markdown(opt_zmat),
                        cartesian=_get_markdown(opt_zmat.get_cartesian()),
                        molden='{}.molden'.format(base_filename),
                        end_time=_get_isostring(end_time),
                        delta_time=str(end_time - start_time).split('.')[0])
    return output


def _get_isostring(time):
    return time.replace(microsecond=0).isoformat()
