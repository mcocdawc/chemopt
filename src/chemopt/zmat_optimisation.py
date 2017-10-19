import inspect
import os
from datetime import datetime
from os.path import basename, join, normpath, splitext

import numpy as np
import scipy.optimize

from cclib.parser.utils import convertor
from chemcoord.xyz_functions import to_molden
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.interface.generic import calculate
from tabulate import tabulate


@substitute_docstr
def optimise(zmolecule, hamiltonian, basis, symbols=None,
             md_out=None, el_calc_input=None,
             molden_out=None, opt_f=None, etol=fixed_defaults['etol'],
             gtol=fixed_defaults['gtol'], **kwargs):
    """Optimize a molecule.

    Args:
        zmolecule (chemcoord.Zmat):
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        symbols (sympy expressions):
        el_calc_input (str): {el_calc_input}
        backend (str): {backend}
        charge (int): {charge}
        calculation_type (str): {calculation_type}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        etol (float): {etol}
        gtol (float): {gtol}

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
    if opt_f is None:
        opt_f = scipy.optimize.minimize
    base = splitext(basename(inspect.stack()[-1][1]))[0]
    if md_out is None:
        md_out = '{}.md'.format(base)
    if molden_out is None:
        molden_out = '{}.molden'.format(base)
    if el_calc_input is None:
        el_calc_input = join('{}_el_calcs'.format(base),
                             '{}.inp'.format(base))
    for filepath in [md_out, molden_out, el_calc_input]:
        rename_existing(filepath)

    t1 = datetime.now()
    if symbols is None:
        # # TODO continue here
        # while not is_converged(energies, grads_X):
        #     energies, grads_X = 1, 2

        V = _get_V_function(zmolecule, el_calc_input, md_out, **kwargs)
        with open(md_out, 'w') as f:
            f.write(_get_header(zmolecule, start_time=_get_isostr(t1),
                                **kwargs))

        try:
            opt_f(V, x0=_get_C_rad(zmolecule), jac=True, method='BFGS')
        except StopIteration:
            pass
        calculated = V(get_calculated=True)
    else:
        pass

    to_molden(
        [x['zmolecule'].get_cartesian() for x in calculated], buf=molden_out)
    t2 = datetime.now()
    with open(md_out, 'a') as f:
        footer = _get_footer(opt_zmat=calculated[-1]['zmolecule'],
                             start_time=t1, end_time=t2,
                             molden_out=molden_out)
        f.write(footer)
    return calculated


def _get_C_rad(zmolecule):
    C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
    C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
    return C_rad.flatten(order='F')


def _get_V_function(zmolecule, el_calc_input, md_out, **kwargs):
    get_zm_from_C = _get_zm_from_C_generator(zmolecule)

    def V(C_rad=None, calculated=[], get_calculated=False):
        if get_calculated:
            return calculated
        elif C_rad is not None:
            zmolecule = get_zm_from_C(C_rad)

            result = calculate(molecule=zmolecule, forces=True,
                               el_calc_input=el_calc_input, **kwargs)
            energy = convertor(result.scfenergies[0], 'eV', 'hartree')
            grad_energy_X = result.grads[0] / convertor(1, 'bohr', 'Angstrom')

            if is_converged(calculated, grad_energy_X):
                raise StopIteration


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
            with open(md_out, 'a') as f:
                f.write(_get_table_row(calculated, grad_energy_X))

            return energy, grad_energy_C.flatten()
        else:
            raise ValueError
    return V


def _get_zm_from_C_generator(zmolecule):
    def get_zm_from_C(C_rad=None, previous_zmats=[zmolecule],
                      get_previous=False):  # pylint:disable=dangerous-default-value
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


def _get_header(zmolecule, backend, hamiltonian, basis, charge, title,
                multiplicity, etol, gtol, start_time, **kwargs):
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
        get_row = '|{:>4.4}| {:^16.16} | {:^16.16} | {:^28.28} |'.format
        header = (get_row('n', 'energy [Hartree]',
                          'delta [Hartree]', 'grad_X_max [Hartree / Angstrom]')
                  + '\n'
                  + get_row(4 * '-', 16 * '-', 16 * '-', 28 * '-'))
        return header

    def _get_calc_setup(backend, hamiltonian, charge, multiplicity,
                        basis, etol, gtol):
        data = [['Hamiltonian', hamiltonian],
                ['Basis', basis],
                ['Charge', charge],
                ['Multiplicity', multiplicity],
                ['Convergencetest energy', etol],
                ['Convergencetest gradient', gtol]
                ]
        return tabulate(data, tablefmt='pipe', headers=['Backend', backend])
    calculation_setup = _get_calc_setup(backend, hamiltonian, charge,
                                        multiplicity, basis, etol, gtol)

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


def _get_table_row(calculated, grad_energy_X):
    n = len(calculated)
    energy = calculated[-1]['energy']
    if n == 1:
        delta = 0.
    else:
        delta = calculated[-1]['energy'] - calculated[-2]['energy']
    grad_energy_X_max = abs(grad_energy_X).max()
    get_str = '|{:>4}| {:16.10f} | {:16.10f} | {:28.10f} |\n'.format
    return get_str(n, energy, delta, grad_energy_X_max)


def rename_existing(filepath):
    if os.path.exists(filepath):
        to_be_moved = normpath(filepath).split(os.path.sep)[0]
        get_path = (to_be_moved + '_{}').format
        found = False
        end = 1
        while not found:
            if not os.path.exists(get_path(end)):
                found = True
            end += 1
        for i in range(end - 1, 1, -1):
            os.rename(get_path(i - 1), get_path(i))
        os.rename(to_be_moved, get_path(1))


def _get_footer(opt_zmat, start_time, end_time, molden_out):
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
                        molden=molden_out,
                        end_time=_get_isostr(end_time),
                        delta_time=str(end_time - start_time).split('.')[0])
    return output


def _get_isostr(time):
    return time.replace(microsecond=0).isoformat()


def is_converged(calculated, grad_energy_X, etol=1e-6, gtol=3e-4):
    """Returns if an optimization is converged.

    Args:
        energies (list): List of energies in hartree.
        grad_energy_X (numpy.ndarray): Gradient in cartesian coordinates
            in Hartree / Angstrom.
        etol (float): Tolerance for the energy.
        gtol (float): Tolerance for the maximum norm of the gradient.

    Returns:
        bool:
    """
    energies = [x['energy'] for x in calculated]
    if len(energies) == 0:
        return False
    elif len(energies) == 1:
        return False
    else:
        return (abs(energies[-1] - energies[-2]) < etol and
                abs(grad_energy_X).max() < gtol)
