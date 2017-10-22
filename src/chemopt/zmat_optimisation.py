import inspect
import os
from functools import partial
from datetime import datetime
from os.path import basename, normpath, splitext

import numpy as np
from scipy.optimize import minimize

from chemcoord.xyz_functions import to_molden
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.interface.generic import calculate
from chemopt.exception import ConvergenceFinished
from tabulate import tabulate


@substitute_docstr
def optimise(zmolecule, hamiltonian, basis,
             symbols=None,
             md_out=None, el_calc_input=None, molden_out=None,
             etol=fixed_defaults['etol'],
             gtol=fixed_defaults['gtol'],
             max_iter=fixed_defaults['max_iter'],
             backend=conf_defaults['backend'],
             charge=fixed_defaults['charge'],
             title=fixed_defaults['title'],
             multiplicity=fixed_defaults['multiplicity'],
             num_procs=None, mem_per_proc=None, **kwargs):
    """Optimize a molecule.

    Args:
        zmolecule (chemcoord.Zmat):
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        symbols (sympy expressions):
        el_calc_input (str): {el_calc_input}
        md_out (str): {md_out}
        molden_out (str): {molden_out}
        backend (str): {backend}
        charge (int): {charge}
        title (str): {title}
        multiplicity (int): {multiplicity}
        etol (float): {etol}
        gtol (float): {gtol}
        max_iter (int): {max_iter}
        num_procs (int): {num_procs}
        mem_per_proc (str): {mem_per_proc}

    Returns:
        list: A list of dictionaries. Each dictionary has three keys:
        ``['energy', 'grad_energy', 'structure']``.
        The energy is given in Hartree
        The energy gradient ('grad_energy') is given in internal coordinates.
        The units are Hartree / Angstrom for bonds and
        Hartree / radians for angles and dihedrals.
        The :class:`~chemcoord.Zmat` instance given by ``zmolecule``
        contains the keys ``['energy', 'grad_energy']`` in ``.metadata``.
    """
    files = _get_defaults(md_out, molden_out, el_calc_input)
    for filepath in files:
        rename_existing(filepath)
    md_out, molden_out, el_calc_input = files

    if num_procs is None:
        num_procs = conf_defaults['num_procs']
    if mem_per_proc is None:
        mem_per_proc = conf_defaults['mem_per_proc']

    t1 = datetime.now()
    if symbols is None:
        header = _get_generic_optimise_header(
            zmolecule=zmolecule, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, multiplicity=multiplicity,
            num_procs=num_procs, mem_per_proc=mem_per_proc,
            start_time=t1, title=title,
            etol=etol, gtol=gtol, max_iter=max_iter)
        with open(md_out, 'w') as f:
            f.write(header)
        calculated, convergence = _zmat_generic_optimise(
            zmolecule=zmolecule, el_calc_input=el_calc_input,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)
    else:
        pass

    to_molden(
        [x['structure'].get_cartesian() for x in calculated], buf=molden_out)
    t2 = datetime.now()
    with open(md_out, 'a') as f:
        footer = _get_footer(
            opt_zmat=calculated[-1]['structure'], start_time=t1, end_time=t2,
            molden_out=molden_out, successful=convergence.successful)
        f.write(footer)
    return calculated


def _zmat_generic_optimise(
        zmolecule, el_calc_input, md_out, backend, hamiltonian, basis, charge,
        title, multiplicity, etol, gtol, max_iter,
        num_procs, mem_per_proc, **kwargs):
    def _get_C_rad(zmolecule):
        C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
        C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
        return C_rad.flatten(order='F')
    V = _get_generic_opt_V(
        zmolecule=zmolecule, el_calc_input=el_calc_input,
        md_out=md_out, backend=backend, hamiltonian=hamiltonian,
        basis=basis, charge=charge, title=title, multiplicity=multiplicity,
        etol=etol, gtol=gtol, max_iter=max_iter,
        num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)
    try:
        minimize(V, x0=_get_C_rad(zmolecule), jac=True, method='BFGS')
    except ConvergenceFinished as e:
        convergence = e
    calculated = V(get_calculated=True)
    return calculated, convergence


def _get_generic_opt_V(
        zmolecule, el_calc_input, md_out, backend,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, **kwargs):
    get_new_zmat = partial(get_zm_from_C, index_to_change=zmolecule.index)

    def V(C_rad=None, get_calculated=False,
          calculated=[]):  # pylint:disable=dangerous-default-value
        if get_calculated:
            return calculated
        elif C_rad is not None:
            try:
                previous_zmat = calculated[-1]['structure'].copy()
            except IndexError:
                new_zmat = zmolecule.copy()
            else:
                new_zmat = get_new_zmat(C_rad, previous_zmat)

            result = calculate(
                molecule=new_zmat, forces=True, el_calc_input=el_calc_input,
                backend=backend, hamiltonian=hamiltonian, basis=basis,
                charge=charge, title=title,
                multiplicity=multiplicity,
                num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

            energy, grad_energy_X = result['energy'], result['gradient']

            grad_X = new_zmat.get_grad_cartesian(as_function=False,
                                                 drop_auto_dummies=True)
            grad_energy_C = np.sum(grad_energy_X.T[:, :, None, None]
                                   * grad_X, axis=(0, 1))

            for i in range(min(3, grad_energy_C.shape[0])):
                grad_energy_C[i, i:] = 0.
            new_zmat.metadata['energy'] = energy
            new_zmat.metadata['grad_energy'] = grad_energy_C
            calculated.append({'energy': energy, 'grad_energy': grad_energy_C,
                               'structure': new_zmat})
            with open(md_out, 'a') as f:
                f.write(_get_table_row(calculated, grad_energy_X))

            if is_converged(calculated, grad_energy_X, etol=etol, gtol=gtol):
                raise ConvergenceFinished(successful=True)
            elif len(calculated) >= max_iter:
                raise ConvergenceFinished(successful=False)

            return energy, grad_energy_C.flatten()
        else:
            raise ValueError
    return V


def get_zm_from_C(C_rad, previous_zmat, index_to_change):
    C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
    C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

    new_zm = previous_zmat.copy()
    new_zm.safe_loc[index_to_change, ['bond', 'angle', 'dihedral']] = C_deg
    return new_zm


def _get_generic_optimise_header(
        zmolecule, backend, hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, start_time, num_procs, mem_per_proc):
    if backend is None:
        backend = conf_defaults['backend']
    get_header = """\
# This is ChemOpt {version} optimising a molecule in internal coordinates.

## Settings for the calculations

{calculation_setup}

## Starting structures
### Starting structure as Zmatrix

{zmat}

### Starting structure in cartesian coordinates

{cartesian}

## Iterations
Starting {start_time}

{table_header}
""".format
    calculation_setup = _get_calc_setup(
        backend=backend, hamiltonian=hamiltonian, charge=charge,
        multiplicity=multiplicity, basis=basis,
        etol=etol, gtol=gtol, max_iter=max_iter,
        num_procs=num_procs, mem_per_proc=mem_per_proc,)

    header = get_header(
        version='0.1.0', title=title, zmat=_get_markdown(zmolecule),
        cartesian=_get_markdown(zmolecule.get_cartesian()),
        calculation_setup=calculation_setup,
        start_time=start_time,
        table_header=_get_table_header())
    return header


# def _get_symb_header(zmolecule, backend, hamiltonian, basis, charge, title,
#                      multiplicity, etol, gtol, start_time,
#                      num_procs, mem_per_proc):
#     if backend is None:
#         backend = conf_defaults['backend']
#     get_header = """\
# # This is ChemOpt {version} optimising a molecule in internal coordinates.
#
# ## Starting structures
# ### Starting structure as Zmatrix
#
# {zmat}
#
# ### Starting structure in cartesian coordinates
#
# {cartesian}
#
# ## Setup for the electronic calculations
# {electronic_calculation_setup}
#
# ## Iterations
# Starting {start_time}
#
# {table_header}
# """.format
#     calculation_setup = _get_calc_setup(
#         backend, hamiltonian, charge, multiplicity, basis,
#         etol, gtol, num_procs, mem_per_proc)
#
#     header = get_header(
#         version='0.1.0', title=title, zmat=_get_markdown(zmolecule),
#         cartesian=_get_markdown(zmolecule.get_cartesian()),
#         electronic_calculation_setup=calculation_setup,
#         start_time=start_time,
#         table_header=_get_table_header())
#     return header


def _get_calc_setup(backend, hamiltonian, charge, multiplicity,
                    basis, etol, gtol, num_procs, mem_per_proc, max_iter):
    data = [['Hamiltonian', hamiltonian],
            ['Basis', basis],
            ['Charge', charge],
            ['Spin multiplicity', multiplicity],
            ['Convergence energy', etol],
            ['Convergence gradient', gtol],
            ['Maximum of iterations', max_iter],
            ['Number of processes', num_procs],
            ['Memory per process', mem_per_proc]
            ]
    return tabulate(data, tablefmt='pipe', headers=['Backend', backend])


def _get_markdown(molecule):
    data = molecule._frame
    return tabulate(data, tablefmt='pipe', headers=data.columns)


def _get_table_header():
    get_row = '|{:>4.4}| {:^16.16} | {:^16.16} | {:^30.30} |'.format
    header = (get_row('n', r'$E [E_h]$', r'$\Delta E [E_h]$',
                      r'$\max(|\nabla_X E |) [E_h$/Å]')
              + '\n'
              + get_row(4 * '-', 16 * '-', 16 * '-', 28 * '-'))
    return header


def _get_table_row(calculated, grad_energy_X):
    n = len(calculated)
    energy = calculated[-1]['energy']
    if n == 1:
        delta = 0.
    else:
        delta = calculated[-1]['energy'] - calculated[-2]['energy']
    grad_energy_X_max = abs(grad_energy_X).max()
    get_str = '|{:>4}| {:+16.10f} | {:+16.10f} | {:+30.10f} |\n'.format
    return get_str(n, energy, delta, grad_energy_X_max)


def rename_existing(filepath):
    if os.path.exists(filepath):
        to_be_moved = normpath(filepath).split(os.path.sep)[0]
        get_path = (to_be_moved + '_{}').format
        found, end = False, 1
        while not found:
            if not os.path.exists(get_path(end)):
                found = True
                os.rename(to_be_moved, get_path(end))
            else:
                end += 1


def _get_footer(opt_zmat, start_time, end_time, molden_out, successful):
    get_output = """\

## Optimised Structures
### Optimised structure as Zmatrix

{zmat}


### Optimised structure in cartesian coordinates

{cartesian}

## Closing

Structures were written to {molden}.

The calculation finished {successfully} at: {end_time}
and needed: {delta_time}.
""".format
    output = get_output(
        zmat=_get_markdown(opt_zmat),
        cartesian=_get_markdown(opt_zmat.get_cartesian()),
        molden=molden_out, end_time=_get_isostr(end_time),
        successfully='successfully' if successful else 'with errors',
        delta_time=str(end_time - start_time).split('.')[0])
    return output


def _get_isostr(time):
    return time.replace(microsecond=0).isoformat()


@substitute_docstr
def is_converged(calculated, grad_energy_X, etol=fixed_defaults['etol'],
                 gtol=fixed_defaults['gtol']):
    """Returns if an optimization is converged.

    Args:
        energies (list): List of energies in hartree.
        grad_energy_X (numpy.ndarray): Gradient in cartesian coordinates
            in Hartree / Angstrom.
        etol (float): {etol}
        gtol (float): {gtol}

    Returns:
        bool:
    """
    if len(calculated) == 0:
        return False
    elif len(calculated) == 1:
        return False
    else:
        delta_energy = calculated[-1]['energy'] - calculated[-2]['energy']
        return abs(delta_energy) < etol and abs(grad_energy_X).max() < gtol


def _get_defaults(md_out: str, molden_out: str, el_calc_input: str) -> str:
    if __name__ == '__main__':
        if md_out is None:
            raise ValueError('md_out has to be provided when executing '
                             'from an interactive session.')
        if molden_out is None:
            raise ValueError('molden_out has to be provided when executing '
                             'from an interactive session.')
        if el_calc_input is None:
            raise ValueError('el_calc_input has to be provided when executing '
                             'from an interactive session.')
    else:
        base = splitext(basename(inspect.stack()[-1][1]))[0]
        if md_out is None:
            md_out = '{}.md'.format(base)
        if molden_out is None:
            molden_out = '{}.molden'.format(base)
        if el_calc_input is None:
            el_calc_input = os.path.join('{}_el_calcs'.format(base),
                                         '{}.inp'.format(base))
        return md_out, molden_out, el_calc_input
