import inspect
import os
from datetime import datetime
from os.path import basename, normpath, splitext

import numpy as np
import sympy
from chemcoord.xyz_functions import to_molden
from scipy.optimize import minimize
from sympy import latex

from chemopt import __version__, export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.exception import ConvergenceFinished
from chemopt.interface.generic import calculate
from tabulate import tabulate


@export
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
    files = _get_default_filepaths(md_out, molden_out, el_calc_input)
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
        header = _get_symb_optimise_header(
            zmolecule=zmolecule, symbols=symbols, backend=backend,
            hamiltonian=hamiltonian, basis=basis, charge=charge,
            multiplicity=multiplicity, num_procs=num_procs,
            mem_per_proc=mem_per_proc, start_time=t1, title=title,
            etol=etol, gtol=gtol, max_iter=max_iter)
        with open(md_out, 'w') as f:
            f.write(header)
        V = _zmat_symb_optimise(
            zmolecule=zmolecule, symbols=symbols, el_calc_input=el_calc_input,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)
        return V

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
        minimize(V, x0=_get_C_rad(zmolecule), jac=True, method='BFGS',
                 options={'gtol': 1e-10})
    except ConvergenceFinished as e:
        convergence = e
    calculated = V(get_calculated=True)
    return calculated, convergence


def _zmat_symb_optimise(
        zmolecule, symbols, el_calc_input, md_out, backend,
        hamiltonian, basis, charge,
        title, multiplicity, etol, gtol, max_iter,
        num_procs, mem_per_proc, **kwargs):
    V = _get_symbolic_opt_V(
        zmolecule=zmolecule, symbols=symbols, el_calc_input=el_calc_input,
        md_out=md_out, backend=backend, hamiltonian=hamiltonian,
        basis=basis, charge=charge, title=title, multiplicity=multiplicity,
        etol=etol, gtol=gtol, max_iter=max_iter,
        num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)
    return V
    try:
        minimize(V, x0=np.array([v for s, v in symbols]),
                 jac=True, method='BFGS')
    except ConvergenceFinished as e:
        convergence = e
    calculated = V(get_calculated=True)
    return calculated, convergence


def _get_generic_opt_V(
        zmolecule, el_calc_input, md_out, backend,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, **kwargs):
    def get_new_zmat(C_rad, previous_zmat):
        C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
        C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

        new_zm = previous_zmat.copy()
        new_zm.safe_loc[zmolecule.index, ['bond', 'angle', 'dihedral']] = C_deg
        return new_zm

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
            grad_energy_C = _get_grad_energy_C(new_zmat, grad_energy_X)

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


def _get_symbolic_opt_V(
        zmolecule, symbols, el_calc_input, md_out, backend,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, **kwargs):
    # Because substitution has a sideeffect on self
    zmolecule = zmolecule.copy()
    zmat_values = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values
    symbolic_expressions = [s for s, v in symbols]

    def V(values=None, get_calculated=False,
          calculated=[]):  # pylint:disable=dangerous-default-value
        if get_calculated:
            return calculated
        elif values is not None:
            substitutions = list(zip(symbolic_expressions, values))
            new_zmat = zmolecule.subs(substitutions)

            result = calculate(
                molecule=new_zmat, forces=True, el_calc_input=el_calc_input,
                backend=backend, hamiltonian=hamiltonian, basis=basis,
                charge=charge, title=title,
                multiplicity=multiplicity,
                num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

            energy, grad_energy_X = result['energy'], result['gradient']

            grad_energy_C = _get_grad_energy_C(new_zmat, grad_energy_X)

            energy_symb = np.sum(zmat_values * grad_energy_C)
            grad_energy_symb = sympy.Matrix([
                energy_symb.diff(arg) for arg in symbolic_expressions])
            grad_energy_symb = np.array(grad_energy_symb.subs(substitutions))
            grad_energy_symb = grad_energy_symb.astype('f8').flatten()

            print(values)
            print(grad_energy_symb)

            new_zmat.metadata['energy'] = energy
            new_zmat.metadata['symbols'] = substitutions
            calculated.append({'energy': energy, 'structure': new_zmat})
            with open(md_out, 'a') as f:
                f.write(_get_table_row(calculated, grad_energy_symb))

            if is_converged(calculated, grad_energy_symb,
                            etol=etol, gtol=gtol):
                raise ConvergenceFinished(successful=True)
            elif len(calculated) >= max_iter:
                raise ConvergenceFinished(successful=False)

            return energy, grad_energy_symb
        else:
            raise ValueError
    return V


def _get_grad_energy_C(zmat, grad_energy_X):
    grad_X = zmat.get_grad_cartesian(as_function=False, drop_auto_dummies=True)
    grad_V_C = np.sum(grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))
    for i in range(min(3, grad_V_C.shape[0])):
        grad_V_C[i, i:] = 0.
    return grad_V_C


def _get_generic_optimise_header(
        zmolecule, backend, hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, start_time, num_procs, mem_per_proc):
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
    settings_table = _get_settings_table(
        backend=backend, hamiltonian=hamiltonian, charge=charge,
        multiplicity=multiplicity, basis=basis,
        etol=etol, gtol=gtol, max_iter=max_iter,
        num_procs=num_procs, mem_per_proc=mem_per_proc,)

    header = get_header(
        version=__version__, zmat=_get_geometry_markdown(zmolecule),
        cartesian=_get_geometry_markdown(zmolecule.get_cartesian()),
        calculation_setup=settings_table,
        start_time=_get_time_isostr(start_time),
        table_header=_get_table_header_generic_opt())
    return header


def _get_symb_optimise_header(
        zmolecule, symbols, backend, hamiltonian, basis, charge, title,
        multiplicity, etol, gtol, max_iter, start_time, num_procs,
        mem_per_proc):
    get_header = """\
# This is ChemOpt {version} optimising a molecule in internal coordinates.

## Settings for the calculations

{calculation_setup}

## Starting structures
### Starting structure as Zmatrix

{zmat}

### Symbols with starting values

{symbols}

## Iterations
Starting {start_time}

{table_header}
""".format
    settings_table = _get_settings_table(
        backend=backend, hamiltonian=hamiltonian, charge=charge,
        multiplicity=multiplicity, basis=basis,
        etol=etol, gtol=gtol, max_iter=max_iter,
        num_procs=num_procs, mem_per_proc=mem_per_proc)

    header = get_header(
        version=__version__, zmat=_get_geometry_markdown(zmolecule),
        calculation_setup=settings_table, symbols=_get_symbol_table(symbols),
        start_time=_get_time_isostr(start_time),
        table_header=_get_table_header_symb_opt())
    return header


def _get_symbol_table(symbols):
    return tabulate([(latex(sym_expr), v) for sym_expr, v in symbols],
                    tablefmt='pipe', headers=['Symbol', 'Start value'])


def _get_settings_table(backend, hamiltonian, charge, multiplicity,
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


def _get_geometry_markdown(molecule):
    return tabulate(molecule._frame, tablefmt='pipe', headers=molecule.columns,
                    floatfmt='.4f')


def _get_table_header_generic_opt():
    get_row = '|{:>4.4}|{:^16.16}|{:^16.16}|{:^31.31}|'.format
    header = (get_row('n', r'$E [E_h]$', r'$\Delta E [E_h]$',
                      r'$\max(|\nabla_X E |) [E_h$/Å]')
              + '\n'
              + get_row(3 * '-' + ':', 15 * '-' + ':',
                        15 * '-' + ':', 30 * '-' + ':'))
    return header


def _get_table_header_symb_opt():
    get_row = '|{:>4.4}|{:^16.16}|{:^16.16}|{:^31.31}|'.format
    header = (get_row('n', r'$E [E_h]$', r'$\Delta E [E_h]$',
                      r'$\max(|\nabla E |) [E_h$/Å]')
              + '\n'
              + get_row(3 * '-' + ':', 15 * '-' + ':',
                        15 * '-' + ':', 30 * '-' + ':'))
    return header


def _get_table_row(calculated, grad_energy):
    n = len(calculated)
    energy = calculated[-1]['energy']
    if n == 1:
        delta = 0.
    else:
        delta = calculated[-1]['energy'] - calculated[-2]['energy']
    # table header was:
    # n, energy, Delta energy, max(abs(grad_energy_X))
    get_str = '|{:>4}|{:16.10f}|{:16.10f}|{:31.10f}|\n'.format
    return get_str(n, energy, delta, abs(grad_energy).max())


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
        zmat=_get_geometry_markdown(opt_zmat),
        cartesian=_get_geometry_markdown(opt_zmat.get_cartesian()),
        molden=molden_out, end_time=_get_time_isostr(end_time),
        successfully='successfully' if successful else 'with errors',
        delta_time=str(end_time - start_time).split('.')[0])
    return output


def _get_time_isostr(time):
    return time.replace(microsecond=0).isoformat()


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


@substitute_docstr
def is_converged(calculated, grad_energy_X,
                 etol=fixed_defaults['etol'], gtol=fixed_defaults['gtol']):
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


def _get_default_filepaths(md_out, molden_out, el_calc_input):
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
