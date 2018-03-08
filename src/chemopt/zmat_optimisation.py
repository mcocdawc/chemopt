import inspect
import os
from datetime import datetime
from os.path import basename, isfile, join, normpath, splitext
from time import sleep

import chemcoord as cc
import numpy as np
import sympy
from chemcoord.xyz_functions import to_molden
from scipy.optimize import minimize
from sympy import latex
from tabulate import tabulate

from chemopt import __version__, export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.exception import ConvergenceFinished
from chemopt.interface import molcas, molpro


@export
@substitute_docstr
def optimise(zmolecule, hamiltonian, basis,
             symbols=None,
             md_out=None, el_calc_dir=None, molden_out=None,
             etol=fixed_defaults['etol'],
             gtol=fixed_defaults['gtol'],
             max_iter=fixed_defaults['max_iter'],
             backend=conf_defaults['backend'],
             charge=fixed_defaults['charge'],
             title=fixed_defaults['title'],
             multiplicity=fixed_defaults['multiplicity'],
             num_procs=None, mem_per_proc=None,
             start_orb=None,
             coord_fmt=fixed_defaults['coord_fmt'],
             **kwargs):
    """Optimize a molecule.

    Args:
        zmolecule (chemcoord.Zmat):
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        symbols (list): A list of tuples. Each tuple consists of a
            sympy symbolic expression and a starting value.
            An example is: ``[(r, 1.1), (alpha, 120)]``.
            Has exactly the same format as the multi-parameter substitution
            in sympy.
        el_calc_dir (str): Specify the input filename for electronic
            calculations. If it is None, the filename of the calling
            python script is used (With the suffix ``.inp`` instead of ``.py``)
            and the files for the electronic calculations will reside in their
            own directory.
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
        start_orb (str): {start_orb}
        coord_fmt (str): {coord_fmt}

    Returns:
        list: A list of dictionaries. The last one is the optimised structure.
        The keys of each dictionary depend on the used optimisation.
        In any case each dictionary has at least two keys:

        * 'energy': The energy in Hartree.
        * 'structure': The Zmatrix.

        If ``symbols`` was ``None`` a generic optimisation was performed and
        the following keys are available:

        * 'grad_energy': The energy gradient ('grad_energy')
        in internal coordinates.
        The units are Hartree / Angstrom for bonds and
        Hartree / radians for angles and dihedrals.

        If ``symbols`` was not ``None`` an optimisation with
        reduced degrees of freedom was performed and
        the following keys are available:

        * 'symbols': A list of tuples containing the symbol and its value.

    """
    files = _get_default_filepaths(md_out, molden_out, el_calc_dir)
    for filepath in files:
        rename_existing(filepath)
    md_out, molden_out, el_calc_dir = files

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
            etol=etol, gtol=gtol, max_iter=max_iter, coord_fmt=coord_fmt)
        with open(md_out, 'w') as f:
            f.write(header)
        calculated, convergence = _zmat_generic_optimise(
            zmolecule=zmolecule, el_calc_dir=el_calc_dir,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, start_orb=start_orb, **kwargs)
    else:
        header = _get_symb_optimise_header(
            zmolecule=zmolecule, symbols=symbols, backend=backend,
            hamiltonian=hamiltonian, basis=basis, charge=charge,
            multiplicity=multiplicity, num_procs=num_procs,
            mem_per_proc=mem_per_proc, start_time=t1, title=title,
            etol=etol, gtol=gtol, max_iter=max_iter, coord_fmt=coord_fmt)
        with open(md_out, 'w') as f:
            f.write(header)
        calculated, convergence = _zmat_symb_optimise(
            zmolecule=zmolecule, symbols=symbols, el_calc_dir=el_calc_dir,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc,
            start_orb=start_orb, **kwargs)

    to_molden(
        [x['structure'].get_cartesian() for x in calculated], buf=molden_out)
    t2 = datetime.now()
    if symbols is None:
        with open(md_out, 'a') as f:
            footer = _get_generic_footer(
                opt_zmat=calculated[-1]['structure'], start_time=t1,
                end_time=t2, molden_out=molden_out,
                successful=convergence.successful, n_iter=len(calculated),
                coord_fmt=coord_fmt)
            f.write(footer)
    else:
        with open(md_out, 'a') as f:
            footer = _get_symbolic_footer(
                symbols=calculated[-1]['symbols'],
                opt_zmat=calculated[-1]['structure'], start_time=t1,
                end_time=t2, molden_out=molden_out,
                successful=convergence.successful, n_iter=len(calculated),
                coord_fmt=coord_fmt)
            f.write(footer)
    return calculated


def _zmat_generic_optimise(
        zmolecule, el_calc_dir, md_out, backend, hamiltonian, basis, charge,
        title, multiplicity, etol, gtol, max_iter,
        num_procs, mem_per_proc, start_orb, **kwargs):
    def _get_C_rad(zmolecule):
        C_rad = zmolecule.loc[:, ['bond', 'angle', 'dihedral']].values.T
        C_rad[[1, 2], :] = np.radians(C_rad[[1, 2], :])
        return C_rad.flatten(order='F')

    if backend == 'molcas':
        V, grad_V = _get_generic_opt_V_molcas(
            zmolecule=zmolecule, el_calc_dir=el_calc_dir,
            md_out=md_out, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, start_orb=start_orb, **kwargs)
    elif backend == 'molpro':
        V, grad_V = _get_generic_opt_V_molpro(
            zmolecule=zmolecule, el_calc_dir=el_calc_dir,
            md_out=md_out, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

    try:
        opt = minimize(V, x0=_get_C_rad(zmolecule), jac=grad_V, method='BFGS',
                       options={'gtol': 1e-10})
    except ConvergenceFinished as e:
        convergence = e
    else:
        if opt.success:
            convergence = ConvergenceFinished(successful=True)
        else:
            convergence = ConvergenceFinished(successful=False)

    calculated = grad_V(get_calculated=True)
    return calculated, convergence


def _zmat_symb_optimise(
        zmolecule, symbols, el_calc_dir, md_out, backend,
        hamiltonian, basis, charge,
        title, multiplicity, etol, gtol, max_iter,
        num_procs, mem_per_proc, start_orb, **kwargs):
    if backend == 'molcas':
        V, grad_V = _get_symbolic_opt_V_molcas(
            zmolecule=zmolecule, symbols=symbols, el_calc_dir=el_calc_dir,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, start_orb=start_orb, **kwargs)
    elif backend == 'molpro':
        V, grad_V = _get_symbolic_opt_V_molpro(
            zmolecule=zmolecule, symbols=symbols, el_calc_dir=el_calc_dir,
            md_out=md_out, backend=backend, hamiltonian=hamiltonian,
            basis=basis, charge=charge, title=title, multiplicity=multiplicity,
            etol=etol, gtol=gtol, max_iter=max_iter,
            num_procs=num_procs, mem_per_proc=mem_per_proc, start_orb=start_orb, **kwargs)
    try:
        opt = minimize(V, x0=np.array([v for s, v in symbols], dtype='f8'),
                       jac=grad_V, method='BFGS', options={'gtol': 1e-10})
    except ConvergenceFinished as e:
        convergence = e
    else:
        if opt.success:
            convergence = ConvergenceFinished(successful=True)
        else:
            convergence = ConvergenceFinished(successful=False)
    calculated = grad_V(get_calculated=True)
    return calculated, convergence


def _get_generic_opt_V_molcas(
        zmolecule, el_calc_dir, md_out,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, start_orb, **kwargs):
    def get_new_zmat(C_rad, previous_zmat):
        C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
        C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

        new_zm = previous_zmat.copy()
        new_zm.safe_loc[zmolecule.index, ['bond', 'angle', 'dihedral']] = C_deg
        return new_zm

    base = splitext(basename(inspect.stack()[-1][1]))[0]
    def input_path(n):
        return join(el_calc_dir, '{}_{:03d}.inp'.format(base, n))

    def output_path(n):
        return join(el_calc_dir, '{}_{:03d}.log'.format(base, n))

    def start_orb_path(n):
        if hamiltonian == 'SCF' or hamiltonian == 'B3LYP':
            return join(el_calc_dir, '{}_{:03d}.ScfOrb'.format(base, n))
        elif hamiltonian == 'RASSCF' or hamiltonian == 'CASPT2':
            return join(el_calc_dir, '{}_{:03d}.RasOrb'.format(base, n))

    def V(C_rad=None, calculated=[]):  # pylint:disable=dangerous-default-value
        try:
            previous_zmat = calculated[-1].copy()
        except IndexError:
            new_zmat = zmolecule.copy()
        else:
            new_zmat = get_new_zmat(C_rad, previous_zmat)

        if isfile(input_path(len(calculated))):
            result = {}
            while len(result.keys()) < 3:
                try:
                    result = molcas.parse_output(output_path(len(calculated)))
                except FileNotFoundError:
                    pass
                sleep(0.5)
        else:
            result = molcas.calculate(
                molecule=new_zmat, forces=True,
                el_calc_input=input_path(len(calculated)),
                start_orb=start_orb_path(len(calculated) - 1) if calculated else start_orb,
                hamiltonian=hamiltonian, basis=basis,
                charge=charge, title=title,
                multiplicity=multiplicity,
                num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

        calculated.append(new_zmat)
        return result['energy']


    def grad_V(C_rad=None, get_calculated=False,
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

            if isfile(input_path(len(calculated))):
                result = {}
                while len(result.keys()) < 3:
                    try:
                        result = molcas.parse_output(output_path(len(calculated)))
                    except FileNotFoundError:
                        pass
                    sleep(0.5)
            else:
                result = molcas.calculate(
                    molecule=new_zmat, forces=True,
                    el_calc_input=input_path(len(calculated)),
                    start_orb=start_orb_path(len(calculated) - 1) if calculated else start_orb,
                    hamiltonian=hamiltonian, basis=basis,
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

            return grad_energy_C.flatten()
        else:
            raise ValueError

    return V, grad_V


def _get_generic_opt_V_molpro(
        zmolecule, el_calc_dir, md_out,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, start_orb, **kwargs):
    def get_new_zmat(C_rad, previous_zmat):
        C_deg = C_rad.copy().reshape((3, len(C_rad) // 3), order='F').T
        C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])

        new_zm = previous_zmat.copy()
        new_zm.safe_loc[zmolecule.index, ['bond', 'angle', 'dihedral']] = C_deg
        return new_zm

    base = splitext(basename(inspect.stack()[-1][1]))[0]
    def input_path(n):
        return join(el_calc_dir, '{}_{:03d}.inp'.format(base, n))

    def output_path(n):
        return join(el_calc_dir, '{}_{:03d}.out'.format(base, n))

    def V(C_rad=None, calculated=[]):  # pylint:disable=dangerous-default-value
        try:
            previous_zmat = calculated[-1].copy()
        except IndexError:
            new_zmat = zmolecule.copy()
        else:
            new_zmat = get_new_zmat(C_rad, previous_zmat)

        if isfile(input_path(len(calculated))):
            result = {}
            while len(result.keys()) < 3:
                try:
                    result = molpro.parse_output(output_path(len(calculated)))
                except FileNotFoundError:
                    pass
                sleep(0.5)
        else:
            result = molpro.calculate(
                molecule=new_zmat, forces=True,
                el_calc_input=input_path(len(calculated)),
                hamiltonian=hamiltonian, basis=basis,
                charge=charge, title=title,
                multiplicity=multiplicity,
                num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

        calculated.append(new_zmat)
        return result['energy']


    def grad_V(C_rad=None, get_calculated=False,
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

            if isfile(input_path(len(calculated))):
                result = {}
                while len(result.keys()) < 3:
                    try:
                        result = molcas.parse_output(output_path(len(calculated)))
                    except FileNotFoundError:
                        pass
                    sleep(0.5)
            else:
                result = molpro.calculate(
                    molecule=new_zmat, forces=True,
                    el_calc_input=input_path(len(calculated)),
                    hamiltonian=hamiltonian, basis=basis,
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

            return grad_energy_C.flatten()
        else:
            raise ValueError

    return V, grad_V


def _get_symbolic_opt_V_molcas(
        zmolecule, symbols, el_calc_dir, md_out, backend,
        hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, num_procs, mem_per_proc, start_orb, **kwargs):
    # Because substitution has a sideeffect on self
    zmolecule = zmolecule.copy()
    value_cols = ['bond', 'angle', 'dihedral']
    symbolic_expressions = [s for s, v in symbols]
    base = splitext(basename(inspect.stack()[-1][1]))[0]
    def input_path(n):
        return join(el_calc_dir, '{}_{:03d}.inp'.format(base, n))

    def output_path(n):
        return join(el_calc_dir, '{}_{:03d}.log'.format(base, n))

    def start_orb_path(n):
        if hamiltonian == 'SCF' or hamiltonian == 'B3LYP':
            return join(el_calc_dir, '{}_{:03d}.ScfOrb'.format(base, n))
        elif hamiltonian == 'RASSCF' or hamiltonian == 'CASPT2':
            return join(el_calc_dir, '{}_{:03d}.RasOrb'.format(base, n))

    def V(values=None):  # pylint:disable=dangerous-default-value
        if hasattr(V, "counter"):
            V.counter += 1
        else:
            V.counter = 0
        substitutions = list(zip(symbolic_expressions, values))
        new_zmat = zmolecule.subs(substitutions)

        if isfile(input_path(V.counter)):
            result = {}
            while len(result.keys()) < 3:
                try:
                    result = molcas.parse_output(output_path(V.counter))
                except FileNotFoundError:
                    pass
                sleep(0.5)
        else:
            result = molcas.calculate(
                molecule=new_zmat, forces=True,
                el_calc_input=input_path(V.counter),
                start_orb=start_orb_path(V.counter - 1) if V.counter else start_orb,
                hamiltonian=hamiltonian, basis=basis,
                charge=charge, title=title,
                multiplicity=multiplicity,
                num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

        return result['energy']

    def grad_V(values=None, get_calculated=False,
          calculated=[]):  # pylint:disable=dangerous-default-value
        if get_calculated:
            return calculated
        elif values is not None:
            substitutions = list(zip(symbolic_expressions, values))
            new_zmat = zmolecule.subs(substitutions)

            if isfile(input_path(len(calculated))):
                result = {}
                while len(result.keys()) < 3:
                    try:
                        result = molcas.parse_output(output_path(len(calculated)))
                    except FileNotFoundError:
                        pass
                    sleep(0.5)
            else:
                result = molcas.calculate(
                    molecule=new_zmat, forces=True,
                    el_calc_input=input_path(len(calculated)),
                    start_orb=start_orb_path(len(calculated) - 1) if calculated else start_orb,
                    hamiltonian=hamiltonian, basis=basis,
                    charge=charge, title=title,
                    multiplicity=multiplicity,
                    num_procs=num_procs, mem_per_proc=mem_per_proc, **kwargs)

            energy, grad_energy_X = result['energy'], result['gradient']

            grad_energy_C = _get_grad_energy_C(new_zmat, grad_energy_X)
            zm_values_rad = zmolecule.loc[:, value_cols].values
            zm_values_rad[:, [1, 2]] = sympy.rad(zm_values_rad[:, [1, 2]])
            energy_symb = np.sum(zm_values_rad * grad_energy_C)
            grad_energy_symb = sympy.Matrix([
                energy_symb.diff(arg) for arg in symbolic_expressions])
            grad_energy_symb = np.array(grad_energy_symb.subs(substitutions))
            grad_energy_symb = grad_energy_symb.astype('f8').flatten()

            new_zmat.metadata['energy'] = energy
            new_zmat.metadata['symbols'] = substitutions
            calculated.append({'energy': energy, 'structure': new_zmat,
                               'symbols': substitutions})
            with open(md_out, 'a') as f:
                f.write(_get_table_row(calculated, grad_energy_symb))

            if is_converged(calculated, etol=etol):
                raise ConvergenceFinished(successful=True)
            elif len(calculated) >= max_iter:
                raise ConvergenceFinished(successful=False)

            return grad_energy_symb
        else:
            raise ValueError



    return V, grad_V


def _get_grad_energy_C(zmat, grad_energy_X):
    grad_X = zmat.get_grad_cartesian(as_function=False, drop_auto_dummies=True)
    grad_V_C = np.sum(grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))
    for i in range(min(3, grad_V_C.shape[0])):
        grad_V_C[i, i:] = 0.
    return grad_V_C


def _get_generic_optimise_header(
        zmolecule, backend, hamiltonian, basis, charge, title, multiplicity,
        etol, gtol, max_iter, start_time, num_procs, mem_per_proc, coord_fmt):
    get_header = """\
# This is ChemOpt {version} optimising a molecule in internal coordinates.

Written by Oskar Weser (oskar.weser@gmail.com)

## Input File

```python
{input_file}
```

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

    input_filepath = basename(inspect.stack()[-1][1])

    # Yes I know: One should not use readlines.
    # BUT: The Input file won't be large!
    with open(input_filepath, 'r') as f:
        input_file = ''.join(f.readlines())

    header = get_header(
        version=__version__,
        input_file=input_file,
        zmat=_get_geometry_markdown(zmolecule, coord_fmt=coord_fmt),
        cartesian=_get_geometry_markdown(
            zmolecule.get_cartesian(), coord_fmt=coord_fmt),
        calculation_setup=settings_table,
        start_time=_get_time_isostr(start_time),
        table_header=_get_table_header_generic_opt())
    return header


def _get_symb_optimise_header(
        zmolecule, symbols, backend, hamiltonian, basis, charge, title,
        multiplicity, etol, gtol, max_iter, start_time, num_procs,
        mem_per_proc, coord_fmt):
    get_header = """\
# This is ChemOpt {version} optimising a molecule in internal coordinates.

Written by Oskar Weser (oskar.weser@gmail.com)

## Input File

```python
{input_file}
```

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

    input_filepath = basename(inspect.stack()[-1][1])

    # Yes I know: One should not use readlines.
    # BUT: The Input file won't be large!
    with open(input_filepath, 'r') as f:
        input_file = ''.join(f.readlines())

    header = get_header(
        version=__version__,
        input_file=input_file,
        zmat=_get_geometry_markdown(zmolecule, coord_fmt=coord_fmt),
        calculation_setup=settings_table,
        symbols=_get_symbol_table(symbols, header=['Symbol', 'Start Value']),
        start_time=_get_time_isostr(start_time),
        table_header=_get_table_header_symb_opt())
    return header


def _get_symbol_table(symbols, header):
    latex_table = [('${}$'.format(latex(symb)), v) for symb, v in symbols]
    return tabulate(latex_table,
                    tablefmt='pipe', headers=header)


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


def _get_geometry_markdown(molecule, coord_fmt=fixed_defaults['coord_fmt']):
    def formatter(x):
        try:
            return '{{:{}}}'.format(coord_fmt).format(x)
        except (TypeError, AttributeError, ValueError):
            return x

    latex_symb = molecule._sympy_formatter()
    if isinstance(molecule, cc.Cartesian):
        to_be_printed = latex_symb._frame.loc[:, ['atom', 'x', 'y', 'z']]
        for col in ['x', 'y', 'z']:
            to_be_printed[col] = to_be_printed[col].apply(formatter)
    elif isinstance(molecule, cc.Zmat):
        columns = ['atom', 'b', 'bond', 'a', 'angle', 'd', 'dihedral']
        latex_symb = latex_symb._abs_ref_formatter(format_as='latex')
        to_be_printed = latex_symb._frame.loc[:, columns]
        for col in ['bond', 'angle', 'dihedral']:
            to_be_printed[col] = to_be_printed[col].apply(formatter)

    return tabulate(to_be_printed, tablefmt='pipe', headers='keys',
                    stralign='right')


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


def _get_generic_footer(opt_zmat, start_time, end_time,
                        molden_out, successful, n_iter, coord_fmt):
    get_output = """\

## Optimised Structures
### Optimised structure as Zmatrix

{zmat}


### Optimised structure in cartesian coordinates

{cartesian}

## Closing

Structures were written to {molden}.

The calculation finished {successfully}
after {n_iter} iterations at: {end_time}
and needed: {delta_time}.
""".format
    output = get_output(
        zmat=_get_geometry_markdown(opt_zmat, coord_fmt=coord_fmt),
        cartesian=_get_geometry_markdown(
            opt_zmat.get_cartesian(), coord_fmt=coord_fmt),
        molden=molden_out, end_time=_get_time_isostr(end_time),
        successfully='successfully' if successful else 'with errors',
        n_iter=n_iter, delta_time=str(end_time - start_time).split('.')[0])
    return output


def _get_symbolic_footer(symbols, opt_zmat, start_time, end_time,
                         molden_out, successful, n_iter, coord_fmt):
    get_output = """\

## Optimised Structures

### Symbols with end values

{symbols_end_values}


### Optimised structure as Zmatrix

{zmat}


### Optimised structure in cartesian coordinates

{cartesian}

## Closing

Structures were written to {molden}.

The calculation finished {successfully}
after {n_iter} iterations at: {end_time}
and needed: {delta_time}.
""".format
    symbol_table = _get_symbol_table(symbols, header=['Symbol', 'End Value'])
    output = get_output(
        symbols_end_values=symbol_table,
        zmat=_get_geometry_markdown(opt_zmat, coord_fmt=coord_fmt),
        cartesian=_get_geometry_markdown(
            opt_zmat.get_cartesian(), coord_fmt=coord_fmt),
        molden=molden_out, end_time=_get_time_isostr(end_time),
        successfully='successfully' if successful else 'with errors',
        n_iter=n_iter, delta_time=str(end_time - start_time).split('.')[0])
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
def is_converged(calculated, grad_energy_X=None,
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
    if len(calculated) in {0, 1, 2}:
        return False
    else:
        delta_energy = calculated[-1]['energy'] - calculated[-2]['energy']
        if grad_energy_X is None:
            return abs(delta_energy) < etol
        else:
            return abs(delta_energy) < etol and abs(grad_energy_X).max() < gtol


def _get_default_filepaths(md_out, molden_out, el_calc_dir):
    if __name__ == '__main__':
        if md_out is None:
            raise ValueError('md_out has to be provided when executing '
                             'from an interactive session.')
        if molden_out is None:
            raise ValueError('molden_out has to be provided when executing '
                             'from an interactive session.')
        if el_calc_dir is None:
            raise ValueError('el_calc_dir has to be provided when executing '
                             'from an interactive session.')
    else:
        base = splitext(basename(inspect.stack()[-1][1]))[0]
        if md_out is None:
            md_out = '{}.md'.format(base)
        if molden_out is None:
            molden_out = '{}.molden'.format(base)
        if el_calc_dir is None:
            el_calc_dir = '{}_el_calcs'.format(base)
        return md_out, molden_out, el_calc_dir
