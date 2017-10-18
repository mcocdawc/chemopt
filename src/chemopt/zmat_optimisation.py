import inspect
import os
from collections import deque
from datetime import datetime
from os.path import basename, join, normpath, splitext

import numpy as np
from numpy import outer, inner, dot, concatenate, append
from numpy.linalg import multi_dot, eigh
import scipy.optimize

from cclib.parser.utils import convertor
from chemcoord.xyz_functions import to_molden
from chemopt.configuration import conf_defaults, fixed_defaults
from chemopt.interface.generic import calculate
from tabulate import tabulate


def optimise(zmolecule, symbols=None, md_out=None, el_calc_input=None,
             molden_out=None, opt_f=None, **kwargs):
    """Optimize a molecule.

    Args:
        zmolecule (chemcoord.Zmat):
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
    V = _get_V_function(zmolecule, el_calc_input, md_out, **kwargs)
    with open(md_out, 'w') as f:
        f.write(_get_header(zmolecule, start_time=_get_isostr(t1), **kwargs))

    energies, structures, gradients_energy_C = [], [], deque([])
    grad_energy_X = None
    hess = None
    zm = zmolecule.copy()
    get_new_zm = _get_new_zm_f_generator(zmolecule)
    n = 1
    while not is_converged(energies, grad_energy_X) and n < 100:
        zm, hess = get_new_zm(structures, gradients_energy_C, hess)
        energy, grad_energy_X, grad_energy_C = V(zm)
        zm.metadata['energy'] = energy
        structures.append(zm)
        energies.append(energy)
        gradients_energy_C.append(grad_energy_C.flatten())
        if len(gradients_energy_C) == 3:
            gradients_energy_C.popleft()
        with open(md_out, 'a') as f:
            f.write(_get_table_row(n, energies, grad_energy_X))
        n += 1

    to_molden([zm.get_cartesian() for zm in structures], buf=molden_out)
    t2 = datetime.now()
    with open(md_out, 'a') as f:
        footer = _get_footer(opt_zmat=structures[-1],
                             start_time=t1, end_time=t2,
                             molden_out=molden_out)
        f.write(footer)
    return structures, energies


def _get_V_function(zmolecule, el_calc_input, md_out, **kwargs):
    def V(zmolecule):
        result = calculate(molecule=zmolecule, forces=True,
                           el_calc_input=el_calc_input, **kwargs)
        energy = convertor(result.scfenergies[0], 'eV', 'hartree')
        grad_energy_X = result.grads[0] / convertor(1, 'bohr', 'Angstrom')

        grad_X = zmolecule.get_grad_cartesian(
            as_function=False, drop_auto_dummies=True)
        grad_energy_C = np.sum(
            grad_energy_X.T[:, :, None, None] * grad_X, axis=(0, 1))

        for i in range(min(3, grad_energy_C.shape[0])):
            grad_energy_C[i, i:] = 0.

        return energy, grad_energy_X, grad_energy_C
    return V


def _get_new_zm_f_generator(zmolecule):
    def get_new_zm(structures, gradients_energy_C, hess_old):
        zmat_values = ['bond', 'angle', 'dihedral']

        def get_C_rad(zmolecule):
            C = zmolecule.loc[:, zmat_values].values
            C[:, [1, 2]] = np.radians(C[:, [1, 2]])
            return C.flatten()

        if len(gradients_energy_C) == 0:
            new_zm, hess_new = zmolecule, None
        else:
            new_zm = structures[-1].copy()
            if len(gradients_energy_C) == 1:
                # @Thorsten here I need to introduce damping
                # @Oskar creating the parametrized Hessian can
                #       probably be done more elegantly (need natoms).
                hess_new = np.diag([0.5, 0.2, 0.1] *
                                   (len(gradients_energy_C[0]) // 3))
                damping = 0.3
                p = - damping * gradients_energy_C[0]
            else:
                last_two_C = [get_C_rad(zm) for zm in structures[-2:]]
                p, hess_new = get_next_step(last_two_C, gradients_energy_C, hess_old)

                # @Thorsten this works but is horribly slow
                # @Oskar let's comment it out then :)
                # hess_new = None
                # damping = 0.3
                # p = - damping * gradients_energy_C[1]

            C_deg = p.reshape((3, len(p) // 3), order='F').T
            C_deg[:, [1, 2]] = np.rad2deg(C_deg[:, [1, 2]])
            new_zm.safe_loc[zmolecule.index, zmat_values] += C_deg
        return new_zm, hess_new
    return get_new_zm


def _get_header(zmolecule, hamiltonian, basis, start_time, backend=None,
                charge=fixed_defaults['charge'], title=fixed_defaults['title'],
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
        get_row = '|{:>4.4}| {:^16.16} | {:^16.16} | {:^28.28} |'.format
        header = (get_row('n', 'energy [Hartree]',
                          'delta [Hartree]', 'grad_X_max [Hartree / Angstrom]')
                  + '\n'
                  + get_row(4 * '-', 16 * '-', 16 * '-', 28 * '-'))
        return header

    def _get_calc_setup(backend, hamiltonian, charge, multiplicity):
        data = [['Hamiltonian', hamiltonian],
                ['Charge', charge],
                ['Multiplicity', multiplicity]]
        return tabulate(data, tablefmt='pipe', headers=['Backend', backend])
    calculation_setup = _get_calc_setup(backend, hamiltonian, charge,
                                        multiplicity)

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


def _get_table_row(n, energies, grad_energy_X):
    energy = energies[-1]
    if n == 1:
        delta = 0.
    else:
        delta = energies[-1] - energies[-2]
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


def is_converged(energies, grad_energy_X, etol=1e-6, gtol=3e-4):
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
    if len(energies) == 0:
        return False
    elif len(energies) == 1:
        return False
    else:
        return (abs(energies[-1] - energies[-2]) < etol and
                abs(grad_energy_X).max() < gtol)


def get_next_step(last_two_C, gradients_energy_C, hess_old):
    r"""Returns the next step and approximated hessian in the BFGS algorithm.

    Args:
        last_two_C (list): A two list of the current and previous zmat values.
            The order is: ``[previous, current]``.
            Each array is flatted out in the following order

            .. math::

                \left[
                    r_1,
                    \alpha_1,
                    \delta_1,
                    r_2,
                    \alpha_2,
                    \delta_2,
                    ...
                \right]

            And again :math:`r_i, \alpha_i, \delta_i`
            are the bond, angle, and dihedral of the :math:`i`-th atom.
            The units are Angstrom and radians.
        gradients_energy_C (collections.deque): A two element deque,
            that contains
            the current and previous gradient in internal coordinates.
            The order is: ``[previous, current]``.
            Each gradient is flatted out in the following order

            .. math::

                \left[
                    \frac{\partial V}{\partial r_1},
                    \frac{\partial V}{\partial \alpha_1},
                    \frac{\partial V}{\partial \delta_1},
                    \frac{\partial V}{\partial r_2},
                    \frac{\partial V}{\partial \alpha_2},
                    \frac{\partial V}{\partial \delta_2},
                    ...
                \right]

            Here :math:`V` is the energy and :math:`r_i, \alpha_i, \delta_i`
            are the bond, angle, and dihedral of the :math:`i`-th atom.
            The units are:

            .. math::

                    &\frac{\partial V}{\partial r_i}
                    &\frac{\text{Hartree}}{\text{Angstrom}}
                \\
                    &\frac{\partial V}{\partial \alpha_i}
                    &\frac{\text{Hartree}}{\text{Radian}}
                \\
                    &\frac{\partial V}{\partial \delta_i}
                    &\frac{\text{Hartree}}{\text{Radian}}


    Returns:
        numpy.ndarray, numpy.ndarray:
        First float array is the next step,
        the second float array is the new approximated hessian.
    """
    if len(gradients_energy_C) != 2:
        raise ValueError('Only deques of length 2 allowed')
    dg = gradients_energy_C[1] - gradients_energy_C[0]
    dx = last_two_C[1] - last_two_C[0]

    GxxtG = multi_dot([hess_old, outer(dx, dx), hess_old])
    xtGx = multi_dot([dx, hess_old, dx])
    correction = outer(dg, dg) / inner(dg, dx) - GxxtG / xtGx
    hess_new = hess_old + correction

    # step determination by rational function method
    long_grad = append(gradients_energy_C[1], 0)
    # print(hess_new.shape, gradients_energy_C[1].shape, long_grad.shape)
    aug_hess = concatenate((hess_new, gradients_energy_C[1][None, :]), axis=0)
    aug_hess = concatenate((aug_hess, long_grad[:, None]), axis=1)
    evals, evecs = eigh(aug_hess)
    lowest_evec = evecs[:, np.argmin(evals)]
    # lowest_evec[-1] might be very low, maybe implement warnings?
    next_step = lowest_evec[:-1] / lowest_evec[-1]


    return next_step, hess_new
