import inspect
import os
import re
import subprocess
from functools import partial
from io import StringIO
from itertools import islice
from os.path import splitext
from subprocess import run

import chemcoord as cc
import numpy as np
from datasize import DataSize

from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.constants import conv_factor


@substitute_docstr
def calculate(molecule, hamiltonian, basis, molcas_exe=None,
              el_calc_input=None,
              sym_group=None,
              charge=fixed_defaults['charge'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              start_orb=None,
              num_procs=None, mem_per_proc=None):
    """Calculate the energy of a molecule using Molcas.

    Args:
        el_calc_input (str): {el_calc_input}
        molecule (chemcoord.Cartesian or chemcoord.Zmat or str):
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        molcas_exe (str): {molcas_exe}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        start_orb (str): {start_orb}
        num_procs (int): {num_procs}
        mem_per_proc (str): {mem_per_proc}

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    if molcas_exe is None:
        molcas_exe = conf_defaults['molcas_exe']
    if num_procs is None:
        num_procs = conf_defaults['num_procs']
    if mem_per_proc is None:
        mem_per_proc = conf_defaults['mem_per_proc']
    if __name__ == '__main__' and el_calc_input is None:
        raise ValueError('el_calc_input has to be provided when executing '
                         'from an interactive session.')
    if el_calc_input is None:
        el_calc_input = '{}.inp'.format(splitext(inspect.stack()[-1][1])[0])

    input_str = generate_input_file(
        molecule=molecule,
        hamiltonian=hamiltonian, basis=basis, charge=charge,
        forces=forces,
        sym_group=sym_group,
        title=title, multiplicity=multiplicity,
        start_orb=start_orb,
        )

    input_path = el_calc_input
    output_path = f'{splitext(input_path)[0]}.log'
    dirname = os.path.dirname(input_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(input_path, 'w') as f:
        f.write(input_str)

    my_env = os.environ.copy()
    my_env['MOLCAS_NPROCS'] = str(num_procs)
    my_env['MOLCAS_MEM'] = str(mem_per_proc)
    run([molcas_exe, '-f', input_path], env=my_env, stdout=subprocess.PIPE)

    return parse_output(output_path)


def parse_output(output_path):
    """Parse a molcas output file.

    Args:
        output_path (str):

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    def read_gradient(f, n_atoms):
        gradient = []
        lines = islice(f, n_atoms)
        for line in lines:
            gradient.append([float(x) for x in line.split()[1:]])
        gradient = np.array(gradient)
        return gradient

    def read_structure(f):
        atoms, coordinates = [], []
        line = f.readline()
        while line != '\n':
            line = line.split()
            atoms.append(line[1])
            coordinates.append(line[5:])
            line = f.readline()
        molecule = cc.Cartesian(atoms=atoms, coords=coordinates)
        remove_digits = partial(re.sub, r'[0-9]+', '')
        molecule['atom'] = molecule['atom'].apply(remove_digits)
        return molecule


    energy = re.compile(
        '.*(RASSCF root number  1 Total energy|'
            'CASPT2 Root  1     Total energy:|Total SCF energy)')
    output = {}
    with open(output_path, 'r') as f:
        line = f.readline()
        while line:
            if 'Cartesian Coordinates / Bohr, Angstrom' in line:
                for _ in range(3):
                    f.readline()
                molecule = read_structure(f)
            elif energy.match(line):
                output['energy'] = float(line.split()[-1])
            elif 'Molecular gradients' in line:
                for _ in range(7):
                    f.readline()
                output['gradient'] = read_gradient(f, len(molecule))
            line = f.readline()

    for key in output:
        molecule.metadata[key] = output[key]
    output['structure'] = molecule
    return output


@substitute_docstr
def generate_input_file(molecule, hamiltonian, basis,
                        charge=fixed_defaults['charge'],
                        forces=fixed_defaults['forces'],
                        title=fixed_defaults['title'],
                        start_orb=None,
                        sym_group=None,
                        multiplicity=fixed_defaults['multiplicity']):
    """Generate a molcas input file.

    Args:
        molecule (chemcoord.Cartesian or chemcoord.Zmat or str):
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}


    Returns:
        str : molcas input.
    """
    if isinstance(molecule, str):
        molecule = molecule.read_xyz(StringIO(molecule))
    elif isinstance(molecule, cc.Zmat):
        molecule = molecule.get_cartesian()

    get_output = """\
&GATEWAY
 Coord
 {geometry}
 Title = {title}
 Basis = {basis}
 {sym_group}

&SEWARD

{hamiltonian_str}

{forces}
""".format

    out = get_output(
        title=title, basis=basis, geometry=molecule.to_xyz(sort_index=False),
        sym_group='' if sym_group is None else f'group = {sym_group}',
        hamiltonian_str=_get_hamiltonian_str(hamiltonian, charge, multiplicity, start_orb),
        forces='&ALASKA' if forces else '')
    return out


def _get_hamiltonian_str(hamiltonian, charge, multiplicity, start_orb):
    if hamiltonian == 'RHF' or hamiltonian == 'B3LYP':
        if start_orb is not None:
            raise ValueError('Start Orb currently not supported.')
        H_str = f'&SCF\n Charge = {charge}\n Spin = {multiplicity}\n'
        if hamiltonian == 'B3LYP':
            H_str += f' KSDFT=B3LYP\n'
    elif hamiltonian == 'RASSCF' or hamiltonian == 'CASPT2':
        H_str = f'&RASSCF\n Charge = {charge}\n Spin = {multiplicity}\n'
        if start_orb is not None:
            H_str += f' INPORB = {start_orb}\n'
        if hamiltonian == 'CASPT2':
            H_str += '\n&CASPT2\n'
    else:
        raise ValueError(f'Unhandled hamiltonian: {hamiltonian}')
    return H_str


def _get_molcas_mem(byte):
    return byte / 1e6
