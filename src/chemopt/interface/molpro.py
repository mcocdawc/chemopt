import inspect
import os
import subprocess
from io import StringIO
from os.path import splitext
from subprocess import run
from datasize import DataSize

import chemcoord as cc
import numpy as np
import re

from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)
from chemopt.constants import conv_factor


@substitute_docstr
def calculate(molecule, hamiltonian, basis, molpro_exe=None,
              el_calc_input=None,
              charge=fixed_defaults['charge'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              wfn_symmetry=fixed_defaults['wfn_symmetry'],
              num_procs=None, mem_per_proc=None):
    """Calculate the energy of a molecule using Molpro.

    Args:
        el_calc_input (str): {el_calc_input}
        molecule (chemcoord.Cartesian or chemcoord.Zmat or str):
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        molpro_exe (str): {molpro_exe}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}
        num_procs (int): {num_procs}
        mem_per_proc (str): {mem_per_proc}

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    if molpro_exe is None:
        molpro_exe = conf_defaults['molpro_exe']
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
        title=title, multiplicity=multiplicity,
        wfn_symmetry=wfn_symmetry)

    input_path = el_calc_input
    output_path = '{}.out'.format(splitext(input_path)[0])
    dirname = os.path.dirname(input_path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(input_path, 'w') as f:
        f.write(input_str)

    run([molpro_exe, '-n {}'.format(num_procs), input_path],
        stdout=subprocess.PIPE)

    return parse_output(output_path)


def parse_output(output_path):
    """Parse a molpro output file.

    Args:
        output_path (str):

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    def read_gradient(f, n_atoms):
        for _ in range(3):
            f.readline()
        gradient = []
        lines_read = 0
        while lines_read < n_atoms:
            line = f.readline()
            if line != '\n':
                gradient.append([float(x) for x in line.split()[1:]])
                lines_read += 1
        gradient = np.array(gradient)
        gradient /= conv_factor('Bohr', 'Angstrom')
        return gradient

    def read_structure(f):
        beggining = f.tell()
        n_atoms = int(f.readline())
        f.seek(beggining)
        molecule = cc.Cartesian.read_xyz(f, nrows=n_atoms, engine='python')
        return molecule

    scf_energy = re.compile('\s*!(RHF|UHF|RKS) STATE 1.1 Energy')
    output = {}
    with open(output_path, 'r') as f:
        line = f.readline()
        while line:
            if 'geometry = {' in line:
                molecule = read_structure(f)
            elif scf_energy.match(line):
                output['energy'] = float(line.split()[-1])
            elif 'GRADIENT FOR STATE' in line:
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
                        multiplicity=fixed_defaults['multiplicity'],
                        wfn_symmetry=fixed_defaults['wfn_symmetry'],
                        mem_per_proc=None):
    """Generate a molpro input file.

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
        mem_per_proc (str): {mem_per_proc}


    Returns:
        str : Molpro input.
    """
    if isinstance(molecule, str):
        molecule = molecule.read_xyz(StringIO(molecule))
    elif isinstance(molecule, cc.Zmat):
        molecule = molecule.get_cartesian()
    if mem_per_proc is None:
        mem_per_proc = conf_defaults['mem_per_proc']

    get_output = """\
*** {title}
memory, {memory}

basis, {basis_str}

geometry = {{
{geometry}
}}

{hamiltonian_str}
{forces}
---
""".format

    hamiltonian_str = _get_hamiltonian_str(
        hamiltonian, molecule.get_electron_number(charge),
        wfn_symmetry, multiplicity)

    out = get_output(title=title, basis_str=basis,
                     geometry=molecule.to_xyz(sort_index=False),
                     hamiltonian_str=hamiltonian_str,
                     forces='forces' if forces else '',
                     memory=_get_molpro_mem(DataSize(mem_per_proc)))
    return out


def _get_wavefn_str(num_e, wfn_symmetry, multiplicity):
    return 'wf, {}, {}, {}'.format(num_e, wfn_symmetry, multiplicity - 1)


def _get_hamiltonian_str(hamiltonian, num_e, wfn_symmetry, multiplicity):
    wfn = _get_wavefn_str(num_e, wfn_symmetry, multiplicity)
    hamiltonian_str = ''
    if hamiltonian != 'B3LYP':
        hamiltonian_str += '{{rhf\n{wfn}}}'.format(wfn=wfn)
    # Intentionally not using elif here:
    if hamiltonian != 'RHF':
        hamiltonian_key = ''
        if hamiltonian in ['MP2', 'CCSD', 'CCSD(T)']:
            hamiltonian_key = hamiltonian.lower()
        elif hamiltonian == 'B3LYP':
            hamiltonian_key = 'uks, b3lyp'
        else:
            raise ValueError('Unhandled hamiltonian: {}'.format(hamiltonian))
        hamiltonian_str += '{{{}\n{}}}'.format(hamiltonian_key, wfn)
    return hamiltonian_str


def _get_molpro_mem(byte):
    word = byte // 8
    for unit in ['', 'k', 'm', 'g']:
        if word <= 1000 or unit == 'g':
            return '{:.2f}, {}'.format(float(word), unit)
        word /= (10 ** 3)
