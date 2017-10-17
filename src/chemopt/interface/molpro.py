import inspect
import os
import subprocess
from io import StringIO
from os.path import splitext
from subprocess import run

import chemcoord as cc

import cclib
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)


@substitute_docstr
def calculate(molecule, hamiltonian, basis, molpro_exe=None,
              el_calc_input=None,
              charge=fixed_defaults['charge'],
              calculation_type=fixed_defaults['calculation_type'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              wfn_symmetry=fixed_defaults['wfn_symmetry']):
    """Calculate the energy of a molecule using Molpro.

    Args:
        el_calc_input (str): {el_calc_input}
        molecule (chemcoord.Cartesian or chemcoord.Zmat or str):
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        molpro_exe (str): {molpro_exe}
        charge (int): {charge}
        calculation_type (str): {calculation_type}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}


    Returns:
        cclib.Parser : A `cclib <https://cclib.github.io/>`_
        parsed data instance.
    """
    if molpro_exe is None:
        molpro_exe = conf_defaults['molpro_exe']
    if el_calc_input is None:
        el_calc_input = '{}.inp'.format(splitext(inspect.stack()[-1][1])[0])

    input_str = generate_input_file(
        molecule=molecule,
        hamiltonian=hamiltonian, basis=basis, charge=charge,
        calculation_type=calculation_type, forces=forces,
        title=title, multiplicity=multiplicity,
        wfn_symmetry=wfn_symmetry)

    input_path = el_calc_input
    output_path = '{}.out'.format(splitext(input_path)[0])
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    with open(input_path, 'w') as f:
        f.write(input_str)

    run([molpro_exe, input_path], stdout=subprocess.PIPE)

    return parse_output(output_path)


def parse_output(output_path):
    """Parse a molpro output file.

    Args:
        output_path (str):

    Returns:
        cclib.Parser : A `cclib <https://cclib.github.io/>`_
        parsed data instance.
    """
    return cclib.parser.molproparser.Molpro(output_path).parse()


@substitute_docstr
def generate_input_file(molecule, hamiltonian, basis, charge=0,
                        calculation_type='Single Point', forces=False,
                        title='', multiplicity=1, wfn_symmetry=1):
    """Generate a molpro input file.

    Args:
        molecule (str): Has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        charge (int): {charge}
        calculation_type (str): {calculation_type}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}


    Returns:
        str : Molpro input.
    """
    if isinstance(molecule, str):
        molecule = molecule.read_xyz(StringIO(molecule))
    elif isinstance(molecule, cc.Zmat):
        molecule = molecule.get_cartesian()

    get_output = """\
*** {title}

gprint, basis
gprint, orbital

basis, {basis_str}

geometry = {{
{geometry}
}}

{hamiltonian_str}
{forces}
{calculation_type}
---
""".format

    hamiltonian_str = _get_hamiltonian_str(
        hamiltonian, molecule.get_electron_number(charge),
        wfn_symmetry, multiplicity)

    out = get_output(title=title, basis_str=_get_basis_str(basis),
                     geometry=molecule.to_xyz(sort_index=False),
                     hamiltonian_str=hamiltonian_str,
                     forces='forces' if forces else '',
                     calculation_type=_get_calculation_type(calculation_type))
    return out


def _get_basis_str(basis):
    """Convert to code-specific strings
    """
    if basis in ['STO-3G', '3-21G', '6-31G', '6-31G(d)', '6-31G(d,p)',
                 '6-31+G(d)', '6-311G(d)']:
        basis_str = basis
    elif basis == 'cc-pVDZ':
        basis_str = 'vdz'
    elif basis == 'cc-pVTZ':
        basis_str = 'vtz'
    elif basis == 'AUG-cc-pVDZ':
        basis_str = 'avdz'
    elif basis == 'AUG-cc-pVTZ':
        basis_str = 'avtz'
    else:
        raise Exception('Unhandled basis type: {}'.format(basis))
    return basis_str


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
            raise Exception('Unhandled hamiltonian: {}'.format(hamiltonian))
        hamiltonian_str += '{{{}\n{}}}'.format(hamiltonian_key, wfn)
    return hamiltonian_str


def _get_calculation_type(calculation_type):
    calc_str = ''
    if calculation_type == 'Single Point':
        pass
    elif calculation_type == 'Equilibrium Geometry':
        calc_str = '{optg}\n'
    elif calculation_type == 'Frequencies':
        calc_str = '{optg}\n{frequencies}'
    else:
        raise Exception('Unhandled calculation type: %s' % calculation_type)
    return calc_str
