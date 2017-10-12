import subprocess
from subprocess import run

import chemcoord as cc

import cclib
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)


@substitute_docstr
def calculate(base_filename, molecule, theory, basis, molpro_exe=None,
              charge=fixed_defaults['charge'],
              calculation_type=fixed_defaults['calculation_type'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              wfn_symmetry=fixed_defaults['wfn_symmetry']):
    """Calculate the energy of a molecule using Molpro.

    Args:
        base_filename (str): {base_filename}
        molecule (:class:`~chemcoord.Cartesian` or :class:`~chemcoord.Zmat`):
        theory (str): {theory}
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

    if isinstance(molecule, cc.Zmat):
        molecule = molecule.get_cartesian()
    input_str = generate_input_file(
        molecule=molecule, theory=theory, basis=basis, charge=charge,
        calculation_type=calculation_type, forces=forces,
        title=title, multiplicity=multiplicity,
        wfn_symmetry=wfn_symmetry)

    input_path = base_filename + '.inp'
    output_path = base_filename + '.out'
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


def generate_input_file(molecule, theory, basis, charge=0,
                        calculation_type='Single Point', forces=False,
                        title='', multiplicity=1, wfn_symmetry=1):
    """Generate a molpro input file.

    Args:
        molecule (:class:`~chemcoord.Cartesian` or :class:`~chemcoord.Zmat`):
        theory (str): {theory}
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
    get_output = """\
*** {title}

gprint, basis
gprint, orbital

basis, {basis_str}

geometry = {{
{geometry}
}}

{theory_str}
{forces}
{calculation_type}
---
""".format

    theory_str = _get_theory_str(theory, molecule.get_electron_number(),
                                 multiplicity, wfn_symmetry)
    out = get_output(title=title, basis_str=_get_basis_str(basis),
                     geometry=molecule.to_xyz(),
                     theory_str=theory_str,
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


def _get_wavefn_str(num_e, multiplicity, wfn_symmetry):
    return 'wf, {}, {}, {}'.format(num_e, wfn_symmetry, multiplicity - 1)


def _get_theory_str(theory, num_e, multiplicity, wfn_symmetry):
    wfn = _get_wavefn_str(num_e, multiplicity, wfn_symmetry)
    theory_str = ''
    if theory != 'B3LYP':
        theory_str += '{{rhf\n{wfn}}}'.format(wfn=wfn)
    # Intentionally not using elif here:
    if theory != 'RHF':
        theory_key = ''
        if theory in ['MP2', 'CCSD', 'CCSD(T)']:
            theory_key = theory.lower()
        elif theory == 'B3LYP':
            theory_key = 'uks, b3lyp'
        else:
            raise Exception('Unhandled theory type: {}'.format(theory))
        theory_str += '{{{}\n{}}}'.format(theory_key, wfn)
    return theory_str


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
