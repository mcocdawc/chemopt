from chemopt import export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)

from chemopt.interface import molpro, molcas


@export
@substitute_docstr
def calculate(molecule, hamiltonian, basis,
              el_calc_input=None, backend=None,
              charge=fixed_defaults['charge'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              **kwargs):
    """Calculate the energy of a molecule.

    Args:
        molecule (chemcoord.Cartesian or chemcoord.Zmat or str):
            If it is a string, it has to be a valid xyz-file.
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        el_calc_input (str): {el_calc_input}
        backend (str): {backend}
        charge (int): {charge}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}

    Returns:
        dict: A dictionary with at least the keys
        ``'structure'`` and ``'energy'`` which contains the energy in Hartree.
        If forces were calculated, the key ``'gradient'`` contains the
        gradient in Hartree / Angstrom.
    """
    if backend is None:
        backend = conf_defaults['backend']
    if backend == 'molpro':
        return molpro.calculate(
            el_calc_input=el_calc_input, molecule=molecule,
            hamiltonian=hamiltonian, basis=basis, charge=charge,
            forces=forces, title=title,
            multiplicity=multiplicity)
    elif backend == 'molcas':
        return molcas.calculate(
            el_calc_input=el_calc_input, molecule=molecule,
            hamiltonian=hamiltonian, basis=basis, charge=charge,
            forces=forces, title=title,
            multiplicity=multiplicity)
    else:
        raise ValueError('Backend {} is not implemented.'.format(backend))
