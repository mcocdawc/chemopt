import chemcoord as cc
from chemopt import export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)

from chemopt.interface import molpro


@export
@substitute_docstr
def calculate(molecule, hamiltonian, basis,
              el_calc_input=None, backend=None,
              charge=fixed_defaults['charge'],
              calculation_type=fixed_defaults['calculation_type'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              **kwargs):
    """Calculate the energy of a molecule.

    Args:
        molecule (:class:`~chemcoord.Cartesian` or :class:`~chemcoord.Zmat`):
        hamiltonian (str): {hamiltonian}
        basis (str): {basis}
        el_calc_input (str): {el_calc_input}
        backend (str): {backend}
        charge (int): {charge}
        calculation_type (str): {calculation_type}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}

    Returns:
        cclib.Parser : A `cclib <https://cclib.github.io/>`_
        parsed data instance.
    """
    if backend is None:
        backend = conf_defaults['backend']
    if isinstance(molecule, cc.Zmat):
        molecule = molecule.get_cartesian()
    if backend == 'molpro':
        return molpro.calculate(
            el_calc_input=el_calc_input, molecule=molecule,
            hamiltonian=hamiltonian, basis=basis, charge=charge,
            calculation_type=calculation_type, forces=forces, title=title,
            multiplicity=multiplicity, **kwargs)
    else:
        raise ValueError('Backend {} is not implemented.'.format(backend))
