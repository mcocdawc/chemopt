import chemcoord as cc

from chemopt import export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)

from . import molpro


@export
@substitute_docstr
def calculate(molecule, theory, basis,
              base_filename, backend=None,
              charge=fixed_defaults['charge'],
              calculation_type=fixed_defaults['calculation_type'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              **kwargs):
    """Calculate the energy of a molecule.

    Args:
        molecule (chemcoord.Cartesian or chemcoord.Zmat):
        theory (str): {theory}
        basis (str): {basis}
        base_filename (str): {base_filename}
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
            base_filename=base_filename, molecule=molecule,
            theory=theory, basis=basis, charge=charge,
            calculation_type=calculation_type, forces=forces, title=title,
            multiplicity=multiplicity, **kwargs)
    else:
        raise ValueError('Backend {} is not implemented.'.format(backend))
