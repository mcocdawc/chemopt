from . import molpro
from chemopt import export
from chemopt.configuration import (conf_defaults, fixed_defaults,
                                   substitute_docstr)


@export
@substitute_docstr
def calculate(base_filename, molecule, theory, basis, backend=None,
              charge=fixed_defaults['charge'],
              calculation_type=fixed_defaults['calculation_type'],
              forces=fixed_defaults['forces'],
              title=fixed_defaults['title'],
              multiplicity=fixed_defaults['multiplicity'],
              wfn_symmetry=fixed_defaults['wfn_symmetry'], **kwargs):
    """Calculate the energy of a molecule.

    Args:
        base_filename (str): {base_filename}
        molecule (:class:`~chemcoord.Cartesian`):
            A molecule in cartesian coordinates.
        theory (str): {theory}
        basis (str): {basis}
        backend (str): {backend}
        molpro_exe (str): {molpro_exe}
        charge (int): {charge}
        calculation_type (str): {calculation_type}
        forces (bool): {forces}
        title (str): {title}
        multiplicity (int): {multiplicity}
        wfn_symmetry (int): {wfn_symmetry}


    Returns:
        :class:`chemcoord.Cartesian`: A new cartesian instance.
    """
    if backend is None:
        backend = conf_defaults['backend']
    if backend == 'molpro':
        return molpro.calculate(
            base_filename=base_filename, molecule=molecule,
            theory=theory, basis=basis, charge=charge,
            calculation_type=calculation_type, forces=forces, title=title,
            multiplicity=multiplicity, wfn_symmetry=wfn_symmetry, **kwargs)
    else:
        raise ValueError('Backend {} is not implemented.'.format(backend))
