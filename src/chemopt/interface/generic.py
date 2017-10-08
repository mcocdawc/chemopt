from . import molpro
from chemopt import export


@export
def calculate(backend, base_filename, molecule, theory, basis,
              charge=0, calculation_type='Single Point', forces=False,
              title='', multiplicity=1, wfn_symmetry=1, **kwargs):
    if backend == 'molpro':
        return molpro.calculate(
            base_filename=base_filename, molecule=molecule,
            theory=theory, basis=basis, charge=charge,
            calculation_type=calculation_type, forces=forces, title=title,
            multiplicity=multiplicity, wfn_symmetry=wfn_symmetry, **kwargs)
    else:
        raise ValueError('Backend {} is not implemented.'.format(backend))
