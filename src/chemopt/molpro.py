def generate_molpro_input(geometry, theory='rhf', charge=0, multiplicity=1, title='', basis='vdz',
                          forces=False):
    output = ("""*** {title}

gprint, basis
gprint, orbital

basis, {basis}

geometry={{
{geometry}
}}

{theory}
{forces}
----""").format
# {{{theory}, wf, {num_electrons}, 1, {multiplicity}}}
    
    atomic_number = constants.elements['atomic_number'].to_dict()
    num_electrons = sum([atomic_number[x] for x in molecule['atom']]) - charge
    
    return output(title=title, basis=basis, geometry=geometry,
                  theory=theory, multiplicity=multiplicity - 1, num_electrons=num_electrons,
                  forces='forces' if forces else ''
                 )
