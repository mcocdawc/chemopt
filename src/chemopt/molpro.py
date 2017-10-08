

def generate_molpro_input(geometry, theory='rhf', charge=0, multiplicity=1,
                          title='', basis='vdz',
                          forces=True):
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
                  theory=theory, forces='forces' if forces else ''
                  )
