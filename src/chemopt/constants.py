
def conv_factor(from_unit, to_unit):
    conversion = {}
    conversion['Bohr'] = 0.52917721067e-10
    conversion['Angstrom'] = 1e-10
    return conversion[from_unit] / conversion[to_unit]
