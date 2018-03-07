# -*- coding: utf-8 -*-
try:
    import configparser
except ImportError:
    # Due to PY27 compatibility
    import ConfigParser as configparser
import os
from warnings import warn

from chemopt.utilities._decorators import Substitution

values = {}
values['hamiltonian'] = {'RHF', 'MP2', 'B3LYP', 'CCSD', 'CCSD(T)'}
values['backend'] = {'molpro', 'molcas'}

fixed_defaults = {}
fixed_defaults['charge'] = 0
fixed_defaults['multiplicity'] = 1
fixed_defaults['forces'] = False
fixed_defaults['wfn_symmetry'] = 1
fixed_defaults['title'] = ''
fixed_defaults['etol'] = 1e-6
fixed_defaults['gtol'] = 6e-4
fixed_defaults['max_iter'] = 100
fixed_defaults['coord_fmt'] = '.4f'


def _give_default_file_path():
    HOME = os.path.expanduser('~')
    filepath = os.path.join(HOME, '.chemoptrc')
    return filepath


def provide_defaults():
    settings = {}
    settings['defaults'] = {}
    settings['defaults']['backend'] = 'molcas'
    settings['defaults']['num_procs'] = 1
    settings['defaults']['num_threads'] = 1
    settings['defaults']['mem_per_proc'] = '150Mb'
    settings['defaults']['molpro_exe'] = 'molpro'
    settings['defaults']['molcas_exe'] = 'molcas'
    return settings


def write_configuration_file(filepath=_give_default_file_path(),
                             overwrite=False):
    """Create a configuration file.

    Writes the current state of defaults into a configuration file.

    .. note:: Since a file is permamently written, this function
        is strictly speaking not sideeffect free.

    Args:
        filepath (str): Where to write the file.
            The default is under both UNIX and Windows ``~/.chemoptrc``.
        overwrite (bool):

    Returns:
        None:
    """
    config = configparser.ConfigParser()
    config.read_dict(settings)

    if os.path.isfile(filepath) and not overwrite:
        try:
            raise FileExistsError
        except NameError:  # because of python2
            warn('File exists already and overwrite is False (default).')
    else:
        with open(filepath, 'w') as configfile:
            config.write(configfile)


def read_configuration_file(settings, filepath=_give_default_file_path()):
    """Read the configuration file.

    .. note:: This function changes ``cc.defaults`` inplace and is
        therefore not sideeffect free.

    Args:
        filepath (str): Where to read the file.
            The default is under both UNIX and Windows ``~/.chemoptrc``.

    Returns:
        None:
    """
    config = configparser.ConfigParser()
    config.read(filepath)

    def get_correct_type(section, key, config):
        """Gives e.g. the boolean True for the string 'True'"""
        def getstring(section, key, config):
            return config[section][key]

        def getinteger(section, key, config):  # pylint:disable=unused-variable
            return config[section].getint(key)

        def getboolean(section, key, config):
            return config[section].getboolean(key)

        def getfloat(section, key, config):  # pylint:disable=unused-variable
            return config[section].getfloat(key)
        special_actions = {}  # Something different than a string is expected
        special_actions['defaults'] = {}
        special_actions['defaults']['num_procs'] = getinteger
        special_actions['defaults']['num_threads'] = getinteger
        try:
            return special_actions[section][key](section, key, config)
        except KeyError:
            return getstring(section, key, config)

    for section in config.sections():
        for k in config[section]:
            settings[section][k] = get_correct_type(section, k, config)
    return settings


settings = provide_defaults()
read_configuration_file(settings)
conf_defaults = settings['defaults']


def get_docstr(key, defaults):
    return "The default is '{}'. The allowed values are {}".format(
        defaults[key], values[key])


docstring = {}

docstring['hamiltonian'] = "The hamiltonian to use for calculating the \
electronic energy. The allowed values are {}.\n".format(values['hamiltonian'])

docstring['basis'] = "The basis set to use for calculating \
the electronic energy."

docstring['multiplicity'] = "The spin multiplicity. \
The default is {}.\n".format(fixed_defaults['multiplicity'])

docstring['charge'] = "The overall charge of the molecule. \
The default is {}.\n".format(fixed_defaults['charge'])

docstring['forces'] = "Specify if energy gradients should be calculated. \
The default is {}.".format(fixed_defaults['forces'])

docstring['el_calc_input'] = "Specify the input filename for \
electronic calculations. \
If it is None, the filename of the calling python script is used \
(With the suffix ``.inp`` instead of ``.py``). \
The output will be ``os.path.splitext(inputfile)[0] + '.inp'``.\n"

docstring['md_out'] = "Specify the output filename for \
chemopt output files. \
If it is None, the filename of the calling python script is used \
(With the suffix ``.md`` instead of ``.py``). \
The output will be ``os.path.splitext(inputfile)[0] + '.md'``.\n"

docstring['molden_out'] = "Specify the output filename for \
the molden file from a geometry optimisation. \
If it is None, the filename of the calling python script is used \
(With the suffix ``.molden`` instead of ``.py``). \
The output will be ``os.path.splitext(inputfile)[0] + '.molden'``.\n"

docstring['backend'] = "Specify which QM program suite shoud be used. \
Allowed values are {}, \
the default is '{}'.\n".format(values['backend'], conf_defaults['backend'])

docstring['molpro_exe'] = "Specify the command to invoke molpro. \
The default is '{}'.\n".format(conf_defaults['molpro_exe'])

docstring['molcas_exe'] = "Specify the command to invoke molcas. \
The default is '{}'.\n".format(conf_defaults['molcas_exe'])

docstring['title'] = "The title to be printed in input and output.\n"

docstring['start_orb'] = "Path to an orbital file, \
if starting orbitals should be used."

docstring['wfn_symmetry'] = "The symmetry of the wavefunction specified \
with the molpro \
`notation <https://www.molpro.net/info/2015.1/doc/manual/node36.html>`_.\n"

docstring['etol'] = "Convergence criterium for the energy."

docstring['gtol'] = "Convergence criterium for the gradient."

docstring['max_iter'] = "Maximum number of iterations. The default is \
'{}'.".format(fixed_defaults['max_iter'])

docstring['num_procs'] = "The number of processes to spawn."

docstring['num_threads'] = "Currently not Implemented"

docstring['mem_per_proc'] = "Memory per process. \
This is a string with a number and a unit like '800 Mb'. \
SI and binary prefixes are supported. \
Uses the  `datasize library <https://pypi.python.org/pypi/datasize>`_ \
for parsing."

docstring['coord_fmt'] = "A string as float formatter for the coordinates \
in the output file of chemopt. \
The default is '{}'".format(fixed_defaults['coord_fmt'])

substitute_docstr = Substitution(**docstring)
