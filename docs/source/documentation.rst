Documentation
==================

Optimiser
+++++++++++++

In non-redundant internal coordinates (Zmatrix)
------------------------------------------------

.. currentmodule:: chemopt

.. autosummary::
    :toctree: src_zmat_optimiser

    ~zmat_optimisation.optimise


Interfaces for electronic structure calculation
+++++++++++++++++++++++++++++++++++++++++++++++++

Generic Interface
------------------


.. currentmodule:: chemopt.interface.generic

.. autosummary::
    :toctree: src_interface_generic

    calculate

Molcas Interface
------------------

.. currentmodule:: chemopt.interface.molcas

.. autosummary::
    :toctree: src_interface_molcas

    calculate
    generate_input_file
    parse_output

Molpro Interface
------------------

.. currentmodule:: chemopt.interface.molpro

.. autosummary::
    :toctree: src_interface_molpro

    calculate
    generate_input_file
    parse_output
