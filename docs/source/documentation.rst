Documentation
==================

Optimiser
+++++++++++++

In non-redundant internal coordinates (Zmatrix)
------------------------------------------------

.. currentmodule:: chemopt.zmat_optimisation

.. autosummary::
    :toctree: src_zmat_optimiser

    optimise
    get_next_step


Interfaces for electronic structure calculation
+++++++++++++++++++++++++++++++++++++++++++++++++

Generic Interface
------------------


.. currentmodule:: chemopt.interface.generic

.. autosummary::
    :toctree: src_interface_generic

    calculate


Molpro Interface
------------------

.. currentmodule:: chemopt.interface.molpro

.. autosummary::
    :toctree: src_interface_molpro

    calculate
    generate_input_file
    parse_output
