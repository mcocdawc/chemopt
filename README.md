========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
        | |landscape|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/chemopt/badge/?style=flat
    :target: https://readthedocs.org/projects/chemopt
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/mcocdawc/chemopt.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/mcocdawc/chemopt

.. |codecov| image:: https://codecov.io/github/mcocdawc/chemopt/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/mcocdawc/chemopt

.. |landscape| image:: https://landscape.io/github/mcocdawc/chemopt/master/landscape.svg?style=flat
    :target: https://landscape.io/github/mcocdawc/chemopt/master
    :alt: Code Quality Status

.. |version| image:: https://img.shields.io/pypi/v/chemopt.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/chemopt

.. |commits-since| image:: https://img.shields.io/github/commits-since/mcocdawc/chemopt/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/mcocdawc/chemopt/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/chemopt.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/chemopt

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/chemopt.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/chemopt

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/chemopt.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/chemopt


.. end-badges

A package for geometry optimization using non redundant internal coordinates (Zmatrices) and symbolic algebra.

* Free software: LGPLv3

Installation
============

::

    pip install chemopt

Documentation
=============

https://chemopt.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
