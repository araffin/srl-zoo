.. _install:

Installation
------------

Recommended configuration: Ubuntu 16.04 with python >=3.5 (or python
2.7)

Using Anaconda
~~~~~~~~~~~~~~

Python 3
^^^^^^^^

Please use ``environment.yml`` file from
`https://github.com/araffin/robotics-rl-srl <https://github.com/araffin/robotics-rl-srl>`__
To create a conda environment from this file:

::

   conda env create -f environment.yml

Python 2
^^^^^^^^

Create the new environment ``srl`` from ``environment.yml`` file:

::

   conda env create -f environment.yml

Then activate it using:

::

   source activate srl

Using Docker
~~~~~~~~~~~~

We provide docker images to work with our repository, please read
*Installation using docker* from
`https://github.com/araffin/robotics-rl-srl <https://github.com/araffin/robotics-rl-srl>`__
for more information.
