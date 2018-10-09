.. _baselines:

SRL Baselines
-------------

SRL Baseline models are saved in ``logs/nameOfTheDataset/baselines/``
folder.

Supervised Learning
~~~~~~~~~~~~~~~~~~~

Example:

::

   python -m baselines.supervised --data-folder path/to/data/folder

Principal Components Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PCA:

::

   python -m baselines.pca --data-folder path/to/data/folder --state-dim 3
