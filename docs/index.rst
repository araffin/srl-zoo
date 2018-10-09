.. SRL Zoo documentation master file, created by
   sphinx-quickstart on Tue Oct  9 11:18:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to State Representation Learning Zoo's documentation!
=============================================================

A collection of State Representation Learning (SRL) methods for
Reinforcement Learning, written using PyTorch.

Github repo: https://github.com/araffin/srl-zoo

.. note::

	This repo is part of the `S-RL Toolbox <https://s-rl-toolbox.readthedocs.io>`_


Available methods:

-  SRL with Robotic Priors + extensions (stereovision, additional
   priors)
-  Denoising Autoencoder (DAE)
-  Variational Autoencoder (VAE) and beta-VAE
-  PCA
-  Supervised Learning
-  Forward, Inverse Models
-  Triplet Network (for stereovision only)
-  Reward loss
-  Combination and stacking of methods
-  Random Features
-  **[experimental]** Reward Prior, Episode-prior, Perceptual Similarity
   loss (DARLA), Mutual Information loss

Related papers:

-  "S-RL Toolbox: Environments, Datasets and Evaluation Metrics for
   State Representation Learning" (Raffin et al., 2018)
   `https://arxiv.org/abs/1809.09369 <https://arxiv.org/abs/1809.09369>`__
-  "State Representation Learning for Control: An Overview" (Lesort et
   al., 2018), link:
   `https://arxiv.org/pdf/1802.04181.pdf <https://arxiv.org/pdf/1802.04181.pdf>`__
-  "Learning State Representations with Robotic Priors" (Jonschkowski
   and Brock, 2015), link:
   `http://tinyurl.com/gly9sma <http://tinyurl.com/gly9sma>`__


.. toctree::
   :maxdepth: 2
   :caption: Guide

   guide/install
   guide/srl
   guide/config
   guide/baselines
   guide/multi_cam
   guide/eval
   guide/tests
   changelog




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
