.. _srl:

Learning a State Representation
-------------------------------

To learn a state representation, you need to enforce constrains on the
representation using one or more losses. For example, to train an
autoencoder, you need to use a reconstruction loss. Most losses are not
exclusive, that means you can combine them.

All losses are defined in ``losses/losses.py``. The available losses
are:

-  autoencoder: reconstruction loss, using current and next observation
-  denoising autoencoder (dae): same as for the auto-encoder, except
   that the model reconstruct inputs from noisy observations containing
   a random zero-pixel mask
-  vae: (beta)-VAE loss (reconstruction + kullback leiber divergence
   loss)
-  inverse: predict the action given current and next state
-  forward: predict the next state given current state and taken action
-  reward: predict the reward (positive or not) given current and next
   state
-  priors: robotic priors losses (see "Learning State Representations
   with Robotic Priors")
-  triplet: triplet loss for multi-cam setting (see *Multiple Cameras*
   section)

**[Experimental]**

-  reward-prior: Maximises the correlation between states and rewards
   (does not make sense for sparse reward)
-  episode-prior: Learn an episode-agnostic state space, thanks to a
   discriminator distinguishing states from same/different episodes
-  perceptual similarity loss (for VAE): Instead of the reconstruction
   loss in the beta-VAE loss, it uses the distance between the
   reconstructed input and real input in the embedding of a pre-trained
   DAE.
-  mutual information loss: Maximises the mutual information between
   states and rewards

All possible arguments can be display using ``python train.py --help``.
You can limit the training set size (``--training-set-size`` argument),
change the minibatch size (``-bs``), number of epochs (``--epochs``),
...

Examples
~~~~~~~~

Train an inverse model:

::

   python train.py --data-folder data/path/to/dataset --losses inverse

Train an autoencoder:

::

   python train.py --data-folder data/path/to/dataset --losses autoencoder

Combining an autoencoder with an inverse model is as easy as:

::

   python train.py --data-folder data/path/to/dataset --losses autoencoder inverse

You can as well specify the weight of each loss:

::

   python train.py --data-folder data/path/to/dataset --losses autoencoder:1 inverse:10

Train a vae with the perceptual similarity loss:

::

   python train.py --data-folder data/path/to/dataset --losses vae perceptual --path-to-dae logs/path/to/pretrained_dae/srl_model.pth --state-dim-dae ST_DIM_DAE

.. _stacking/splitting-models-instead-of-combining-them:

Stacking/Splitting Models Instead of Combining Them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because losses do not optimize the same objective and can be opposed, it
may make sense to stack representations learned with different
objectives, instead of combining them. For instance, you can stack an
autoencoder (with a state dimension of 20) with an inverse model (of
dimension 2) using the previous weights:

::

   python train.py --data-folder data/path/to/dataset --losses autoencoder:1:20 inverse:10:2 --state-dim 22

The details of how models are splitted can be found inside the
``SRLModulesSplit`` class, defined in ``models/modules.py``. All models
share the same *encoder* or *features extractor*, that maps observations
to states.

Addtional example: split and combine losses.
Reward loss on 50 dimensions and forward + inverse losses on 2 dimensions
(note the `-1` that specify that losses are applied on the same split):

::

  python train.py --data-folder data/path/to/dataset --losses reward:1:50 inverse:1:2 forward:1:-1 --state-dim 52


Predicting States on the Whole Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you trained your model on a subset of a dataset, you can predict
states for the whole dataset (or on a subset) using:

::

   python -m evaluation.predict_dataset --log-dir logs/path/to/log_folder/

use ``-n 1000`` to predict on the first 1000 samples only.

Predicting Reward Using a Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to predict the reward (train a classifier for positive or
null reward) using ground truth states or learned states, you can use
``evaluation/predict_reward.py`` script. Ground Truth:

::

   python -m evaluation.predict_reward --data-folder data/dataset_name/ --training-set-size 50000

On Learned States:

::

   python -m evaluation.predict_reward --data-folder data/dataset_name/ -i log/path/to/states_rewards.npz
