.. _multi_cam:

Multiple Cameras
----------------

Stacked Observations
~~~~~~~~~~~~~~~~~~~~

Using the ``custom_cnn`` and ``mlp`` architecture, it is possible to
pass pairs of images from different views stacked along the channels'
dimension i.e of dim (224,224,6).

To use this functionality to perform state representation learning,
enable ``--multi-view`` (see usage of script train.py), and use a
dataset generated for the purpose.

Triplets of Observations
~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, it is possible to learn representation of states using a
dataset of triplets, i.e tuples made of an anchor, a positive and a
negative observation.

The anchor and the positive observation are views of the scene at the
same time step, but from different cameras.

The negative example is an image from the same camera as the anchor but
at a different time step selected randomly among images in the same
record.

In our case, enable ``triplet`` as a loss (``--losses``) to use the
TCN-like architecture made of a pre-trained ResNet with an extra fully
connected layer (embedding).

To use this functionality also enable ``--multi-view``, and use a
dataset generated for the purpose. Related papers:

-  "Time-Contrastive Networks: Self-Supervised Learning from Video" (P.
   Sermanet et al., 2017), paper:
   `https://arxiv.org/abs/1704.06888 <https://arxiv.org/abs/1704.06888>`__
