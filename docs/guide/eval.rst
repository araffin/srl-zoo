.. _eval:

Evaluation and Plotting
-----------------------

Learned Space Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To view the learned state and play with the latent space of a trained
model, you may use:

.. code:: bash

   python -m enjoy.enjoy_latent --log-dir logs/nameOfTheDataset/nameOfTheModel

Create a Report
~~~~~~~~~~~~~~~

A report contains knn-mse, ground-truth correlation (GTC and GTC_mean) and information about each model present in a log folder.

Create a csv report file using:

::

   python -m evaluation.gather_results -i logs/nameOfTheDataset/


Plot a Learned Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  usage: representation_plot.py [-h] [-i INPUT_FILE] [--data-folder DATA_FOLDER]
                                [--color-episode] [--plot-against]
                                [--pretty-plot-against] [--correlation]
                                [--projection] [--print-corr]

  Plotting script for representation

  optional arguments:
    -h, --help            show this help message and exit
    -i INPUT_FILE, --input-file INPUT_FILE
                          Path to a npz file containing states and rewards
    --data-folder DATA_FOLDER
                          Path to a dataset folder, it will plot ground truth
                          states
    --color-episode       Color states per episodes instead of reward
    --plot-against        Plot against each dimension
    --pretty-plot-against
                          Plot against each dimension (diagonals are
                          distributions + cleaner look)
    --correlation         Plot correlation coeff against each dimension
    --projection          Plot 1D projection of predicted state on ground truth
    --print-corr          Only print correlation measurements


You can plot a learned representation with:

::

   python -m plotting.representation_plot -i path/to/states_rewards.npz

You can also plot ground truth states with:

::

   python -m plotting.representation_plot --data-folder path/to/datasetFolder/

To have a different color per episode, you have to pass
``--data-folder`` argument along with ``--color-episode``.

Plotting each dimension of the state representation against another:

::

   python -m plotting.representation_plot -i path/to/states_rewards.npz --plot-against

**[Evaluation plot]** Plotting the matrix of correlation with the ground
truth states:

::

   python -m plotting.representation_plot -i path/to/states_rewards.npz --data-folder path/to/datasetFolder/ --correlation

**[Ground Truth Correlation]** It measures the maximum correlation (in absolute value) in the learned representation
for each dimension of the ground truth states. GTC_mean is the mean of the GTC vector. :

::

   python -m plotting.representation_plot -i path/to/states_rewards.npz --data-folder path/to/datasetFolder/ --correlation --print-corr

Interactive Plot
~~~~~~~~~~~~~~~~

You can have an interactive plot of a learned representation using:

::

   python -m plotting.interactive_plot --data-folder path/to/datasetFolder/ -i path/to/states_rewards.npz

When you click on a state in the representation plot (left click for 2D,
**right click for 3D plots**!), it shows the corresponding image along
with the reward and the coordinates in the space.

Pass ``--multi-view`` as argument to visualize in case of multiple
cameras.

You can also plot ground truth states when you don't specify a npz file:

::

   python -m plotting.interactive_plot --data-folder path/to/datasetFolder/

Create a KNN Plot and Compute KNN-MSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:

::

   python evaluation/knn_images.py [-h] --log-folder LOG_FOLDER [--seed SEED]
                        [-k N_NEIGHBORS] [-n N_SAMPLES] [--n-to-plot N_TO_PLOT]
                        [--relative-pos] [--ground-truth] [--multi-view]

   KNN plot and KNN MSE

   optional arguments:
     -h, --help            show this help message and exit
     --log-folder LOG_FOLDER
                           Path to a log folder
     --seed SEED           random seed (default: 1)
     -k N_NEIGHBORS, --n-neighbors N_NEIGHBORS
                           Number of nearest neighbors (default: 5)
     -n N_SAMPLES, --n-samples N_SAMPLES
                           Number of test samples (default: 5)
     --n-to-plot N_TO_PLOT
                           Number of samples to plot (default: 5)
     --relative-pos        Use relative position as ground_truth
     --ground-truth        Compute KNN-MSE for ground truth
     --multi-view          To deal with multi view data format


Example:

::

   python evaluation/knn_images.py --log-folder path/to/an/experiment/log/folder
