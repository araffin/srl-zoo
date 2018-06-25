from __future__ import print_function, division, absolute_import

import time
import threading
import multiprocessing as mp
from collections import OrderedDict
import glob
import random

import cv2
import numpy as np
import torch as th

from .utils import preprocessInput
from .preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, getNChannels


def preprocessImage(image):
    """
    :param image: (numpy matrix)
    :return: (numpy matrix)
    """
    # Resize
    im = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    im = preprocessInput(im.astype(np.float32), mode="image_net")
    return im


def imageWorker(image_queue, output_queue, exit_event, multi_view=False, triplets=False):
    """
    Worker that preprocess images
    :param image_queue: (multiprocessing.Queue) queue with the path to the images
    :param output_queue: (multiprocessing.Queue) queue where the preprocessed image
                          will be added
    :param exit_event: (multiprocessing.Event) Event for exiting the loop
    :param multi_view: (bool) enables dual camera mode
    :param triplets: (bool) enables loading of negative example (third image)
    """
    while not exit_event.is_set():
        idx, image_path = image_queue.get()

        if idx is None:
            image_queue.put((None, None))
            break
        # Remove trailing .jpg if present
        image_path = image_path.split('.jpg')[0]
        if multi_view:

            images = []
            for i in range(2):
                im = cv2.imread("{}_{}.jpg".format(image_path, i + 1))
                if im is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(image_path, i + 1))
                images.append(preprocessImage(im))

            ####################
            # loading a negative observation

            if triplets:
                # End of file format for positive & negative observations (camera 1) - length : 6 characters
                extra_chars = '_1.jpg'

                # getting path for all files of same record episode, e.g path_to_data/record_001/frame[0-9]{6}*
                digits_path = glob.glob(image_path[:-6] + '[0-9]*' + extra_chars)

                # getting the current & all frames' timesteps
                current = int(image_path[-6:])
                # For all others extract last 6 digits (timestep) after removing the extra chars
                all_frame_steps = [int(k[:-len(extra_chars)][-6:]) for k in digits_path]
                # removing current positive timestep from the list
                all_frame_steps.remove(current)

                # negative timestep by random sampling
                length_set_steps = len(all_frame_steps)
                negative = all_frame_steps[random.randint(0, length_set_steps - 1)]
                negative_path = '{}{:06d}'.format(image_path[:-6], negative)

                im3 = cv2.imread(negative_path + "_1.jpg")
                if im3 is None:
                    raise ValueError("tried to load {}_{}.jpg, but it was not found".format(negative_path, 1))
                im3 = preprocessImage(im3)
                # stacking along channels
                images.append(im3)

            im = np.dstack(images)

        else:
            im = cv2.imread(image_path + ".jpg")
            if im is None:
                raise ValueError("tried to load {}.jpg, but it was not found".format(image_path))
            im = preprocessImage(im)

        output_queue.put((idx, im))
        del im  # Free memory


class CustomDataLoader(object):
    """
    Data loader for efficiently loading images on the fly.
    It uses workers, a prefetch thread and cache the data for efficiency.

    :param minibatchlist: [[int]] list of list of int (observations ids)
    :param images_path: (numpy 1D array of str)
    :param test_batch_size: (int)
    :param cache_capacity: (int) number of images that can be cached
    :param multi_view: (bool) enables dual camera mode
    :param triplets: (bool) enables loading of negative observation
    :param n_workers: (int) number of processes used for preprocessing
    :param auto_cleanup: (bool) Whether to clean up preprocessing thread and cache after each epoch
    [WARNING] Set to False, you MUST clean up the loader manually (by calling cleanUp() method)
    It may also produce deadlocks
    """

    def __init__(self, minibatchlist, images_path, test_batch_size=512, cache_capacity=5000,
                 n_workers=5, auto_cleanup=True, multi_view=False, triplets=False):
        super(CustomDataLoader, self).__init__()

        self.n_minibatches = len(minibatchlist)
        # Total number of images
        self.n_samples = len(images_path)
        # Copy data to avoid side effects
        # (it uses more memory but prevent from weird bugs)
        self.minibatchlist = np.array(minibatchlist[:])
        # Copy useful array to avoid side effects
        self.images_path = images_path[:]
        # Save minibatches original order
        self.minibatches_indices = np.arange(len(minibatchlist), dtype=np.int64)
        self.original_minibatchlist = self.minibatchlist.copy()

        # Index of the minibatch in the iterator
        self.current_idx = 0
        # Index of the minibatch thta is currently preprocessed
        self.current_preprocessed_idx = 0
        # Whether we are in training/test mode
        self.is_training = True
        self.test_batch_size = test_batch_size
        # Cache
        # It will contains the tensors
        # corresponding to preprocessed images
        # it speeds up things
        self.cache = OrderedDict()
        self.cached_indices = None
        self.cache_capacity = cache_capacity
        # Prefetch
        self.preprocess_result = None
        # Event set when the requested
        # minibatch is ready (= preprocessing finished)
        self.ready_event = threading.Event()
        # Event set when the iterator has return
        # the preprocess_result
        self.result_given_event = threading.Event()
        # Event used to reset the iterator
        # It allows to wait when the iterator has finished
        # one iteration on all the minibatches
        self.reset_event = threading.Event()
        # Event used to terminate preprocessing thread
        self.thread_exit = threading.Event()
        # This variable will contain reference
        # to the preprocessing thread
        self.thread = None
        # Whether to clean up preprocessing thread and cache after each epoch
        self.auto_cleanup = auto_cleanup

        # Multiprocessing
        self.n_workers = n_workers
        # Workers input queue
        self.image_queues = [mp.Queue() for _ in range(self.n_workers)]
        self.output_queue = mp.Queue()
        # Event used to shutdown workers
        self.exit_event = mp.Event()
        # keep track of images that remain to be preprocessed
        self.n_sent, self.n_received = 0, 0
        self.shutdown = False
        self.multi_view = multi_view
        self.triplets = triplets

        if self.n_workers <= 0:
            raise ValueError("n_workers <= 0 in the data loader")

        # Start the workers, there is one input queue (image queue) per worker
        # and a common output_queue
        self.workers = []
        for i in range(self.n_workers):
            w = mp.Process(target=imageWorker, args=(self.image_queues[i], self.output_queue,
                                                     self.exit_event, self.multi_view, self.triplets))
            w.daemon = True  # ensure that the worker exits on process exit
            w.start()
            self.workers.append(w)

    def resetMinibatches(self):
        """
        Restore minibatches to their original order
        """
        self.minibatches_indices = np.arange(len(self.minibatchlist), dtype=np.int64)
        self.minibatchlist = self.original_minibatchlist.copy()

    def trainMode(self):
        """
        Switch to train mode and reset the iterator
        It uses the minibatchlist pass at initialization
        """
        self.is_training = True
        self.resetMinibatches()
        # Reset the iterator
        self.resetIterator()

    def testMode(self):
        """
        Switch to test mode (faster mode) and reset the iterator
        Next observations are not computed
        """
        self.is_training = False
        self.minibatchlist = []
        for i in range(self.n_samples // self.test_batch_size + 1):
            start_idx = i * self.test_batch_size
            end_idx = min(self.n_samples, (i + 1) * self.test_batch_size)
            self.minibatchlist.append(np.arange(start_idx, end_idx))
        # Reset the iterator
        self.resetIterator()

    @staticmethod
    def createMinibatchList(x_indices, y_values, batch_size):
        """
        Create list of minibatches (contains the observations indices)
        along with the corresponding list of targets
        Warning: this may create minibatches of different length
        :param x_indices: (numpy 1D array)
        :param y_values: (numpy tensor)
        :param batch_size: (int)
        :return: [numpy array], [numpy tensor]
        """
        targets = []
        minibatchlist = []
        n_minibatches = len(x_indices) // batch_size + 1
        for i in range(0, n_minibatches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x_indices))
            excerpt = slice(start_idx, end_idx)
            # Remove excerpt with no elements
            if len(x_indices[excerpt]) > 0:
                minibatchlist.append(x_indices[excerpt])
                targets.append(y_values[excerpt])

        return minibatchlist, targets

    def deleteOldCache(self):
        """
        Delete oldest keys in the cache to save memory
        """
        n_in_cache = len(self.cache.keys())
        n_to_delete = n_in_cache - self.cache_capacity + 1
        if n_to_delete > 0:
            # Delete first n_to_delete elements (oldest entries)
            for key in list(self.cache.keys())[:n_to_delete]:
                del self.cache[key]

    def cleanUp(self):
        """
        Exit preprocessing thread
        """
        self.thread_exit.set()
        self.resetIterator()
        self.thread.join()
        self.thread = None

    def resetIterator(self):
        """
        Reset the iterator so we can do
        a full pass on the minibatches
        """
        self.current_idx = 0
        self.resetPreprocessingThread()

    def resetCache(self):
        """
        Delete the current cache (free memory)
        """
        del self.cache
        self.cache = OrderedDict()

    def resetAndShuffle(self):
        """
        Reset the iterator and shuffle the minibatches
        """
        self.shuffleMinitbatchesOrder()
        self.resetIterator()

    def shuffleMinitbatchesOrder(self):
        """
        Shuffle list of minibatches
        """
        self.resetMinibatches()
        indices = np.random.permutation(self.n_minibatches).astype(np.int64)
        self.minibatches_indices = indices
        self.minibatchlist = self.minibatchlist[indices]

    def resetQueues(self):
        """
        Reset sent and received count
        and empty queues
        """
        self.n_sent, self.n_received = 0, 0
        # Clear queues
        for q in self.image_queues + [self.output_queue]:
            while not q.empty():
                idx, _ = q.get()
                if idx is not None:
                    print("Warning, queue not empty")

    def resetPreprocessingThread(self):
        """
        Clean up preprocessing thread
        """
        self.preprocess_result = None
        self.current_preprocessed_idx = 0
        # Reset the different events
        self.ready_event.clear()
        self.result_given_event.clear()
        # Notify the preprocessing thread
        # to start again preprocessing the data
        self.reset_event.set()

    def launchPreprocessing(self):
        """
        Create the thread that feed
        the workers with data
        """
        self.thread_exit.clear()
        t = threading.Thread(target=self.preprocessingThread)
        self.thread = t
        t.deamon = True
        t.start()

    def preprocessingThread(self):
        """
        Preprocess minibatches
        It waits for a reset event at the end of an iteration
        """
        while not self.thread_exit.is_set():
            if self.current_preprocessed_idx >= len(self.minibatchlist):
                self.reset_event.clear()
                self.reset_event.wait()
                if self.thread_exit.is_set():
                    continue
            self._processNextMinibatch()

    def _sendToWorkers(self, batch_size, indices_list, obs_dict):
        """
        Fill workers queues and concatenate result
        in a numpy array
        """
        # Preprocessing loop, it fills workers queues
        for indices, key in zip(indices_list, obs_dict.keys()):

            obs = np.zeros((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, getNChannels()), dtype=np.float32)
            # Reset queues and received count
            self.resetQueues()

            # Retrieve known images from cache:
            self.cached_indices = np.zeros(len(indices)).astype(bool)
            known_images = set(self.cache.keys())
            # Dict to associate minibatch indices to image indices
            minibatch_idx_to_idx = {}
            # Do not preprocessed again cached images
            for j, idx in enumerate(indices):
                minibatch_idx_to_idx[j] = idx
                if idx in known_images:
                    self.cached_indices[j] = True
                    obs[j, :, :, :] = self.cache[idx]

            # Fill the workers queues
            self._putImages(indices)

            # Wait for workers
            while self.n_received < self.n_sent:
                j, im = self.output_queue.get(timeout=3)  # 3s timeout
                obs[j, :, :, :] = im
                # Cache the preprocessed image
                self.cache[minibatch_idx_to_idx[j]] = im
                self.n_received += 1
            # Channel first
            obs = np.transpose(obs, (0, 3, 2, 1))
            obs_dict[key] = th.from_numpy(obs)
            # Free memory
            del obs

        # Delete old elements in the cache
        # to limit its size (regulated by self.cache_capacity)
        self.deleteOldCache()

        if self.preprocess_result is not None:
            # Wait before overwritting result
            self.result_given_event.wait()
            self.result_given_event.clear()
            # Wait a bit to avoid overwritting preprocess_result
            # before it returns data
            # time.sleep(0.01)

    def _processNextMinibatch(self):
        """
        Send images to workers and compute
        """
        # Alias to improve readability
        i = self.current_preprocessed_idx
        obs_indices = self.minibatchlist[i]

        batch_size = len(obs_indices)
        # If we are training we need addional tensors
        # (next obs)
        if self.is_training:
            # Retrieve observations
            # Define a dict to modify it in the for loop
            obs_dict = OrderedDict([('obs', None), ('next_obs', None)])
            indices_list = [obs_indices, obs_indices + 1]
        else:
            obs_dict = OrderedDict([('obs', None)])
            indices_list = [obs_indices]

        self._sendToWorkers(batch_size, indices_list, obs_dict)

        if self.is_training:
            self.preprocess_result = self.minibatches_indices[i], obs_dict['obs'], obs_dict['next_obs']
        else:
            self.preprocess_result = obs_dict['obs']

        # Notify iterator that the minibatch is ready
        self.ready_event.set()
        self.current_preprocessed_idx += 1

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        """
        Called automatically when doing a for loop
        on this object
        """
        if self.thread is None:
            self.launchPreprocessing()

        if self.current_idx < len(self.minibatchlist):
            # Wait for the data to be ready
            self.ready_event.wait()
            # Reset the event
            self.ready_event.clear()
            result = self.preprocess_result
            # Notify preprocess thread that self.preprocess_result
            # can be overwritten
            self.result_given_event.set()
            self.current_idx += 1
            return result
        else:
            # Free memory by resetting preprocess_result
            self.preprocess_result = None
            if self.auto_cleanup:
                self.resetCache()
                # Exit Prefetch thread
                self.cleanUp()
            raise StopIteration

    next = __next__  # Python 2 compatibility

    def _putImages(self, indices):
        """
        Put images to be processed in the queues
        :param indices: [int] List of image indices
        """
        for j, idx in enumerate(indices):
            # If the image is not in the cache
            if not self.cached_indices[j]:
                # Retrieve image full path
                image_path = 'data/{}'.format(self.images_path[idx])
                # Add it to a worker queue
                self.image_queues[j % self.n_workers].put((j, image_path))
                self.n_sent += 1

    def _shutdown_workers(self):
        """
        Method used to shutdown processes
        It set the exit_event and release the queues
        by sending `None`
        """
        if not self.shutdown:
            self.shutdown = True
            self.exit_event.set()
            for i, w in enumerate(self.workers):
                self.image_queues[i].put((None, None))
                w.join()

    def __del__(self):
        """
        Shutdown processes when the object is deleted
        """
        if len(self.workers) > 0:
            self._shutdown_workers()


class SupervisedDataLoader(CustomDataLoader):
    """
    Data loader for baxter images for supervised learning.
    It uses workers, a prefetch thread.
    :param x_indices: (numpy 1D array)
    :param y_values: (numpy tensor)
    :param images_path: (numpy 1D array of str)
    :param batch_size: (int)
    :param is_training: (bool) Whether to create tensor that keep track of the gradient or not
    :param no_targets: (bool) Set to true, only inputs are generated
    :param n_workers: (int) number of processes used for preprocessing
    :param auto_cleanup: (bool) Whether to clean up preprocessing thread and cache after each epoch
    [WARNING] Set to False, you MUST clean up the loader manually (by calling cleanUp() method)
    """

    def __init__(self, x_indices, y_values, images_path, batch_size, is_training=True,
                 no_targets=False, n_workers=5, auto_cleanup=True):
        # Create minibatch list
        minibatchlist, targets = self.createMinibatchList(x_indices, y_values, batch_size)

        # Whether to yield targets together with output
        # (not needed when plotting or predicting states)
        self.no_targets = no_targets
        self.targets = np.array(targets)
        self.original_targets = self.targets.copy()

        # Here the cache is not useful: we do not have observations
        # that are present in different minibatches
        super(SupervisedDataLoader, self).__init__(minibatchlist, images_path, cache_capacity=0,
                                                   n_workers=n_workers, auto_cleanup=auto_cleanup)
        # Training mode is the default one
        if not is_training:
            self.testMode()

    def resetMinibatches(self):
        """
        Restore minibatches to their original order
        """
        self.minibatchlist = self.original_minibatchlist.copy()
        self.targets = self.original_targets.copy()

    def shuffleMinitbatchesOrder(self):
        """
        Shuffle list of minibatches and targets
        """
        self.resetMinibatches()
        indices = np.random.permutation(self.n_minibatches).astype(np.int64)
        self.minibatchlist = self.minibatchlist[indices]
        self.targets = self.targets[indices]

    def _processNextMinibatch(self):
        """
        Send images to workers and compute inputs/targets
        """
        # Alias to improve readability
        i = self.current_preprocessed_idx
        obs_indices = self.minibatchlist[i]
        targets = th.from_numpy(self.targets[i]).requires_grad_(self.is_training)

        batch_size = len(obs_indices)
        obs_dict = OrderedDict([('obs', None)])
        indices_list = [obs_indices]

        self._sendToWorkers(batch_size, indices_list, obs_dict)

        if self.no_targets:
            self.preprocess_result = obs_dict['obs']
        else:
            self.preprocess_result = obs_dict['obs'], targets.clone()

        # Notify iterator that the minibatch is ready
        self.ready_event.set()
        self.current_preprocessed_idx += 1

    def trainMode(self):
        """
        Switch to train mode
        """
        self.is_training = True

    def testMode(self):
        """
        Switch to test mode
        Tensors will not keep track of the gradient
        """
        self.is_training = False


class AutoEncoderDataLoader(CustomDataLoader):
    """
    Data loader for baxter images for autoencoder.
    It uses workers, a prefetch thread.
    :param x_indices: (numpy 1D array)
    :param images_path: (numpy 1D array of str)
    :param batch_size: (int)
    :param noise_factor: (float)
    :param is_training: (bool) Whether to create tensor that keep track of the gradient or not
    :param no_targets: (bool) Set to true, only inputs are generated
    :param n_workers: (int) number of processes used for preprocessing
    :param auto_cleanup: (bool) Whether to clean up preprocessing thread and cache after each epoch
    [WARNING] Set to False, you MUST clean up the loader manually (by calling cleanUp() method)
    """

    def __init__(self, x_indices, images_path, batch_size, noise_factor=0.0, is_training=True,
                 no_targets=False, n_workers=5, auto_cleanup=True, multi_view=False):
        # Create minibatch list
        minibatchlist, _ = self.createMinibatchList(x_indices, x_indices, batch_size)

        # Whether to yield targets together with output
        # (not needed when plotting or predicting states)
        self.no_targets = no_targets
        self.noise_factor = noise_factor
        self.multi_view = multi_view

        # Here the cache is not useful: we do not have observations
        # that are present in different minibatches
        super(AutoEncoderDataLoader, self).__init__(minibatchlist, images_path, cache_capacity=0, n_workers=n_workers,
                                                    auto_cleanup=auto_cleanup, multi_view=multi_view)
        # Training mode is the default one
        if not is_training:
            self.testMode()

    def resetMinibatches(self):
        """
        Restore minibatches to their original order
        """
        self.minibatchlist = self.original_minibatchlist.copy()

    def shuffleMinitbatchesOrder(self):
        """
        Shuffle list of minibatches
        """
        self.resetMinibatches()
        indices = np.random.permutation(self.n_minibatches).astype(np.int64)
        self.minibatchlist = self.minibatchlist[indices]

    def _processNextMinibatch(self):
        """
        Send images to workers and compute inputs/targets
        """
        # Alias to improve readability
        i = self.current_preprocessed_idx
        obs_indices = self.minibatchlist[i]
        # Warning: the noise is not consistent
        # for validation set (different at each iteration)
        if self.noise_factor > 0:
            noise_shape = (len(obs_indices), getNChannels(), IMAGE_HEIGHT, IMAGE_WIDTH)
            noise = self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=noise_shape).astype(np.float32)
            noise = th.from_numpy(noise).requires_grad_(self.is_training)

        batch_size = len(obs_indices)
        obs_dict = OrderedDict([('obs', None)])
        indices_list = [obs_indices]

        self._sendToWorkers(batch_size, indices_list, obs_dict)

        if self.no_targets:
            self.preprocess_result = obs_dict['obs']
        else:
            if self.noise_factor > 0:
                self.preprocess_result = obs_dict['obs'] + noise, obs_dict['obs'].clone()
            else:
                self.preprocess_result = obs_dict['obs'], obs_dict['obs'].clone()

        # Notify iterator that the minibatch is ready
        self.ready_event.set()
        self.current_preprocessed_idx += 1

    def trainMode(self):
        """
        Switch to train mode
        """
        self.is_training = True

    def testMode(self):
        """
        Switch to test mode
        Tensors will not keep track of the gradient
        """
        self.is_training = False
