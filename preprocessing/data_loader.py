from __future__ import print_function, division, absolute_import

import time
import threading
import multiprocessing as mp
from collections import OrderedDict

import cv2
import numpy as np
import torch as th
from torch.autograd import Variable

from .utils import preprocessInput
from .preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS


def imageWorker(image_queue, output_queue, exit_event):
    """
    Worker that preprocess images
    :param image_queue: (multiprocessing.Queue) queue with the path to the images
    :param output_queue: (multiprocessing.Queue) queue where the preprocessed image
                          will be added
    :param exit_event: (multiprocessing.Event) Event for exiting the loop
    """
    while not exit_event.is_set():
        idx, image_path = image_queue.get()

        if idx is None:
            image_queue.put((None, None))
            break

        im = cv2.imread(image_path)
        # Resize
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        # Normalize
        im = preprocessInput(im.astype(np.float32), mode="image_net")
        output_queue.put((idx, im))
        del im  # Free memory


class BaxterImageLoader(object):
    """
    Data loader for baxter images.
    It uses workers, a prefetch thread and cache the data for efficiency.

    :param minibatchlist: [[int]] list of list of int (observations ids)
    :param images_path: (numpy 1D array of str)
    :param same_actions: [numpy matrix]
    :param dissimilar: [numpy matrix]
    :param test_batch_size: (int)
    :param cache_capacity: (int) number of images that can be cached
    :param n_workers: (int) number of processes used for preprocessing
    :param auto_cleanup: (bool) Whether to clean up preprocessing thread and cache after each epoch
    [WARNING] Set to False, you MUST clean up the loader manually (by calling cleanUp() method)
    """

    def __init__(self, minibatchlist, images_path, same_actions,
                 dissimilar, test_batch_size=512, cache_capacity=5000,
                 n_workers=5, auto_cleanup=True):
        super(BaxterImageLoader, self).__init__()

        self.n_minibatches = len(minibatchlist)
        # Total number of images
        self.n_samples = len(images_path)
        # Copy data to avoid side effects
        # (it uses more memory but prevent from weird bugs)
        self.minibatchlist = np.array(minibatchlist[:])
        # Save minibatches original order
        self.original_minibatchlist = np.array(minibatchlist[:])
        self.images_path = images_path[:]
        self.dissimilar = dissimilar[:]
        self.same_actions = same_actions[:]

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

        if self.n_workers <= 0:
            raise ValueError("n_workers <= 0 in the data loader")

        # Start the workers, there is one input queue (image queue) per worker
        # and a common output_queue
        self.workers = []
        for i in range(self.n_workers):
            w = mp.Process(target=imageWorker, args=(self.image_queues[i], self.output_queue, self.exit_event))
            w.daemon = True  # ensure that the worker exits on process exit
            w.start()
            self.workers.append(w)

    def trainMode(self):
        """
        Switch to train mode and reset the iterator
        It uses the minibatchlist pass at initialization
        """
        self.is_training = True
        self.minibatchlist = self.original_minibatchlist.copy()
        # Reset the iterator
        self.resetIterator()

    def testMode(self):
        """
        Switch to test mode (faster mode) and reset the iterator
        Next observations, same and dissimilar pairs are not computed
        """
        self.is_training = False
        self.minibatchlist = []
        for i in range(self.n_samples // self.test_batch_size + 1):
            start_idx = i * self.test_batch_size
            end_idx = min(self.n_samples, (i + 1) * self.test_batch_size)
            self.minibatchlist.append(np.arange(start_idx, end_idx))
        # Reset the iterator
        self.resetIterator()

    def deleteOldCache(self):
        """
        Delete oldest keys in the cache to save memory
        """
        n_in_cache = len(self.cache.keys())
        n_to_delete = n_in_cache - self.cache_capacity + 1
        if n_to_delete > 0:
            # Delete first n_to_delete elements (oldest entries)
            for key in self.cache.keys()[:n_to_delete]:
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
        np.random.shuffle(self.minibatchlist)

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
            obs = np.zeros((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS), dtype=np.float32)
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
            obs_dict[key] = Variable(th.from_numpy(obs), volatile=not self.is_training)
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
        # (next obs, dissimilar and similar pairs)
        if self.is_training:
            diss = self.dissimilar[i][np.random.permutation(self.dissimilar[i].shape[0])]
            same = self.same_actions[i][np.random.permutation(self.same_actions[i].shape[0])]
            # Convert to torch tensor
            diss, same = th.from_numpy(diss), th.from_numpy(same)

            # Retrieve observations
            # Define a dict to modify it in the for loop
            obs_dict = {'obs': None, 'next_obs': None}
            indices_list = [obs_indices, obs_indices + 1]
        else:
            obs_dict = {'obs': None}
            indices_list = [obs_indices]

        self._sendToWorkers(batch_size, indices_list, obs_dict)

        if self.is_training:
            self.preprocess_result = obs_dict['obs'], obs_dict['next_obs'], diss.clone(), same.clone()
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


class SupervisedDataLoader(BaxterImageLoader):
    def __init__(self, x_indices, y_values, images_path, batch_size, is_training=True,
                 no_targets=False, n_workers=5, cache_capacity=5000, auto_cleanup=True):
        # Create minibatch list
        # Warning: this may create minibatches of different length
        targets = []
        minibatchlist = []
        n_minibatches = len(x_indices) // batch_size + 1
        for i in range(0, n_minibatches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(x_indices))
            excerpt = slice(start_idx, end_idx)
            minibatchlist.append(x_indices[excerpt])
            targets.append(y_values[excerpt])

        # Whether to yield targets together with output
        # (not needed when plotting or predicting states)
        self.no_targets = no_targets
        self.targets = np.array(targets)

        super(SupervisedDataLoader, self).__init__(minibatchlist, images_path, [], [],
                                                   cache_capacity=cache_capacity,
                                                   n_workers=n_workers, auto_cleanup=auto_cleanup)
        # Training mode is the default one
        if not is_training:
            self.testMode()

    def shuffleMinitbatchesOrder(self):
        """
        Shuffle list of minibatches and targets
        """
        indices = np.random.permutation(self.n_minibatches).astype(np.int64)
        self.minibatchlist = self.minibatchlist[indices]
        self.targets = self.targets[indices]

    def _processNextMinibatch(self):
        """
        Send images to workers and compute
        """
        # Alias to improve readability
        i = self.current_preprocessed_idx
        obs_indices = self.minibatchlist[i]
        targets = Variable(th.from_numpy(self.targets[i]))

        batch_size = len(obs_indices)
        obs_dict = {'obs': None}
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
        self.is_training = True

    def testMode(self):
        self.is_training = False
