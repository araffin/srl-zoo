from pipeline import NO_PAIRS_ERROR
from utils import printRed

import torch as th
import numpy as np

def overSampling(batch_size, m_list, pairs, function_on_pairs, actions, rewards):
    """
    Look for minibatches missing pairs of observations with the similar/dissimilar rewards (see params)
    Sample for each of those minibatches an observation from another batch that satisfies the
    similarity/dissimilarity with the 1rst observation.
    return the new pairs & the modified minibatch list
    :param batch_size: (int)
    :param m_list: (list) mini-batch list
    :param pairs: similar / dissimilar pairs
    :param function_on_pairs: (function) findDissimilar applied to pairs
    :param actions: (np.ndarray)
    :param rewards: (np.ndarray)
    :return: (list, list) pairs, mini-batch list modified
    """
    # For a each minibatch_id
    if function_on_pairs.__name__ == "findDissimilar":
        pair_name = 'dissimilar pairs'
    else:
        pair_name = 'Unknown pairs'
    counter = 0
    for minibatch_id, d in enumerate(pairs):
        #print("what is it ?: ", minibatch_id, d)
        do = True
        if len(d) == 0:
            counter += 1
        # Do if it contains no similar pairs of samples
        while do and len(d) == 0:
            # for every minibatch & obs of a mini-batch list
            for m_id, minibatch in enumerate(m_list):
                for i in range(batch_size):
                    # Look for similar samples j in other minibatches m_id
                    for j in function_on_pairs(i, m_list[minibatch_id], minibatch, actions, rewards):
                        # Copy samples - done once
                        if (j != i) & (minibatch_id != m_id) and do:
                            m_list[minibatch_id][j] = minibatch[j]
                            pairs[minibatch_id] = np.array([[i, j]])
                            do = False
    print('Dealt with {} minibatches - {}'.format(counter, pair_name))
    return pairs, m_list


def findDissimilar(index, minibatch1, minibatch2, actions, rewards):
    """
    check which samples should be dissimilar
    because they lead to different rewards after the same actions
    :param index: (int)
    :param minibatch1: (np.ndarray)
    :param minibatch2: (np.ndarray)
    :param actions: (np.ndarray)
    :param rewards: (np.ndarray)
    :return: (dict, np.ndarray)
    """
    return np.where((actions[minibatch2] == actions[minibatch1[index]]) *
                    (rewards[minibatch2 + 1] != rewards[minibatch1[index] + 1]))[0]


def findSameActions(index, minibatch, actions):
    """
    Get observations indices where the same action was performed
    as in a reference observation
    :param index: (int)
    :param minibatch: (np.ndarray)
    :param actions: (np.ndarray)
    :return: (np.ndarray)
    """
    return np.where(actions[minibatch] == actions[minibatch[index]])[0]


def findPriorsPairs(batch_size, minibatchlist, actions, rewards, n_actions, n_pairs_per_action):
    """

    :param batch_size: (int)
    :param minibatchlist: ([[int]])
    :param actions: (np.ndarray)
    :param rewards: (np.ndarray)
    :param n_actions: (int)
    :param n_pairs_per_action: ([int])
    :return: ([np.ndarray], [np.ndarray])
    """
    dissimilar_pairs = [
        np.array(
            [[i, j] for i in range(batch_size) for j in findDissimilar(i, minibatch, minibatch, actions, rewards) if
             j > i],
            dtype='int64') for minibatch in minibatchlist]

    # sampling relevant pairs to have at least a pair of dissimilar obs in every minibatches
    dissimilar_pairs, minibatchlist = overSampling(batch_size, minibatchlist, dissimilar_pairs,
                                                   findDissimilar, actions, rewards)
    # same_actions: list of arrays, each containing one pair of observation ids
    same_actions_pairs = [
        np.array([[i, j] for i in range(batch_size) for j in findSameActions(i, minibatch, actions) if j > i],
                 dtype='int64') for minibatch in minibatchlist]

    for pair, minibatch in zip(same_actions_pairs, minibatchlist):
        for i in range(n_actions):
            n_pairs_per_action[i] += np.sum(actions[minibatch[pair[:, 0]]] == i)

    # Stats about pairs
    print("Number of pairs per action:")
    print(n_pairs_per_action)
    print("Pairs of {} unique actions".format(np.sum(n_pairs_per_action > 0)))

    for item in same_actions_pairs + dissimilar_pairs:
        if len(item) == 0:
            msg = "No same actions or dissimilar pairs found for at least one minibatch (currently is {})\n".format(
                batch_size)
            msg += "=> Consider increasing the batch_size or changing the seed"
            printRed(msg)
            sys.exit(NO_PAIRS_ERROR)
    return dissimilar_pairs, same_actions_pairs


# From https://github.com/pytorch/pytorch/pull/4411
def correlationMatrix(mat, eps=1e-8):
    """
    Returns Correlation matrix for mat. It is the equivalent of numpy np.corrcoef

    :param mat: (th.Tensor) Shape: (N, D)
    :param esp: (float) Small value to avoid division by zero.
    :return: (th.Tensor) The correlation matrix Shape: (N, N)
    """
    assert mat.dim() == 2, "Input must be a 2D matrix."
    mat_bar = mat - mat.mean(1).repeat(mat.size(1)).view(mat.size(1), -1).t()
    cov_matrix = mat_bar.mm(mat_bar.t()).div(mat_bar.size(1) - 1)
    inv_stddev = th.rsqrt(th.diag(cov_matrix) + eps)
    cor_matrix = cov_matrix.mul(inv_stddev.expand_as(cov_matrix))
    cor_matrix.mul_(inv_stddev.expand_as(cov_matrix).t())
    return cor_matrix.clamp(-1.0, 1.0)
