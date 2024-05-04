import numpy as np
import re
import itertools
from collections import Counter


def add_noise(x, embedding_dim, random_type=None, weight=0.0):
    seq_length = len(x[0])
    batch_size = len(x)
    noise = np.ones([batch_size, seq_length, embedding_dim])

    for bi in range(batch_size):
        if random_type == 'Gaussian':
            noise[bi, :, :] = np.random.normal(1, weight, [seq_length, embedding_dim])

    return noise