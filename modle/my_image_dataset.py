# https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


class image_dataset(object):

    def __init__(self, images, labels, fake_data=False, one_hot=False):

        if fake_data:
            self._num_examples = 1000
            self.one_hot = one_hot

        else:
            assert images.shape[0] == labels.shape[
                0], ('images.shape:%s labels.shape:%s' % (
                    images.shape, labels.shape))
            self._num_examples = images.shape[0]

            images = images.astype(np.float32)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epochs
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end]
