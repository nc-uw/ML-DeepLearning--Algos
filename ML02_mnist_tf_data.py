"""The data reader for MNIST.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import numpy as np
import pandas as pd

def data_reader(filename, num_pixel=784):
    """Reads in the MNIST data.

    Args:
        filename: String for the filename in CSV format.
        num_pixel: Integer for the number of pixel columns.

    Returns:
        images: Numpy array of pixel values.
        labels: Numpy array for digit class labels.
    """
    dataframe = pd.read_csv(filename, sep=',')
    labels = dataframe.ix[:, 'label'].values
    # Only keeps images for digit 3 or 5.
    # TODO: Update this for multi-class classifcation.
    #to_keep = ((labels == 3) | (labels == 5))
    #sub_dataframe = dataframe[to_keep]
    sub_dataframe = dataframe
    #labels = (sub_dataframe.ix[:, 'label'] == 3).values
    pixel_cols = ['pixel{0}'.format(i) for i in xrange(num_pixel)]
    images = sub_dataframe.ix[:, pixel_cols].values
    return images, labels


class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape %s labels.shape %s' % (images.shape, labels.shape))

        self.num_samples = images.shape[0]
        # TODO: Converts the image value from [0, 255] to [0.0, 1.0].
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epoches_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle=True):
        """Return the next batch_size sized samples."""
        start = self._index_in_epoch
        if start == 0 and shuffle:
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
        # Extracts the next batch data.
        if start + batch_size > self.num_samples:
            # One epoch is done.
            self._epoches_completed += 1
            self._index_in_epoch = 0
            end = self.num_samples
        else:
            end = start + batch_size
            self._index_in_epoch += batch_size
        return self._images[start:end], self._labels[start:end]
