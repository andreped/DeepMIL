import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
import scipy
import numba as nb
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import threading
from collections import Generator


@nb.jit(nopython=True)
def sc_any(array):
    for x in array.flat:
        if x:
            return True
    return False


def maxminscale(tmp):
    if sc_any(tmp):
        tmp = tmp - np.amin(tmp)
        tmp = tmp / np.amax(tmp)
    return tmp


# quite slow -> Don't use this! Not optimized and doesn't fit our problem!
def add_affine_transform2(input_im, max_deform):
    random_20 = uniform(-max_deform, max_deform, 2)
    random_80 = uniform(1 - max_deform, 1 + max_deform, 2)

    mat = np.array([[1, 0, 0],
                    [0, random_80[0], random_20[0]],
                    [0, random_20[1], random_80[1]]]
                   )
    input_im = affine_transform(input_im, mat, output_shape=np.shape(input_im))
    return input_im


# random 2d shift
def add_shift2(input_im, max_shift):
    sequence = [round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift))]
    input_im = shift(input_im, sequence, order=0, mode='constant')
    return input_im


# random 3d shift
def add_shift3(input_im, max_shift):
    sequence = [1, round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift))]
    input_im = shift(input_im, sequence, order=0, mode='constant')
    return input_im


# 2d rotate
def add_rotation2(input_im, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im = rotate(input_im, angle_xy, axes=(0, 1), reshape=False, mode='constant', order=1)

    return input_im


# apply same random rotation for a stack of images
def add_rotation3(input_im, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im = rotate(input_im, angle_xy, axes=(1, 2), reshape=False, mode='constant', order=1)

    return input_im


# random flip 2d
def add_flip2(input_im):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        # flip_ax = random_integers(0, 1)
        flip_ax = 1  # horizontal flip only

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, axis=flip_ax)

    return input_im


# random flip 3D, apply same transform to all images in stack
def add_flip3(input_im):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        # flip_ax = random_integers(0, 1)
        flip_ax = 2  # horizontal flip only

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, axis=flip_ax)

    return input_im


def add_rotation2_ll(input_im):
    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)

    # rotate
    input_im = np.rot90(input_im, k, axes=(0, 1))

    return input_im


def add_scaling2(input_im, r_limits):
    min_scaling, max_scaling = r_limits
    scaling_factor = np.random.uniform(min_scaling, max_scaling)

    def crop_or_fill(image, shape):
        image = np.copy(image)
        for dimension in range(2):
            if image.shape[dimension] > shape[dimension]:
                # Crop
                if dimension == 0:
                    image = image[:shape[0], :]
                elif dimension == 1:
                    image = image[:, :shape[1]]
            else:
                # Fill
                if dimension == 0:
                    new_image = np.zeros((shape[0], image.shape[1]))
                    new_image[:image.shape[0], :] = image
                elif dimension == 1:
                    new_image = np.zeros((shape[0], shape[1]))
                    new_image[:, :image.shape[1]] = image
                image = new_image
        return image

    input_im = crop_or_fill(scipy.ndimage.zoom(input_im, [scaling_factor, scaling_factor], order=1), input_im.shape)

    return input_im


def add_gamma2(input_im, r_max):
    # randomly choose direction of transform
    val = np.float32(random_integers(0, 1))
    val = 2 * val - 1

    # randomly choose gamma factor
    r_max = 3
    r = round(uniform(1, r_max))

    input_im = input_im ** (r ** val)
    input_im = input_im - np.amin(input_im)
    input_im = input_im / np.amax(input_im)

    return input_im


class batch_gen(Generator):

    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.batch_size = batch_size
        self.batch = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self.next()

    def next(self):
        with self.lock:
            batch = self.batch
            batch_size = self.batch_size
            self.batch = self.batch + self.batch_size
            # ... TODO: Didn't bother to finish this ...

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)




















