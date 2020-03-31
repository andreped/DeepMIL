# batch gen
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
import scipy
import numba as nb
import os
import matplotlib.pyplot as plt


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


"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""


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
        #flip_ax = random_integers(0, 1)
        flip_ax = 1 # horizontal flip only

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, axis=flip_ax)

    return input_im


# random flip 3D, apply same transform to all images in stack
def add_flip3(input_im):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        #flip_ax = random_integers(0, 1)
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

"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""
def add_gamma2(input_im, r_max):
    # randomly choose whether to augment or not
    if np.random_integers(0, 1) == 1:

        # randomly choose direction of transform
        val = np.float32(random_integers(0, 1))
        val = 2 * val - 1

        # randomly choose gamma factor
        r = round(uniform(1, r_max))

        input_im = input_im ** (r ** val)
        input_im = input_im - np.amin(input_im)
        input_im = input_im / np.amax(input_im)

    return input_im


"""
gaussian blur aug
"""
#def add_gaussBlur2(input_im, ):



"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
		->  'rotate': 20 , in deg. slow
		->	'shift': 20, in pixels. slow
		->	'affine': 0.2 . should be between 0.05 and 0.3. slow
		->	'flip': 1, fast
"""

def batch_gen3(file_list, batch_size, aug={}, nb_classes=2, input_shape=(16, 512, 512, 1), epochs=1, data_path='',
               mask_flag=True, bag_size=1, model_type=""):
    
    input_shape = input_shape[1:]

    for i in range(epochs):
        batch = 0

        # if nb_slices = 1 =>

        # shuffle samples for each epoch
        random.shuffle(file_list)  # patients are shuffled, but chunks are after each other

        # store CTs
        batch_bags = []
        batch_bag_label = []

        for filename in file_list:
            filename = data_path + filename + "/"

            # initialize at start of batch
            if batch == 0:
                batch_bags = []
                batch_bag_label = []
            
            #curr_bag = []

            # read whole volume as an array
            f = h5py.File(filename + "1.h5", 'r')
            curr_bag = np.array(f['data']).astype(np.float32)
            curr_bag_label = np.array(f['output']).astype(np.float32)
            if mask_flag:
                mask = np.array(f['lungmask']).astype(np.float32)
                curr_bag[mask == 0] = 0
            f.close()

            # shuffle images in stack to make network invariant to position (DON'T USE WITH 3DCNN!)
            #if model_type != "3DCNN":
            #    slice_order = list(range(curr_bag.shape[0]))
            #    np.random.shuffle(slice_order)
            #    curr_bag = curr_bag[slice_order]

            # apply specified agumentation on both image stack and ground truth
            if 'gauss' in aug:
                curr_bag = add_gaussBlur2(curr_bag.copy(), aug["sigma"])

            if 'rotate' in aug:  # -> do this last maybe?
                curr_bag = add_rotation3(curr_bag.copy(), aug['rotate'])

            if 'affine' in aug:
                curr_bag = add_affine_transform2(curr_bag.copy(), aug['affine'])

            if 'shift' in aug:
                curr_bag = add_shift3(curr_bag.copy(), aug['shift'])

            if 'flip' in aug:
                curr_bag = add_flip3(curr_bag.copy())

            if 'zoom' in aug:
                curr_bag = add_scaling2(curr_bag.copy(), aug['zoom'])

            if model_type == "InceptionalMIL2D":
                curr_bag = np.stack((curr_bag,)*3, axis=-1)
            else:
                curr_bag = np.expand_dims(curr_bag, axis=-1)

            if model_type == "I am not one-hotting anymore":  # model_type == "3DCNN":
                #curr_bag = np.expand_dims(curr_bag, axis=0)
                mask = np.zeros(2, dtype=np.float32)
                mask[int(curr_bag_label[0])] = 1
                batch_bag_label.append(mask)
                batch_bags.append(curr_bag)
            elif model_type == "2DMIL":
                batch_bags.append(curr_bag)  # (curr_bag))
                batch_bag_label.append(curr_bag_label[0] * np.ones(len(curr_bag)))
            else:
                batch_bags.append(curr_bag)  # ((curr_bag))
                batch_bag_label.append(curr_bag_label)  # (curr_bag_label[0] * np.ones(len(curr_bag)))

            # append CT/bag and label to batch
            del curr_bag, curr_bag_label

            batch += 1
            if batch == batch_size:
                if model_type == "3DCNN":
                    batch_bags = np.array(batch_bags)
                    batch_bag_label = np.array(batch_bag_label)
                batch = 0
                yield batch_bags, batch_bag_label

def batch_length(file_list):
    length = len(file_list)
    print('CTs in generator:', length)
    return length
