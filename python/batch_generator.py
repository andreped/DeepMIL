# batch gen
import random
import h5py
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, affine_transform, zoom
from numpy.random import random_sample, rand, random_integers, uniform
import scipy
import numba as nb
import os


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
def add_affine_transform3(input_im, output, max_deform):
    random_20 = uniform(-max_deform, max_deform, 2)
    random_80 = uniform(1 - max_deform, 1 + max_deform, 2)

    mat = np.array([[1, 0, 0],
                    [0, random_80[0], random_20[0]],
                    [0, random_20[1], random_80[1]]]
                   )
    input_im[:, :, :, 0] = affine_transform(input_im[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 0] = affine_transform(output[:, :, :, 0], mat, output_shape=np.shape(input_im[:, :, :, 0]))
    output[:, :, :, 1] = affine_transform(output[:, :, :, 1], mat, output_shape=np.shape(input_im[:, :, :, 0]))

    output[output < 0.5] = 0
    output[output >= 0.5] = 1

    return input_im, output


"""
###
input_im:		input image, 5d ex: (1,64,256,256,1) , (dimi0, z, x, y, channel)
output:			ground truth, 5d ex: (1,64,256,256,2), (dimi0, z, x, y, channel)
max_shift:		the maximum amount th shift in a direction, only shifts in x and y dir
###
"""


# random 3d shift
def add_shift3(input_im, output, max_shift):
    sequence = [0, round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift))]

    input_im[..., 0] = shift(input_im[..., 0], sequence, order=0, mode='constant')
    output[..., 0] = shift(output[..., 0], sequence, order=0, mode='constant', cval=1)
    output[..., 1] = shift(output[..., 1], sequence, order=0, mode='constant', cval=0)

    return input_im, output


# apply same random rotation for a stack of images
def add_rotation3(input_im, output, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im = rotate(input_im, angle_xy, axes=(1, 2), reshape=False, mode='constant', order=1)
    output[..., 0] = rotate(output[..., 0], angle_xy, axes=(1, 2), reshape=False, mode='constant', cval=1, order=0)
    output[..., 1] = rotate(output[..., 1], angle_xy, axes=(1, 2), reshape=False, mode='constant', cval=0, order=0)

    return input_im, output


# random flip
def add_flip3(input_im, output):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        flip_ax = random_integers(0, 2)

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, flip_ax)
        output = np.flip(output, flip_ax)

    return input_im, output


def add_rotation3_ll(input_im, output):
    # randomly choose which axis to rotate around
    # val = np.float32(random_integers(0, 1))
    # val = 2*val - 1

    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)

    # rotate
    input_im[:, :, :, 0] = np.rot90(input_im[:, :, :, 0], k, axes=(1, 2))

    output[:, :, :, 1] = np.rot90(output[:, :, :, 1], k, axes=(1, 2))
    output[:, :, :, 0] = np.rot90(output[:, :, :, 0], k, axes=(1, 2))

    return input_im, output


def add_scaling3(input_im, output, r_limits):

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
                    image = image[:, :shape[1], :]
            else:
                # Fill
                if dimension == 0:
                    new_image = np.zeros((shape[0], image.shape[1], shape[2]))
                    new_image[:image.shape[0], :, :] = image
                elif dimension == 1:
                    new_image = np.zeros((shape[0], shape[1], shape[2]))
                    new_image[:, :image.shape[1], :] = image
                image = new_image
        return image

    for i in range(input_im.shape[0]):
        input_im[i] = crop_or_fill(scipy.ndimage.zoom(input_im[i], [scaling_factor,scaling_factor,1], order=1), input_im.shape[1:])
        output[i] = crop_or_fill(scipy.ndimage.zoom(output[i], [scaling_factor, scaling_factor, 1], order=0), output.shape[1:])

    return input_im, output

"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""


def add_gamma3(input_im, output):
    # randomly choose direction of transform
    val = np.float32(random_integers(0, 1))
    val = 2 * val - 1

    # randomly choose gamma factor
    r_max = 3
    r = round(uniform(1, r_max))

    input_im[:, :, :, 0] = input_im[:, :, :, 0] ** (r ** val)
    input_im = input_im - np.amin(input_im)
    input_im = input_im / np.amax(input_im)

    return input_im, output


"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
		->  'rotate': 20 , in deg. slow
		->	'shift': 20, in pixels. slow
		->	'affine': 0.2 . should be between 0.05 and 0.3. slow
		->	'flip': 1, fast
"""


def batch_gen3(file_list, batch_size, aug={}, nb_classes=2, input_shape=(16, 512, 512, 1), epochs=1, data_path=''):

    for i in range(epochs):
        batch = 0

        bag_batch = []
        bag_label = []

        # shuffle samples for each epoch
        random.shuffle(file_list)  # patients are shuffled, but chunks are after each other

        for filename in file_list:
            filename = data_path + filename + "/"

            bag_batch = []
            bag_label = []

            tmp_bag = []

            # randomly extract bag_batch number of samples from
            for file in os.listdir(filename):

                f = h5py.File(filename + file, 'r')
                input_im = np.array(f['data']).astype(np.float32)
                #output = np.array(f['output']).astype(np.float32)
                f.close()

                #input_im = np.squeeze(input_im, axis=0)
                #output = np.squeeze(output, axis=0)

                # apply specified agumentation on both image stack and ground truth
                if 'rotate' in aug:  # -> do this last maybe?
                    input_im, output = add_rotation3(input_im.copy(), output.copy(), aug['rotate'])

                if 'affine' in aug:
                    input_im, output = add_affine_transform3(input_im.copy(), output.copy(), aug['affine'])

                if 'shift' in aug:
                    input_im, output = add_shift3(input_im.copy(), output.copy(), aug['shift'])

                if 'flip' in aug:
                    input_im, output = add_flip3(input_im.copy(), output.copy())

                if 'zoom' in aug:
                    input_im, output = add_scaling3(input_im.copy(), output.copy(), aug['zoom'])
                
                input_im = np.expand_dims(input_im, axis=-1)
                input_im = np.expand_dims(input_im, axis=0)

                #tmp = np.zeros(2, dtype=np.float32)
                #tmp[int(output[0])] = 1
                #output = tmp.copy()
                
                #print(input_im.shape)
                #print(output)

                # add augmented sample to batch
                #bag_batch.append(input_im)
                #bag_label.append(output)
                tmp_bag.append(input_im)
                #del input_im, output

                '''
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(im[batch, 8, ..., 0], cmap="gray")
                ax[1].imshow(gt[batch, 8, ..., 0], cmap="gray")
                ax[2].imshow(gt[batch, 8, ..., 1], cmap="gray")
                plt.show()
                '''

                #batch = batch + 1
                #if batch == bag_size:
                #    batch = 0
            
            #print("---")
            #print(filename)
            f = h5py.File(filename + file, 'r')
            #input_im = np.array(f['data']).astype(np.float32)
            output = np.array(f['output']).astype(np.float32)
            f.close()
            #tmp_bag = np.concatenate(tmp_bag)
            #print(tmp_bag.shape)
            bag_batch.append(tmp_bag) #tmp_bag.copy() #.append(tmp_bag)
            #print(len(tmp_bag))
            #print()
            #bag_batch = tmp_bag.copy()
            #bag_label.append([output])
            bag_batch = tmp_bag.copy()
            bag_label = [output]

            batch += 1
            if batch == batch_size:
                out_batch = bag_batch.copy()
                out_label = bag_label.copy()
                batch = 0

                bag_label = []
                bag_batch = []
                tmp_bag = []
                yield out_batch, out_label

def batch_length(file_list):
    length = len(file_list)
    print('chunks in generator:', length)
    return length
