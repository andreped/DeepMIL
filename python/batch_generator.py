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


# random 3d shift
def add_shift3(input_im, max_shift):
    sequence = [round(uniform(-max_shift, max_shift)), round(uniform(-max_shift, max_shift))]
    input_im = shift(input_im, sequence, order=0, mode='constant')
    return input_im


# apply same random rotation for a stack of images
def add_rotation3(input_im, max_angle):
    # randomly choose how much to rotate for specified max_angle
    angle_xy = round(uniform(-max_angle, max_angle))

    # rotate chunks
    input_im = rotate(input_im, angle_xy, axes=(0, 1), reshape=False, mode='constant', order=1)
    
    return input_im


# random flip
def add_flip3(input_im):
    # randomly choose whether or not to flip
    if (random_integers(0, 1) == 1):
        # randomly choose which axis to flip against
        #flip_ax = random_integers(0, 1)
        flip_ax = 1 # horizontal flip only

        # flip CT-chunk and corresponding GT
        input_im = np.flip(input_im, axis=flip_ax)

    return input_im


def add_rotation3_ll(input_im):
    # randomly choose rotation angle: 0, +-90, +,180, +-270
    k = random_integers(0, high=3)

    # rotate
    input_im = np.rot90(input_im, k, axes=(0, 1))

    return input_im


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

    return input_im

"""
performs intensity transform on the chunk, using gamma transform with random gamma-value
"""
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

    for i in range(epochs):
        batch = 0

        # if nb_slices = 1 =>
        input_shape = input_shape[1:]

        # shuffle samples for each epoch
        random.shuffle(file_list)  # patients are shuffled, but chunks are after each other

        # store CTs
        bag = []
        gt_bag = []

        for filename in file_list:
            filename = data_path + filename + "/"
            
            tmp_bag = []

            # get slices nicely sorted
            slices = np.array(os.listdir(filename))
            tmp = np.array([int(x.split(".")[0]) for x in slices])
            slices = slices[np.argsort(tmp)]
            #print(slices)

            # shuffle data
            #np.random.shuffle(slices)

            # randomly extract bag_batch number of samples from
            for file in slices: #os.listdir(filename): # <- perhaps shuffle the order? Does that makes sense?

                f = h5py.File(filename + file, 'r')
                input_im = np.array(f['data']).astype(np.float32)
                if mask_flag:
                    mask = np.array(f['lungmask']).astype(np.float32)
                    input_im[mask == 0] = 0
                f.close()

                #orig = input_im.copy()

                '''
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(input_im, cmap="gray")
                ax[1].imshow(orig, cmap="gray")
                ax[2].imshow(mask, cmap="gray")
                plt.show()
                '''

                # apply specified agumentation on both image stack and ground truth
                if 'gauss' in aug:
                    input_im = add_gaussBlur2(input_im.copy(), aug["sigma"])
                
                if 'rotate' in aug:  # -> do this last maybe?
                    input_im = add_rotation2(input_im.copy(), aug['rotate'])

                if 'affine' in aug:
                    input_im = add_affine_transform2(input_im.copy(), aug['affine'])

                if 'shift' in aug:
                    input_im = add_shift2(input_im.copy(), aug['shift'])

                if 'flip' in aug:
                    input_im = add_flip2(input_im.copy())

                if 'zoom' in aug:
                    input_im = add_scaling2(input_im.copy(), aug['zoom'])
                
                #input_im = np.expand_dims(input_im, axis=-1)
                #input_im = np.expand_dims(input_im, axis=0)

                # add augmented sample to temporary bag
                tmp_bag.append(input_im)
                del input_im #, output
            
            ff = h5py.File(filename + file, 'r')
            output = np.array(ff['output']).astype(np.float32)
            ff.close()

            bag_batch = np.array(tmp_bag) #.copy()
            bag_label = output
            gt_bag.append(bag_label)

            bag_batch = bag_batch[:bag_size]
            tmp = np.zeros((bag_size,) + (bag_batch.shape[1:])).astype(np.float32)
            tmp[:bag_batch.shape[0]] = bag_batch
            bag_batch = tmp.copy()
            del tmp

            bag_batch = np.moveaxis(bag_batch, 0, -1)
            bag.append(bag_batch)

            batch += 1
            if batch == batch_size:
                bag_batch = np.array(bag)
                gt_batch = np.array(gt_bag)
                #print()
                if model_type == "3DCNN":
                    bag_batch = np.expand_dims(bag_batch, axis=-1)
                #print(bag_batch.shape)
                batch = 0
                tmp_bag = []
                bag = []
                gt_bag = []
                yield bag_batch, gt_batch

def batch_length(file_list):
    length = len(file_list)
    print('CTs in generator:', length)
    return length
