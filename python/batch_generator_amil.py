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
from data_aug import *


"""
aug: 		dict with what augmentation as key and what degree of augmentation as value
		->  'rotate': 20 , in deg. slow
		->	'shift': 20, in pixels. slow
		->	'affine': 0.2 . should be between 0.05 and 0.3. slow
		->	'flip': 1, fast
"""

def batch_gen3_amil(file_list, N=None, batch_size=1, aug={}, nb_classes=2, input_shape=(16, 512, 512, 1), slab_shape=(16, 512, 512, 1), epochs=1,
               data_path='', mask_flag=True, bag_size=1, model_type=""):
    
    #input_shape = input_shape[1:]
    while True:  # <- necessary for end of training (last epoch)
        for i in range(epochs):
            batch = 0

            # TODO: instead of preparing positive/negative chunks beforehand, let's do a sampling scheme to balance the classes (negative: all samples in batch are negative,
            #  positive: only one needs to be positive -> clear imbalance, need to balance this)

            # shuffle samples for each epoch (within each label)
            for x in file_list:
                np.random.shuffle(x)

            for n in range(N):
                # randomly choose which bag label to create
                val = np.random.choice([0, 1])

                # if negative is chosen, generate an all random negative batch of user-specified batch_size
                if val == 0:
                    batch_file_list = np.random.choice(file_list[val], batch_size, replace=False)
                else:
                    tmp_list = file_list[val].copy()
                    first_sample = np.random.choice(tmp_list)
                    flattened_list = [x for sublist in file_list for x in sublist]
                    flattened_list.remove(first_sample)
                    batch_file_list = [first_sample] + list(np.random.choice(flattened_list, batch_size - 1))

                #print()
                #print(val)
                #print(batch_file_list)
                # finally, need to shuffle returned list
                np.random.shuffle(batch_file_list)

                # store CTs
                batch_bags = []
                batch_bag_label = []

                for filename in batch_file_list:

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

                    #exit()

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

                    if 'gamma' in aug:
                        curr_bag = add_gamma2(curr_bag.copy(), aug["gamma"])

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
                    elif "AMIL" in model_type:
                        batch_bags.append(curr_bag)  # (curr_bag))
                        batch_bag_label.append(curr_bag_label[0] * np.ones(len(curr_bag)))
                    elif "HMIL" in model_type:
                        #print(curr_bag.shape)
                        #print(curr_bag_label)
                        batch_bags.append(curr_bag)
                        batch_bag_label.append(curr_bag_label)
                    else:
                        batch_bags.append(curr_bag)  # ((curr_bag))
                        batch_bag_label.append(curr_bag_label)  # (curr_bag_label[0] * np.ones(len(curr_bag)))

                    # append CT/bag and label to batch
                    del curr_bag, curr_bag_label

                    batch += 1
                    if batch == batch_size:
                        if (model_type == "3DCNN"):  #  or ("HMIL" in model_type):
                            batch_bags = np.array(batch_bags)
                            batch_bag_label = np.array(batch_bag_label)
                        if ("AMIL" in model_type):
                            batch_bags = np.concatenate(batch_bags)
                            #batch_bags = np.array(batch_bags)
                            #batch_bag_label = np.array(batch_bag_label)
                            #print(batch_bags.shape)
                           # batch_bags = list(batch_bags)
                            batch_bags = [batch_bags]
                            #print("---")
                            #print(shapes)
                            #print(np.unique(batch_bag_label))
                            batch_bag_label = [np.unique(batch_bag_label)[-1] * np.ones(int(input_shape[0]/slab_shape[0]*batch_size))]
                            #batch_bag_label.append(np.unique(batch_bag_label)[-1] * np.ones(len(batch_bags[0])))
                            #batch_bag_label = list(batch_bag_label)
                        batch = 0
                        yield batch_bags, batch_bag_label

def batch_length(file_list):
    length = len(file_list)
    print('CTs in generator:', length)
    return length