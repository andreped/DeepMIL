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


def batch_gen_features3(file_list, batch_size, aug={}, nb_classes=2, input_shape=(16, 512, 512, 1), epochs=1,
                        data_path='', mask_flag=True, bag_size=1):
    for i in range(epochs):
        batch = 0

        # shuffle samples for each epoch
        random.shuffle(file_list)  # patients are shuffled, but chunks are after each other

        features_batch = []
        label_batch = []

        nb_features = 2048

        for filename in file_list:
            filename = data_path + filename + "/"

            # get slices
            # slices = os.listdir(filename)
            # np.random.shuffle(slices)

            # get slices nicely sorted
            slices = np.array(os.listdir(filename))
            tmp = np.array([int(x.split(".")[0]) for x in slices])
            slices = slices[np.argsort(tmp)]
            # print(slices)

            # shuffle data
            # np.random.shuffle(slices)
            features = []

            # randomly extract bag_batch number of samples from
            for file in slices:  # os.listdir(filename): # <- perhaps shuffle the order? Does that makes sense?

                f = h5py.File(filename + file, 'r')
                feat = np.array(f['features']).astype(np.float32)
                f.close()

                features.append(feat)

            ff = h5py.File(filename + file, 'r')
            output = np.array(ff['output']).astype(np.float32)
            ff.close()

            features = np.array(features)
            features = features.flatten()
            features = features[:bag_size]

            tmp = np.zeros(int(bag_size * nb_features))
            tmp[:len(features)] = features
            features = tmp.copy()
            del tmp


            features_batch.append(features)
            label_batch.append(output[0])

            batch += 1
            if batch == batch_size:
                out = np.array(features_batch)
                gt = np.array(label_batch)
                features_batch = []
                label_batch = []
                batch = 0
                yield out, gt


def batch_length(file_list):
    length = len(file_list)
    print('CTs in generator:', length)
    return length
