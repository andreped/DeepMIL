import numpy as np
import os
import sys
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from datetime import date
from tqdm import tqdm
import multiprocessing as mp
from tensorflow.python.keras.applications.vgg16 import VGG16
from nibabel.processing import resample_to_output
import configparser

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def minmaxscale(x):
    if len(np.unique(x)) > 1:
        x = (x - x.min()) / (x.max() - x.min())
    return x


def func(path):

    class_val, curr_ct = path

    curr_id = curr_ct.split("/")[-1].split(".")[0]
    #print(curr_id)

    #curr_ct = str(curr_ct)
    #curr_path = all_data_path + curr_ct + ".nii.gz"

    # read CT
    nib_volume = nib.load(curr_ct)
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    data = resampled_volume.get_data().astype('float32')

    # resize to get (512, 512) output images
    from scipy.ndimage import zoom
    img_size = input_shape[1]
    data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

    # pre-processing
    data[data < hu_clip[0]] = hu_clip[0]
    data[data > hu_clip[1]] = hu_clip[1]

    # intensity normalize [0, 1]
    data = minmaxscale(data)

    # fix orientation
    data = np.flip(data, axis=1)  # np.rot90(data, k=1, axes=(0, 1))
    data = np.flip(data, axis=0)

    # get z axis first
    data = np.swapaxes(data, 0, -1)

    # get lung mask, but don't mask the data. Add it in the generated data file, to be used in batchgen if of interest
    gt_nib_volume = nib.load(data_path[:-1] + "_lungmask/" + curr_ct.split(data_path)[1].split(".")[0] + "_lungmask.nii.gz")
    resampled_volume = resample_to_output(gt_nib_volume, new_spacing, order=0)
    gt = resampled_volume.get_data().astype('float32')

    # fix orientation
    gt = np.flip(gt, axis=1)
    gt = np.flip(gt, axis=0)

    # get z axis first
    gt = np.swapaxes(gt, 0, -1)

    gt[gt > 0] = 1
    data_shapes = data.shape
    gt_shapes = gt.shape
    gt = zoom(gt, [data_shapes[0] / gt_shapes[0],
                   data_shapes[1] / gt_shapes[1],
                   data_shapes[2] / gt_shapes[2]], order=0)
    # data[gt == 0] = 0 # mask CT with lung mask # <- DONT MASK, one can easily do that if of interest in batchgen
    del gt_shapes, data_shapes

    # for each CT, make a folder and store each sample in its own respective file
    curr_end_path = end_path + str(class_val) + "_" + curr_id + "/"
    if not os.path.exists(curr_end_path):
        os.makedirs(curr_end_path)

    if input_shape[0] == 1:
        for j in range(data.shape[0]):
            if np.count_nonzero(data[j]) == 0:
                continue
            with h5py.File(curr_end_path + str(j) + ".h5", "w") as ff:
                ff.create_dataset("data", data=data[j].astype(np.float32), compression="gzip", compression_opts=4)
                ff.create_dataset("output", data=np.array([class_val]), compression="gzip", compression_opts=4)
                ff.create_dataset("lungmask", data=gt[j].astype(np.float32), compression="gzip", compression_opts=4)
    else:  # slab_size != 1:
        num = input_shape[0]
        for j in range(int(np.ceil(data.shape[0] / input_shape[0]))):
            tmp = np.zeros(input_shape, dtype=np.float32)
            slab_CT = data[int(num * j):int(num * (j + 1))]
            tmp[:slab_CT.shape[0]] = slab_CT
            if np.count_nonzero(tmp) == 0:
                continue
            tmp2 = np.zeros(input_shape, dtype=np.float32)
            slab_GT = gt[int(num * j):int(num * (j + 1))]
            tmp2[:slab_GT.shape[0]] = slab_GT

            with h5py.File(curr_end_path + str(j) + ".h5", "w") as ff:
                ff.create_dataset("data", data=tmp.astype(np.float32), compression="gzip", compression_opts=4)
                ff.create_dataset("output", data=np.array([class_val]), compression="gzip", compression_opts=4)
                ff.create_dataset("lungmask", data=tmp2.astype(np.float32), compression="gzip", compression_opts=4)



if __name__ == '__main__':

    # today's date
    dates = date.today()
    dates = dates.strftime("%d%m") + dates.strftime("%Y")[:2]

    # read and parse config file
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    conf = config["Default"]

    # mask_flag = True #True or False
    input_shape = eval(conf["input_shape"])  # (1, 256, 256)
    hu_clip = eval(conf["hu_clip"])  # [-1024, 1024]
    new_spacing = eval(conf["new_spacing"])  # [1., 1., 2.]

    data_path = conf["data_path"]  # "/mnt/EncryptedPathology/DeepMIL/healthy_sick/"
    datasets_path = conf["datasets_path"]  # "/home/andrep/workspace/DeepMIL/data/"
    del config
    end_path = datasets_path + dates + "_binary_healthy_sick" +\
               "_shape_" + str(input_shape).replace(" ", "") +\
               "_huclip_" + str(hu_clip).replace(" ", "") +\
               "_spacing_" + str(new_spacing).replace(" ", "") +\
               "/"

    sets = ["negative", "positive"] #["Healthy", "Sick"]

    locs = []
    for i, curr_set in enumerate(sets):
        loc = data_path + curr_set + "/"
        for path in os.listdir(loc):
            curr_path = loc + path
            locs.append([i, curr_path])

    if not os.path.exists(end_path):
        os.makedirs(end_path)

    proc_num = int(conf["threads"])  # 16
    p = mp.Pool(proc_num)
    num_tasks = len(locs)
    r = list(tqdm(p.imap(func, locs), "CT", total=num_tasks))  # list(tqdm(p.imap(func,gts),total=num_tasks))
    p.close()
    p.join()