import numpy as np
import os, sys
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from datetime import date
from tqdm import tqdm
import multiprocessing as mp
from tensorflow.python.keras.applications.vgg16 import VGG16
from nibabel.processing import resample_to_output
from lungmask import lungmask

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
    new_spacing = [1., 1., 2.]
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    data = resampled_volume.get_data().astype('float32')

    # resize to get (512, 512) output images
    from scipy.ndimage import zoom
    img_size = 256
    data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

    # pre-processing
    data[data < -1024] = -1024
    data[data > 1024] = 1024

    # intensity normalize [0, 1]
    data = minmaxscale(data)

    # fix orientation
    data = np.flip(data, axis=1) #np.rot90(data, k=1, axes=(0, 1))
    data = np.flip(data, axis=0)

    # get z axis first
    data = np.swapaxes(data, 0, -1)
    
    if mask_flag:
        # mask data to only keep lung area
        img = sitk.ReadImage(curr_ct)
        gt = lungmask.apply(img)
        gt[gt > 0] = 1
        data_shapes = data.shape
        gt_shapes = gt.shape
        gt = zoom(gt, [data_shapes[0] / gt_shapes[0],\
                data_shapes[1] / gt_shapes[1],\
                data_shapes[2] / data_shapes[2]], order=0)
        data[gt == 0] = 0
        del gt, img, gt_shapes, data_shapes

    # get trained encoder
    #model = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    # for each CT, make a folder and store each sample in its own respective file
    curr_end_path = end_path + str(class_val) + "_" + curr_id + "/"
    if not os.path.exists(curr_end_path):
        os.makedirs(curr_end_path)

    if dim == 2:
        for i in range(data.shape[0]):
            if np.count_nonzero(data[i]) == 0:
                continue
            #print(i)
            #tmp = data[i]
            #print()
            #print(tmp.shape)
            #tmp = np.repeat(tmp[..., np.newaxis], 3, -1)
            #print(tmp.shape)
            #tmp = np.expand_dims(tmp, axis=0)
            #print(tmp.shape)
            #features = model.predict(tmp).flatten()
            with h5py.File(curr_end_path + str(i) + ".h5", "w") as f:
                f.create_dataset("data", data=data[i].astype(np.float32), compression="gzip", compression_opts=4)
                #f.create_dataset("features/" + str(i), data=features.astype(np.float32), compression="gzip", compression_opts=4)
                f.create_dataset("output", data=np.array([class_val]), compression="gzip", compression_opts=4)


if __name__ == '__main__':

    # today's date
    dates = date.today()
    dates = dates.strftime("%d%m") + dates.strftime("%Y")[:2]

    dim = 2
    mask_flag = False #True

    data_path = "/mnt/EncryptedPathology/DeepMIL/healthy_sick/"
    datasets_path = "/home/andrep/workspace/DeepMIL/data/"
    end_path = datasets_path + dates + "_" + "dim_" + str(dim) + "_binary_healthy_sick" +\
            "_lungmask_" + str(mask_flag) + "/"

    sets = ["negative", "positive"] #["Healthy", "Sick"]

    locs = []
    for i, curr_set in enumerate(sets):
        loc = data_path + curr_set + "/"
        for path in os.listdir(loc):
            curr_path = loc + path
            locs.append([i, curr_path])

    if not os.path.exists(end_path):
        os.makedirs(end_path)

    # nice sort
    #locs = np.array(locs)
    #gts = np.array(gts)
    #index = np.argsort(locs)
    #locs = locs[index]
    #gts = gts[index]

    proc_num = 16 # 16
    p = mp.Pool(proc_num)
    num_tasks = len(locs)
    r = list(tqdm(p.imap(func, locs), "CT", total=num_tasks))  # list(tqdm(p.imap(func,gts),total=num_tasks))
    p.close()
    p.join()












exit()

##### HEALTHY/SICK LUNG DATASET

classes = ["Healthy", "Sick"]
classes_nums = np.array([0, 1])

# 2D or 3D
dim = 2

datasets_path = "/home/andrep/workspace/DeepMIL/data/"
end_path = datasets_path + dates + "_" + "dim_" + str(dim) + "_binary_healthy_sick" + "/"

if not os.path.exists(end_path):
    os.makedirs(end_path)

for value_class, curr_class in enumerate(tqdm(classes, "Class: ")):
    curr_data_path = data_path + curr_class + "/"

    curr_end_class_path = end_path + str(curr_class)

    if not os.path.exists(curr_end_class_path):
        os.path.exists(curr_end_class_path)

    for curr_ct in tqdm(os.listdir(curr_data_path), "CT: "):
        curr_path = curr_data_path + curr_ct

        itkimage = sitk.ReadImage(curr_path)
        data = sitk.GetArrayFromImage(itkimage).astype(np.float32)

        # pre-processing
        data[data < -1024] = -1024
        data[data > 1024] = 1024

        if dim == 2:
            for i in tqdm(range(data.shape[0]), "Slab: "):
                tmp_data = data[0]
                with h5py.File(curr_end_class_path + curr_ct.split(".nii.gz")[0] + ".h5", "w") as f:
                    f.create_dataset("input", data=tmp_data.astype(np.float32), compression="gzip", compression_opts=4)
                    f.create_dataset("output", data=np.array([value_class]), compression="gzip", compression_opts=4)

    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[F")
