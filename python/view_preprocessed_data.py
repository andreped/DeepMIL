import matplotlib
#matplotlib.use('GTk3Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import SimpleITK as sitk
import cv2
from matplotlib.widgets import Slider
from tqdm import tqdm
from skimage.morphology import binary_opening, disk, ball, remove_small_holes, remove_small_objects, binary_dilation
from numpy.random import shuffle
import os
import h5py
import cv2
from scipy.ndimage import binary_fill_holes
import nibabel as nib
from nibabel.processing import *
from copy import deepcopy
#from lungmask import lungmask
import shutil
from data_aug import *


def import_set(tmp, file):
    f = h5py.File(file, 'r')
    tmp = np.array(f[tmp])
    f.close()
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    return tmp

def unique_patients(tmp):
    l = []
    for t in tmp:
        l.append(int(t.split("/")[-2]))
    return np.unique(l)

def images(event):
    ax[0].clear()
    ax[0].imshow(data_orig[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
    #ax[0].imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
    #ax[0].imshow(gt_b[int(slider2.val)], cmap=cmap2)
    ax[0].set_title('CT + lungmask')
    ax[0].set_axis_off()

    ax[1].clear()
    ax[1].imshow(masked[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(image + " | " + str(sets[curr_label]))
    ax[1].set_axis_off()

    ax[2].clear()
    ax[2].imshow(gt[int(slider2.val)], cmap="gray", vmin=0, vmax=1)
    ax[2].set_title('lungmask')
    ax[2].set_axis_off()

    f.suptitle('slice ' + str(int(slider2.val)))
    f.canvas.draw_idle()


def up_scroll_alt(event):
    if event.key == "up":
        if (slider2.val + 2 > data.shape[0]):
            1
        # print("Whoops, end of stack", print(slider2.val))
    else:
        slider2.set_val(slider2.val + 1)


def down_scroll_alt(event):
    if event.key == "down":
        if (slider2.val - 1 < 0):
            1
        # print("Whoops, end of stack", print(slider2.val))
    else:
        slider2.set_val(slider2.val - 1)


def up_scroll(event):
    if event.button == 'up':
        if (slider2.val + 2 > data.shape[0]):
            1
        # print("Whoops, end of stack", print(slider2.val))
    else:
        slider2.set_val(slider2.val + 1)


def down_scroll(event):
    if event.button == 'down':
        if (slider2.val - 1 < 0):
            1
        # print("Whoops, end of stack", print(slider2.val))
    else:
        slider2.set_val(slider2.val - 1)


def minmaxscale(tmp):
    if len(np.unique(tmp)) > 1:
        tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    return tmp


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1

    #dataset = '040320_binary_healthy_sick_shape_(1,256,256)_huclip_[-1024,1024]_spacing_[1.0,1.0,2.0]'
    #dataset = '040320_binary_healthy_sick_shape_(1,128,128)_huclip_[-1024,1024]_spacing_[1.0,1.0,2.0]'
    #dataset = "250320_binary_healthy_emphysema_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"
    #dataset = "270320_binary_healthy_emphysema_shape_(64,256,256)_huclip_[-1050,-750]_spacing_[1,1,1]_3DCNN"
    #dataset = "290320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"
    dataset = "290320_binary_healthy_cancer_shape_(1,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    dataset = "300320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    dataset = "050520_binary_healthy_sick_shape_(128,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN_True_2DMIL"
    #data_path = "/mnt/EncryptedPathology/DeepMIL/datasets/" + dataset + "/"
    data_path = "/home/andrep/workspace/DeepMIL/data/" + dataset + "/"

    features_flag = False  # Default: False

    sets = ["negative", "positive"] #["Healthy", "Sick"]

    aug = {}  # {'flip':1, 'rotate':10, 'shift':10}
    #aug = {'flip':1, 'rotate':10, 'gamma':[0.5, 1.5]}

    locs = os.listdir(data_path)[::-1]
    #locs = locs[3104:]
    np.random.shuffle(locs)

    for filename in tqdm(locs, "CT:"):

        #print()
        #print("\n--")
        #print(filename)
        path = data_path + filename + "/"

        # get slices nicely sorted
        file = "1.h5"  # np.array(os.listdir(filename))[0]

        try:
            f = h5py.File(path + file, 'r')
        except:
            print("didn't fucking work")
        data = np.array(f['data']).astype(np.float32)
        gt = np.array(f['lungmask']).astype(np.float32)
        if features_flag:
            features = np.array(f['features']).astype(np.float32)
        f.close()

        #'''

        image = filename
        curr_label = int(filename.split("_")[0])

        #data.append(input_im)
        #lungmask.append(lungmask_im)

        if features_flag:
            features = np.array(features)

        data_orig = np.array(data)
        #gt = np.array(lungmask)

        masked = data_orig.copy()
        masked[gt == 0] = 0

        # apply specified agumentation on both image stack and ground truth
        if 'gauss' in aug:
            masked = add_gaussBlur2(masked.copy(), aug["sigma"])

        if 'rotate' in aug:  # -> do this last maybe?
            masked = add_rotation3(masked.copy(), aug['rotate'])

        if 'affine' in aug:
            masked = add_affine_transform2(masked.copy(), aug['affine'])

        if 'shift' in aug:
            masked = add_shift3(masked.copy(), aug['shift'])

        if 'flip' in aug:
            masked = add_flip3(masked.copy())

        if 'zoom' in aug:
            masked = add_scaling2(masked.copy(), aug['zoom'])

        if 'gamma' in aug:
            masked = add_gamma2(masked.copy(), aug["gamma"])


        # generate boundary image
        gt_b = np.zeros_like(gt)
        for i in range(gt.shape[0]):
            if len(np.unique(gt[i])) > 1:
                gt_b[i] = cv2.Canny((gt[i] * 255).astype(np.uint8), 0, 255)

        gt_b = gt_b.astype(np.float32)
        gt_b = gt_b / np.amax(gt_b)

        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 3)]
        cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 3)]
        cmap2 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
        colors = [(0, 1, 0, i) for i in np.linspace(0, 1, 3)]
        cmap3 = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

        f, ax = plt.subplots(1, 3, figsize=(24, 12))
        f.canvas.mpl_connect('key_press_event', up_scroll_alt)
        f.canvas.mpl_connect('key_press_event', down_scroll_alt)
        f.canvas.mpl_connect('scroll_event', up_scroll)
        f.canvas.mpl_connect('scroll_event', down_scroll)

        s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
        slider1 = Slider(s1ax, 'alpha', 0, 1.0, dragging=True, valstep=0.05)

        s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
        slider2 = Slider(s2ax, 'slice', 0, data_orig.shape[0] - 1, valstep=1, valfmt='%1d')

        # init
        slider1.set_val(0.3)
        slider2.set_val(0)
        f.subplots_adjust(bottom=0.15)

        slider1.on_changed(images)
        slider2.on_changed(images)
        slider2.set_val(slider2.val)

        plt.show()

        #'''
