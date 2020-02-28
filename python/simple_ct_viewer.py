import matplotlib
matplotlib.use('TkAgg')
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
from lungmask_old import *
from lungmask import lungmask


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
    ax[0].imshow(data_orig[int(slider2.val)], cmap='gray')
    ax[0].imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
    ax[0].imshow(gt_b[int(slider2.val)], cmap=cmap2)
    #ax[0].imshow(brain[int(slider2.val)], cmap3, alpha=float(slider3.val))
    ax[0].set_title('CT + lungmask')
    ax[0].set_axis_off()

    ax[1].clear()
    ax[1].imshow(gt[int(slider2.val)], cmap='gray')
    ax[1].set_title('lungmask')
    ax[1].set_axis_off()

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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1

    data_path = "/mnt/EncryptedPathology/DeepMIL/healthy_sick/"

    sets = ["negative", "positive"] #["Healthy", "Sick"]

    locs = []
    for i, curr_set in enumerate(sets):
        loc = data_path + curr_set + "/"
        for path in os.listdir(loc):
            curr_path = loc + path
            locs.append(curr_path)

    np.random.shuffle(locs)

    for name in tqdm(locs, "CT:"):

        print(name)

        # read CT
        nib_volume = nib.load(name)
        new_spacing = [1., 1., 2.]
        resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
        data = resampled_volume.get_data().astype('float32')
        
        # resize to get (512, 512) output images
        from scipy.ndimage import zoom
        img_size = 512
        data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

        # fix orientation
        
        #data = np.rot90(data, k=1, axes=(0, 1))
        #data = np.flip(data, axis=0)
        #data = np.flip(data, axis=-1)
        data = np.flip(data, axis=0)
        data = np.flip(data, axis=1)
        #data = np.rot90(data, k=2, axes=(0, 1))
        #data = np.flip(data, axis=-1)
        #data = np.rot90(data, k=1, axes=(0, 1))
        #data = np.flip(data, axis=0)

        # get z axis first
        data = np.swapaxes(data, 0, -1)

        data_orig = data.copy()

        # pre-processing
        data_orig[data_orig < -1024] = -1024
        data_orig[data_orig > 1024] = 1024

        # intensity normalize [0, 1]
        data_orig = minmaxscale(data_orig)

        #gt = lungmask3D(data.astype(np.float32), morph=False)
        img = sitk.ReadImage(name)
        gt = lungmask.apply(img)
        gt[gt > 0] = 1
        data_shapes = data.shape
        gt_shapes = gt.shape
        gt = zoom(gt, [data_shapes[0] / gt_shapes[0],\
                data_shapes[1] / gt_shapes[1],\
                data_shapes[2] / data_shapes[2]], order=0)
    
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

        f, ax = plt.subplots(1, 2, figsize=(12, 12))
        f.canvas.mpl_connect('key_press_event', up_scroll_alt)
        f.canvas.mpl_connect('key_press_event', down_scroll_alt)
        f.canvas.mpl_connect('scroll_event', up_scroll)
        f.canvas.mpl_connect('scroll_event', down_scroll)

        s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
        slider1 = Slider(s1ax, 'alpha', 0, 1.0, dragging=True, valstep=0.05)

        s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
        slider2 = Slider(s2ax, 'slice', 0, data.shape[0] - 1, valstep=1, valfmt='%1d')

        # init
        slider1.set_val(0.3)
        slider2.set_val(0)
        f.subplots_adjust(bottom=0.15)

        slider1.on_changed(images)
        slider2.on_changed(images)
        slider2.set_val(slider2.val)

        plt.show()
