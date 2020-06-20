import numpy as np
import os
import h5py
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import configparser
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from scipy.ndimage import zoom
from model import AtnVGG, Attention, GatedAttention


def getClassDistribution(tmp):
    cc = []
    classes = [int(x.split("_")[0]) for x in tmp]
    return [sum(classes == x) for x in np.unique(classes)]


def images(event):
    ax[0].clear()
    ax[0].imshow(data_orig[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
    ax[0].imshow(heatmap[int(slider2.val)], vmin=0, vmax=1, alpha=float(slider0.val))
    #ax[0].imshow(gt[int(slider2.val)], cmap=cmap, alpha=float(slider1.val))
    #ax[0].imshow(gt_b[int(slider2.val)], cmap=cmap2)
    ax[0].set_title('CT + lungmask')
    ax[0].set_axis_off()

    ax[1].clear()
    ax[1].imshow(masked[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(image + " | " + str(curr_label) + " | " + str(ranks[int(slider2.val)]))
    ax[1].set_axis_off()

    ax[2].clear()
    ax[2].imshow(lungmask[int(slider2.val)], cmap="gray", vmin=0, vmax=1)
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


def import_set(sets, name, datasets_path, num=None):
    with h5py.File(datasets_path + "pytorch_dataset_" + name + ".h5", "r") as ff:
        tmp = np.array(ff[sets])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    if num != None:
        tmp = tmp[:num]
    return tmp


if __name__ == '__main__':

    # for inference with GPU
    CUDA = False and torch.cuda.is_available()
    torch.cuda.set_device(1)
    torch.manual_seed(1)
    if CUDA:
        torch.cuda.manual_seed(1)
        print('\nGPU is ON!')

    data_path = "/home/andrep/workspace/DeepMIL/data/"  # "/mnt/EncryptedPathology/DeepMIL/datasets/"
    datasets_path = "/home/andrep/workspace/DeepMIL/output/datasets/"
    save_model_path = "/home/andrep/workspace/DeepMIL/output/models/"
    configs_path = "/home/andrep/workspace/DeepMIL/output/configs/"

    curr_dataset = "020420_binary_healthy_cancer_shape_(128,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"

    # whether to make figure or not
    draw_flag = False
    mask_flag = False

    print("---")

    # preprocessed dataset params
    datagen_date = "060520"
    input_shape = (128, 256, 256)
    slab_shape = (128, 256, 256)
    negative_class = "healthy"
    positive_class = "sick-emphysema"
    hu_clip = [-1024, 1024]
    new_spacing = [1, 1, 2]
    CNN3D_flag = True
    MIL_type = 2

    # train params
    modelchoice = 1  # 1 : modified non-gated AttentionMIL
    batch_size = 16
    nepoch = 200
    split_val1 = 0.8
    split_val2 = 0.9
    lr = 1e-3  # 0.0005

    curr_date = "200620"
    curr_time = "145023"

    # current dataset path
    data_name = str(datagen_date) + "_binary_" + negative_class + "_" + positive_class + \
                "_input_" + str(input_shape).replace(" ", "") + \
                "_slab_" + str(slab_shape).replace(" ", "") + \
                "_huclip_" + str(hu_clip).replace(" ", "") + \
                "_spacing_" + str(new_spacing).replace(" ", "") + \
                "_3DCNN_" + str(CNN3D_flag) + \
                "_" + str(MIL_type) + "DMIL"
    #datapath += data_name + "/"

    # current model and dataset
    name = curr_date + "_" + curr_time + "_" + "binary_" + negative_class + "_" + positive_class
    #curr_dataset = datasets_path + curr_dataset + "/"

    # now init model and
    print('Init Model')
    if modelchoice == 1:
        model = AtnVGG()
    elif modelchoice == 2:
        model = GatedAttention()
    if CUDA:
        model.cuda()

    # set weights
    #model.load_state_dict(torch.load(save_model_path + "pytorch_model_" + name + ".pt"))
    #print(model)
    #exit()
    #model = torch.load(save_model_path + "pytorch_model_" + name + ".pt")
    model_stuff = torch.load(save_model_path + "pytorch_model_" + name + ".pt")
    model.load_state_dict(model_stuff)
    model.eval()  # inference mode

    all_sets = ["train", "val", "test"]
    #all_sets = ["val", "test"]
    #all_sets = ["val", "test"]

    for sets in all_sets:
        #sets = "test"
        curr_set = import_set(sets, name, datasets_path)
        curr_set = np.unique(curr_set)  # removes copies
        np.random.shuffle(curr_set)
        #print(curr_set)

        print(getClassDistribution(curr_set))

        num = 50  # -1  # 300

        gts = []
        preds = []

        print(sets)

        for curr_ct in tqdm(curr_set[:num], "CT:"):
            path = data_path + data_name + "/" + curr_ct + "/1.h5"

            image = curr_ct

            print(path)

            with h5py.File(path, "r") as f:
                data = np.array(f["data"])
                lungmask = np.array(f["lungmask"])
                gt = np.array(f["output"])[0]
                curr_label = gt

            if draw_flag:
                data_orig = data.copy()
                masked = data.copy()

            # mask data using lungmask
            if mask_flag:
                data[lungmask == 0] = 0
                masked = data.copy()

            # inference
            bag = np.expand_dims(data, axis=1)
            bag = torch.from_numpy(bag)

            #pred = model.forward(bag)

            Y_prob, Y_hat, A = model.forward(bag)
            error = Y_hat.eq(float(gt)).cpu().float().mean().item()

            Y_prob = np.squeeze(Y_prob.cpu().float().detach().numpy())
            Y_hat = int(np.squeeze(Y_hat.cpu().float().detach().numpy()))
            A = np.squeeze(A.cpu().float().detach().numpy())
            print(Y_prob)
            print(Y_hat)
            print(gt)
            #print(A)
            #print(len(A))

            gts.append(gt)
            preds.append(Y_hat)

        # classification report:
        report = classification_report(gts, preds)
        print(sets)
        print(report)
        print()


