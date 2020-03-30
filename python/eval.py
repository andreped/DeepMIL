import numpy as np
import os
import h5py
from tensorflow.python.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import configparser
from models import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import tensorflow.keras.backend as K


def getClassDistribution(tmp):
    cc = []
    classes = [int(x.split("_")[0]) for x in tmp]
    return [sum(classes == x) for x in np.unique(classes)]


def images(event):
    ax[0].clear()
    ax[0].imshow(data_orig[int(slider2.val)], cmap='gray', vmin=0, vmax=1)
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
    with h5py.File(datasets_path + "dataset_" + name + ".h5", "r") as f:
        tmp = np.array(f[sets])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    if num != None:
        tmp = tmp[:num]
    return tmp


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_path = "/mnt/EncryptedPathology/DeepMIL/datasets/"
    datasets_path = "/home/andrep/workspace/DeepMIL/output/datasets/"
    save_model_path = "/home/andrep/workspace/DeepMIL/output/models/"
    configs_path = "/home/andrep/workspace/DeepMIL/output/configs/"

    #name = "260320_155100_binary_healthy_emphysema"
    #name = "280320_000256_binary_healthy_cancer"
    #curr_dataset = "250320_binary_healthy_emphysema_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"
    #curr_dataset = "270320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"

    curr_dataset = "280320_binary_healthy_cancer_shape_(1,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    #name = "290320_012340_binary_healthy_cancer"
    name = "300320_020457_binary_healthy_cancer"

    # read and parse config file
    config = configparser.ConfigParser()
    config.read(configs_path + "config_" + name + ".ini")

    print("Model configs: ")
    for section in config.sections():
        print("\n--", section)
        for subsection in config[section]:
            print(subsection + " = " + config[section][subsection])

    # Preprocessing
    input_shape = eval(config["Preprocessing"]["input_shape"])  # (1, 256, 256)
    nb_classes = int(config["Preprocessing"]["nb_classes"])  # 2

    # Architecture
    valid_model_types = ["simple", "2DCNN", "2DMIL", "3DMIL", "2DFCN", "MLP", "3DCNN", "2DMIL_hybrid", "DeepFCNMIL", "InceptionalMIL2D"]  # TODO: This is not set by configFile
    model_type = config["Architecture"]["model_type"]
    convs = eval(config["Architecture"]["convs"])
    nb_dense_layers = int(config["Architecture"]["nb_dense_layers"])
    dense_val = int(config["Architecture"]["dense_val"])
    stride = int(config["Architecture"]["stride"])
    L_dim = int(config["Architecture"]["L_dim"])
    dense_dropout = eval(config["Architecture"]["dense_dropout"])
    spatial_dropout = eval(config["Architecture"]["spatial_dropout"])
    weight_decay = float(config["Architecture"]["weight_decay"])  # 0.0005 #0.0005
    useGated = eval(config["Architecture"]["use_gated"])  # False # Default: True
    bag_size = int(config["Architecture"]["bag_size"])
    # bag_size = 50  # TODO: This is dynamic, which results in me being forced to use batch size = 1, fix this! I want both dynamic bag_size & bigger batch size
    use_bn = eval(config["Architecture"]["use_bn"])

    print(model_type)

    if model_type == "VGGNet2D":
        network = VGGNet2D(input_shape=(*input_shape[1:], 1), nb_classes=nb_classes)
        network.nb_dense_layers = nb_dense_layers
        network.dense_size = dense_val
        network.L_dim = L_dim
        network.set_convolutions(convs)
        network.set_spatial_dropout(spatial_dropout)
        network.set_dense_dropout(dense_dropout)
        # network.set_stride = 1
        network.set_bn(use_bn)
        network.set_weight_decay(weight_decay)
        model = network.create()

    elif model_type == "2DMIL":  # TODO: Rename to AMIL
        network = DeepMIL2D(input_shape=input_shape[1:] + (1,), nb_classes=nb_classes)  # (1,), nb_classes=2)
        network.set_convolutions(convs)
        network.set_dense_size(dense_val)
        network.nb_dense_layers = nb_dense_layers
        network.set_dense_dropout(dense_dropout)
        network.set_spatial_dropout(spatial_dropout)
        network.set_stride = 1
        network.set_bn(use_bn)
        network.set_weight_decay(weight_decay)
        model = network.create()
    else:
        raise Exception("You fucked up...")

    print(model.summary())

    # load weights
    model.load_weights(save_model_path + "model_" + name + ".h5")

    #model = load_model(save_model_path + "model_" + name + ".h5")

    #all_sets = ["train", "val", "test"]
    all_sets = ["val", "test"]

    for sets in all_sets:
        #sets = "test"
        curr_set = import_set(sets, name, datasets_path)
        curr_set = np.unique(curr_set)  # removes copies
        np.random.shuffle(curr_set)
        #print(curr_set)

        print(getClassDistribution(curr_set))

        #continue

        num = 300

        gts = []
        preds = []

        print(sets)

        for curr_ct in tqdm(curr_set[:num], "CT:"):
            path = data_path + curr_dataset + "/" + curr_ct + "/1.h5"

            image = curr_ct

            with h5py.File(path, "r") as f:
                data = np.array(f["data"])
                lungmask = np.array(f["lungmask"])
                gt = np.array(f["output"])[0]

            # fix orientation
            data = np.flip(data, axis=2)
            data = np.flip(data, axis=1)

            curr_label = str(gt)

            #'''
            print(data.shape)

            data_orig = np.array(data)

            masked = data_orig.copy()
            masked[lungmask == 0] = 0
            #'''

            # mask data using lungmask
            #data[lungmask == 0] = 0

            if model_type == "3DCNN":
                softmax = np.squeeze(model.predict(np.expand_dims(np.expand_dims(data, axis=0), axis=-1)))
            elif model_type == "2DMIL":
                softmax = np.squeeze(model.predict([(np.expand_dims(data, axis=-1))]))
            #pred = np.argmax(softmax)
            pred = np.mean(softmax)

            #print(softmax)
            #print(gt, pred)
            #print(gt, np.round(pred))

            pred = int(np.round(pred))


            #'''
            # get alpha layer input to output function
            ak = K.function([model.layers[0].input], [model.layers[-3].output])

            # feed the sample in the function and get the result
            ak_output = ak([(np.expand_dims(data, axis=-1))])
            ak_output = np.array(ak_output[0])

            # rescale the weight as described in the paper
            minimum = ak_output.min()
            maximum = ak_output.max()
            ak_output = (ak_output - minimum) / (maximum - minimum)

            # rank on size
            ranks = np.argsort(np.squeeze(ak_output))

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

            gts.append(gt)
            preds.append(pred)

        # classification report:
        report = classification_report(gts, preds)
        print(sets)
        print(report)
        print()


