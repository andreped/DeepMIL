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
import tensorflow as tf
from gradcam import *
from scipy.ndimage import zoom
from tensorflow.keras.applications import imagenet_utils


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
    with h5py.File(datasets_path + "dataset_" + name + ".h5", "r") as f:
        tmp = np.array(f[sets])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    if num != None:
        tmp = tmp[:num]
    return tmp


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "-1"

    data_path = "/home/andrep/workspace/DeepMIL/data/"  # "/mnt/EncryptedPathology/DeepMIL/datasets/"
    datasets_path = "/home/andrep/workspace/DeepMIL/output/datasets/"
    save_model_path = "/home/andrep/workspace/DeepMIL/output/models/"
    configs_path = "/home/andrep/workspace/DeepMIL/output/configs/"

    #name = "260320_155100_binary_healthy_emphysema"
    #name = "280320_000256_binary_healthy_cancer"
    #curr_dataset = "250320_binary_healthy_emphysema_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"
    #curr_dataset = "270320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,1]_3DCNN"

    curr_dataset = "290320_binary_healthy_cancer_shape_(1,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    curr_dataset = "300320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    curr_dataset = "020420_binary_healthy_cancer_shape_(128,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    curr_dataset = "060520_binary_healthy_sick-emphysema_(128,256,26,"
    #name = "290320_012340_binary_healthy_cancer"
    name = "300320_020457_binary_healthy_cancer"
    name = "310320_181353_binary_healthy_cancer"
    name = "020420_144240_binary_healthy_cancer"
    name = "040420_181214_binary_healthy_cancer"
    name = "060520_170709_binary_healthy_sick-emphysema"

    #curr_dataset = "300320_binary_healthy_cancer_shape_(64,256,256)_huclip_[-1024,1024]_spacing_[1,1,2]_3DCNN"
    #name = "310320_025826_binary_healthy_cancer"
    #name = "310320_064512_binary_healthy_cancer"

    # whether to make figure or not
    draw_flag = True

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
    mask_flag = config["Design"]["mask_flag"]




    print("---")

    # Paths
    data_path = config["Paths"]["data_path"]  # "/home/andrep/workspace/DeepMIL/data/"
    save_model_path = config["Paths"]["save_model_path"]  # '/home/andrep/workspace/DeepMIL/output/models/'
    history_path = config["Paths"]["history_path"]  # '/home/andrep/workspace/DeepMIL/output/history/'
    datasets_path = config["Paths"]["datasets_path"]  # '/home/andrep/workspace/DeepMIL/output/datasets/'
    configs_path = config["Paths"]["configs_path"]  # '/home/andrep/workspace/DeepMIL/output/configs/'

    # Preprocessing
    input_shape = eval(config["Preprocessing"]["input_shape"])  # (128, 256, 256)
    slab_shape = eval(config["Preprocessing"]["slab_shape"])  # (16, 256, 256)
    nb_classes = int(config["Preprocessing"]["nb_classes"])  # 2
    classes = np.array(eval(config["Preprocessing"]["classes"]))  # [0, 1]
    new_spacing = eval(config["Preprocessing"]["new_spacing"])  # [1., 1., 2.]
    hu_clip = eval(config["Preprocessing"]["hu_clip"])  # [-1024, 1024]
    datagen_date = config["Preprocessing"]["datagen_date"]  # "040320"
    negative_class = config["Preprocessing"]["negative_class"]  # "healthy"
    positive_class = config["Preprocessing"]["positive_class"]  # "sick
    CNN3D_flag = eval(config["Preprocessing"]["CNN3D_flag"])
    MIL_type = eval(config["Preprocessing"]["MIL_type"])
    slices = int(config["Preprocessing"]["slices"])
    nb_features = int(config["Preprocessing"]["nb_features"])

    # Design
    val1 = float(config["Design"]["val1"])  # 0.8  # split train 80% of data
    val2 = float(config["Design"]["val2"])  # 0.9  # split val 90%-val1=90%-80%=10% of data -> remaining test
    mask_flag = eval(config["Design"]["mask_flag"])  # False # <- USE FALSE, SOMETHING WRONG WITH LUNGMASK (!)

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
    cnn_dropout = eval(config["Architecture"]["cnn_dropout"])

    # Training configs
    epochs = int(config["Training"]["epochs"])  # 200
    lr = eval(config["Training"]["lr"])  # 1e-3, 5e-5
    batch_size = int(config["Training"]["batch_size"])  # 64
    train_aug = eval(config["Training"]["train_aug"])  # {} # {'flip': 1, 'rotate': 20, 'shift': int(np.round(window * 0.1))}  # , 'zoom':[0.75, 1.25]}
    val_aug = eval(config["Training"]["val_aug"])  # {}
    # loss = config["Training"]["loss"]  # <- This should be set automatically given which model is chosen
    # metric = config["Training"]["metric"]  # <- This should be set automatically given which model is chosen

    # path to training data #  # "_binary_healthy_emphysema" + \
    data_name = datagen_date + "_binary_" + negative_class + "_" + positive_class + \
                "_input_" + str(input_shape).replace(" ", "") + \
                "_slab_" + str(slab_shape).replace(" ", "") + \
                "_huclip_" + str(hu_clip).replace(" ", "") + \
                "_spacing_" + str(new_spacing).replace(" ", "") + \
                "_3DCNN_" + str(CNN3D_flag) + \
                "_" + str(MIL_type) + "DMIL"
    # if model_type == "3DCNN" or model_type == "InceptionalMIL2D" or model_type == "2DMIL":
    # data_name += "_" + "3DCNN"  # str(model_type)
    data_path += data_name + "/"  # NOTE: Updates data_path here to the preprocessed data (!)

    curr_dataset = data_path



    print()
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
    elif model_type == "3DCNN":
        model = load_model(save_model_path + "model_" + name + ".h5")
    else:
        raise Exception("You fucked up...")

    print(model.summary())

    # load weights
    if not model_type == "3DCNN":
        model.load_weights(save_model_path + "model_" + name + ".h5")

    #model = load_model(save_model_path + "model_" + name + ".h5")

    all_sets = ["train", "val", "test"]
    #all_sets = ["val", "test"]
    all_sets = ["test"]

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
            path = curr_dataset + curr_ct + "/1.h5"

            image = curr_ct

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

            if model_type == "3DCNN":
                data_model_input = np.expand_dims(np.expand_dims(data, axis=0), axis=-1)
                softmax = np.squeeze(model.predict(data_model_input))
                pred = softmax
            elif model_type == "2DMIL":
                data_model_input = [(np.expand_dims(data, axis=-1))]
                softmax = np.squeeze(model.predict(data_model_input))
                pred = np.mean(softmax)
            #pred = np.argmax(softmax)
            print(softmax)

            #print(softmax)
            print(gt, pred)
            #print(gt, np.round(pred))

            pred = int(np.round(pred))
            print(gt, pred)
            print()


            #'''

            if draw_flag:

                # only draw if cancerous CT
                if gt == 0:
                    continue

                if model_type == "2DMIL":
                    # get alpha layer input to output function
                    ak = K.function([model.layers[0].input], [model.layers[-3].output])

                    # feed the sample in the function and get the result
                    ak_output = ak([(np.expand_dims(data, axis=-1))])
                    ak_output = np.array(ak_output[0])

                    #print(ak_output)

                    # rescale the weight as described in the paper
                    minimum = ak_output.min()
                    maximum = ak_output.max()
                    ak_output = (ak_output - minimum) / (maximum - minimum)

                    # rank on size
                    ranks = np.argsort(np.squeeze(ak_output))
                    #print(ranks)
                else:
                    ranks = list(range(data.shape[0]))

                if model_type == "3DCNN":
                    ## TODO: Test out DeepExplain for XAI stuff -> No. Use Grad-CAM instead
                    # initialize our gradient class activation map and build the heatmap
                    cam = GradCAM(model, pred)
                    heatmap = cam.compute_heatmap(data_model_input)

                    # resize the resulting heatmap to the original input image dimensions
                    data_shapes = data.shape
                    curr_shapes = heatmap.shape
                    heatmap = zoom(heatmap, [data_shapes[0] / curr_shapes[0],
                                             data_shapes[1] / curr_shapes[1],
                                             data_shapes[2] / curr_shapes[2]], order=1)

                    # and then overlay heatmap on top of the image
                    # heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
                    # (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

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

                s0ax = plt.axes([0.25, 0.14, 0.5, 0.03])
                slider0 = Slider(s0ax, 'XAI', 0, 1.0, dragging=True, valstep=0.05)

                s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
                slider1 = Slider(s1ax, 'alpha', 0, 1.0, dragging=True, valstep=0.05)

                s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])
                slider2 = Slider(s2ax, 'slice', 0, data_orig.shape[0] - 1, valstep=1, valfmt='%1d')

                # init
                slider0.set_val(0.5)
                slider1.set_val(0.3)
                slider2.set_val(0)
                f.subplots_adjust(bottom=0.15)

                slider0.on_changed(images)
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


