import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD,Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
import numpy as np
from metrics import bag_accuracy, bag_loss
from custom_layers import Mil_Attention, Last_Sigmoid
from batch_generator import batch_gen3
import os
from math import ceil


def upsample_balance(tmps):
    new = [[], []]
    maxs = max([len(x) for x in tmps])
    for i, tmp in enumerate(tmps):
        curr_length = len(tmp)
        if curr_length == maxs:
            new_curr = tmp.copy()
        else:
            new_curr = np.tile(tmp, int(ceil(maxs / curr_length)))[:maxs]
        new[i] = new_curr
    return new


if __name__ == '__main__':


    # today's date
    dates = date.today()
    dates = dates.strftime("%d%m") + dates.strftime("%Y")[:2]

    # current model name
    name = dates + "_traditional_classification"

    # paths
    data_path = "/home/andrep/workspace/DeepMIL/data/190220_dim_2_binary_healthy_sick/"
    save_model_path = '/home/andrep/workspace/DeepMIL/output/models/'
    history_path = '/home/andrep/workspace/DeepMIL/output/history/'
    datasets_path = '/home/andrep/workspace/DeepMIL/output/datasets/'

    print("\n\n :) \n\n")

    # split data
    cts = os.listdir(data_path)

    # shuffle data
    np.random.shuffle(cts)

    # split data
    val = 0.8
    val2 = 0.9
    N = len(cts)
    train_dir = cts[:int(N*val)]
    val_dir = cts[int(N*val):int(N*val2)]
    test_dir = cts[:int(N*val3)]

    # save random generated data sets
    f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
    f.create_dataset("test", data=np.array(test_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("val", data=np.array(val_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.close()


    # pre-process data
    from tensorflow.python.keras.applications.inception_v3 import InceptionV3
    from tensorflow.python.keras.applications.resnet50 import ResNet50
    from tensorflow.python.keras.applications.vgg16 import VGG16
    from tensorflow.python.keras.applications.vgg19 import VGG19
    from tensorflow.python.keras.models import Input, Model, Sequential

    model = VGG19(include_top=False, weights='imagenet', input_shape=(512, 512, 1))

    # extract features using pre-trained encoder
    # for one image in CT
    features = model.predict(data.flatten())

