from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, \
    multiply, BatchNormalization, Conv3D, MaxPooling3D
import numpy as np
from metrics import bag_accuracy, bag_loss
from custom_layers import Mil_Attention, Last_Sigmoid
from batch_generator import batch_gen3
import os
from math import ceil
import h5py
from datetime import date, datetime
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
import shutil
from models import *
from batch_gen_features import *
import configparser
import sys
from timeit import default_timer as timer  # for timing stuff


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


def balance_amil(tmps, batch_size=1):
    tmps[0] = np.tile(tmps[0], 1)  # TODO: Is this correct? It makes sense for batch_size=2
    return tmps


def getClassDistribution(x):
    cc = []
    for ll in x:
        cc.append(len(ll))
    return cc


def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n


def mil_prediction(pred, n=1):
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[pred.shape[0] - n : pred.shape[0]]
    return (tf.gather(pred, i), i)


def step_bag_gradient(inputs, model):
    x, y = inputs

    with tf.GradientTape() as tape:
        logits = model(x, training=True)  # TODO: NO. tf.nn.softmax(logits) is here, not in model
        pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)  # n=1 # TODO: Only keep largest attention?
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(y, (1, 2)),  # TODO: Perhaps y is uint8 here (?)
            logits=tf.gather(logits, top_idx),
        )
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.trainable_variables)

    return grad, loss, pred

#@tf.function
#def train_step(inputs, model):


def step_bag_val(inputs, model):
    x, y = inputs

    logits = model(x, training=True)  # TODO: Do I want to have training=True here? It was set as True (as for train)
    pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)  # n=1 # TODO: Only keep largest attention?
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(y, (1, 2)),
        logits=tf.gather(logits, top_idx),
    )
    loss = tf.reduce_mean(loss)

    return loss, pred


def show_progbar(cur_step, num_instances, loss, acc, color_code, batch_size, time_per_step):
    TEMPLATE = "\r{}{}/{} [{:{}<{}}] - ETA: {}:{:02d} ({:>3.1f}s/step) - loss: {:>3.4f} - acc: {:>3.4f} \033[0;0m"

    progbar_length = 20

    curr_batch = cur_step  # int(cur_step // batch_size)
    nb_batches = int(num_instances // batch_size)
    ETA = (nb_batches - curr_batch) * time_per_step

    mins = int(ETA // 60)
    secs = int(np.round(ETA % 60))

    sys.stdout.write(TEMPLATE.format(
        color_code,
        curr_batch,
        nb_batches,
        "=" * min(int(progbar_length*(cur_step/nb_batches)), progbar_length),
        "-",
        progbar_length,
        mins,
        secs,
        time_per_step,
        loss,
        acc
    ))
    sys.stdout.flush()


def show_progbar_merged(cur_step, num_instances, loss, val_loss, acc, val_acc, color_code, batch_size, time_per_step, time_per_epoch):
    TEMPLATE = "\r{}{}/{} [{:{}<{}}] - {}:{:02d} ({:>3.1f}s/step) - loss: {:>3.4f} - acc: {:>3.4f} - val_loss: {:>3.4f} - val_acc: {:>3.4f}\033[0;0m"
    progbar_length = 20

    nb_batches = int(num_instances // batch_size)

    sys.stdout.write(TEMPLATE.format(
        color_code,
        cur_step,  # int(cur_step // batch_size),
        nb_batches,
        "=" * min(int(progbar_length*(cur_step/nb_batches)), progbar_length),
        "-",
        progbar_length,
        int(time_per_epoch // 60),
        int(np.round(time_per_epoch % 60)),
        time_per_step,
        loss,
        acc,
        val_loss,
        val_acc
    ))
    sys.stdout.flush()


#if __name__ == '__main__':

# read and parse training config file
config = configparser.ConfigParser()
config.read(sys.argv[1])

# whether to use GPU or not
GPU = config["GPU"]["useGPU"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # "0"

# check TF version
#v = tf.__version__.split(".")[:2]
#if tf[0] == '2' and int(tf[1]) > 1:
#    tf_version = "new"
#else:
#    tf_version = "old"


# TODO: CREATE DATA BY (from python/)
# source venv/bin/activate
# python ../NeuroDBUtils/main.py -c main_config.ini


# dynamically grow the memory used on the GPU (FOR TF==2.*)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#import tensorflow as tf
#configs = tf.ConfigProto()
#configs.gpu_options.allow_growth = True
#sess = tf.Session(config=configs)

if GPU == "-1":
    print("No GPU is being used...")
else:
    print("GPU in use: " + GPU)

# current date
curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
print("Today's date: ")
print(curr_date)

# current time
curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
print("Current time: ")
print(curr_time)

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
#loss = config["Training"]["loss"]  # <- This should be set automatically given which model is chosen
#metric = config["Training"]["metric"]  # <- This should be set automatically given which model is chosen

# path to training data #  # "_binary_healthy_emphysema" + \
data_name = datagen_date + "_binary_" + negative_class + "_" + positive_class + \
            "_input_" + str(input_shape).replace(" ", "") + \
            "_slab_" + str(slab_shape).replace(" ", "") + \
            "_huclip_" + str(hu_clip).replace(" ", "") + \
            "_spacing_" + str(new_spacing).replace(" ", "") +\
            "_3DCNN_" + str(CNN3D_flag) +\
            "_" + str(MIL_type) + "DMIL"
#if model_type == "3DCNN" or model_type == "InceptionalMIL2D" or model_type == "2DMIL":
#data_name += "_" + "3DCNN"  # str(model_type)
data_path += data_name + "/"  # NOTE: Updates data_path here to the preprocessed data (!)

# name of output (to save everything as
name = curr_date + "_" + curr_time + "_" + "binary_" + negative_class + "_" + positive_class  # healthy_cancer"  # "binary_healthy_emphysema"

# save current configuration file with all corresponding data
config_out_path = configs_path + "config_" + name + ".ini"
shutil.copyfile(sys.argv[1], config_out_path)


## prepare data -> balance classes, and CTs
tmps = [[], []]
for path in os.listdir(data_path):
    tmps[int(path[0])].append(path)

# shuffle each class
for i, tmp in enumerate(tmps):
    np.random.shuffle(tmp)
    tmps[i] = tmp

# 80, 20 split => merge test and validation set
val = (val1 * np.array([len(x) for x in tmps])).astype(int)
train_dir = [[], []]
val_dir = [[], []]
test_dir = [[], []]
for i, tmp in enumerate(tmps):
    length = len(tmp)
    val = int(val1 * length)
    val2 = int(val2 * length)
    train_dir[i] = tmp[:val]
    val_dir[i] = tmp[val:val2]
    test_dir[i] = tmp[val2:]

# merge val and test dirs  # TODO: Need to merge for cancer set as less samples for one class than the other
for i, c in enumerate(test_dir):
    val_dir[i] += c
test_dir = val_dir.copy()

# distribution before balancing
print("Class distribution on all sets before balancing: ")
print(getClassDistribution(train_dir))
print(getClassDistribution(val_dir))
print(getClassDistribution(test_dir))

# balance classes
train_dir = upsample_balance(train_dir)
val_dir = upsample_balance(val_dir)

# special balancing for AMIL, upweights negative class based on batch size
# TODO: This isn't good enough for large batch size (!)
#if "AMIL" in model_type:
#    train_dir = balance_amil(train_dir, batch_size)
#    val_dir = balance_amil(val_dir, batch_size)
#    #exit()

# for AMIL random sampling
train_dirs = train_dir.copy()
val_dirs = val_dir.copy()

# distribution after
print("Class distribution on all sets after balancing: ")
print(getClassDistribution(train_dir))
print(getClassDistribution(val_dir))
print(getClassDistribution(test_dir))

test_dir = [item for sublist in test_dir for item in sublist]
val_dir = [item for sublist in val_dir for item in sublist]
train_dir = [item for sublist in train_dir for item in sublist]

# hematoma MIL stuff
CONVERGENCE_EPOCH_LIMIT = 50
epsilon = 1e-4

#train_dir = train_dir[:5]
#val_dir = val_dir[:5]

# save random generated data sets
if os.path.exists(datasets_path + 'dataset_' + name + '.h5'):
    os.remove(datasets_path + 'dataset_' + name + '.h5')

f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
f.create_dataset("test", data=np.array(test_dir).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("val", data=np.array(val_dir).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("train", data=np.array(train_dir).astype('S200'), compression="gzip", compression_opts=4)
f.close()

print("Model name: ")
print(name)

print("Model configs: ")
for section in config.sections():
    print("\n--", section)
    for subsection in config[section]:
        print(subsection + " = " + config[section][subsection])
print("\n ----")

# close config file
del config  # TODO: Is this the best way? Didn't find any close-method, so I just deleted the variable -> 2sneaky?

print("\n\n\n TRIED SHUFFLING STUFF \n\n\n")

### This is where the fun begins


#if not "hema" in model_type: # TODO: THIS SHOULDN'T BE NECESSARY (!)
if not (("HMIL" in model_type) or (model_type == "VGGNet2D")):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()  # TODO: Don't use this with hematomaMIL (!)

# define model
if model_type == "simple":
    model = model2D()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr),  # , beta_1=0.9, beta_2=0.999), # <- Default params
        loss=bag_loss,
        metrics=[bag_accuracy]
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_bag_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "2DAMIL":  # renames from 2DMIL -> 2DAMIL
    print(input_shape[1:])
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

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr),  # , beta_1=0.9, beta_2=0.999), # <- Default params
        loss=bag_loss,  # bag_loss
        metrics=[bag_accuracy]  # [bag_accuracy]
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_bag_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "3DAMIL":  # renames from 2DMIL -> 2DAMIL
    print(input_shape[1:])
    print(input_shape)
    print(slab_shape)
    print(input_shape[0]/slab_shape[0])
    print((int(input_shape[0]/slab_shape[0]), ) + slab_shape + (1,))
    network = DeepMIL3D(input_shape=slab_shape + (1,), nb_classes=nb_classes)  # (1,), nb_classes=2)
    network.set_convolutions(convs)
    network.set_dense_size(dense_val)
    network.nb_dense_layers = nb_dense_layers
    network.set_dense_dropout(dense_dropout)
    network.set_spatial_dropout(spatial_dropout)
    network.set_stride = 1
    network.set_bn(use_bn)
    network.set_weight_decay(weight_decay)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr),  # , beta_1=0.9, beta_2=0.999), # <- Default params
        loss=bag_loss,  # bag_loss
        metrics=[bag_accuracy]  # [bag_accuracy]
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_bag_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "2DMIL_hybrid":
    network = DeepMIL2D_hybrid(input_shape=(bag_size,) + input_shape[1:] + (1,), nb_classes=2)  # (1,), nb_classes=2)
    network.set_convolutions(convs)
    network.set_dense_dropout(dense_dropout)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
        loss=bag_loss,
        metrics=[bag_accuracy]
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_bag_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "FCNMIL":
    model = DeepFCNMIL(input_shape=(bag_size,) + input_shape[1:], nb_classes=2)

    # optimization setup
    model.compile(
        optimizer='adadelta',
        loss='sparse_categorical_crossentropy',  # TODO: binary_crossentropy or sparse_categorical_crossentropy (?)
        metrics=['accuracy']
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "Inceptional2DMIL":
    print(input_shape)
    network = InceptionalMIL2D(input_shape=input_shape + (3,), nb_classes=2)
    network.set_dense_size(dense_val)
    network.nb_dense_layers = nb_dense_layers
    network.set_dense_dropout(dense_dropout)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr),  # , beta_1=0.9, beta_2=0.999), # <- Default params
        loss=bag_loss,
        metrics=[bag_accuracy]
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_bag_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )
elif model_type == "2DCNN":
    network = Benchline3DCNN(input_shape=input_shape[1:] + (bag_size,), nb_classes=nb_classes)
    network.nb_dense_layers = nb_dense_layers
    network.dense_size = dense_val
    network.L_dim = L_dim
    network.set_convolutions(convs)
    network.set_dense_dropout(dense_dropout)
    network.set_spatial_dropout(spatial_dropout)
    network.set_bn(use_bn)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',  # 'binary_crossentropy',
        metrics=['accuracy']  # ['accuracy']
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "2DFCN":
    network = Benchline3DFCN(input_shape=input_shape[1:] + (bag_size,), nb_classes=2)
    network.nb_dense_layers = nb_dense_layers
    network.dense_size = dense_val
    network.L_dim = L_dim
    network.set_convolutions(convs)
    network.set_dense_dropout(dense_dropout)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',  # 'binary_crossentropy',
        metrics=['accuracy']  # ['accuracy']
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )
elif model_type == "MLP":
    network = MLP(input_shape=(int(bag_size * nb_features),), nb_classes=2)
    network.nb_dense_layers = nb_dense_layers
    network.dense_size = dense_val
    network.set_dense_dropout(dense_dropout)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',  # 'binary_crossentropy',
        metrics=['accuracy']  # ['accuracy']
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,  # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
        mode='max',  # 'auto',
        period=1
    )

elif model_type == "3DCNN":
    print(input_shape)
    print((*input_shape, 1))

    network = CNN3D(input_shape=(*input_shape, 1), nb_classes=2)
    network.nb_dense_layers = nb_dense_layers
    network.dense_size = dense_val
    network.L_dim = L_dim
    network.set_convolutions(convs)
    network.set_spatial_dropout(spatial_dropout)
    network.set_dense_dropout(dense_dropout)
    network.set_stride(stride)
    network.set_bn(use_bn)
    network.set_weight_decay(weight_decay)
    network.set_cnn_dropout(cnn_dropout)
    model = network.create()

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr),  # beta_1=0.9, beta_2=0.999),
        loss='binary_crossentropy',  # 'binary_crossentropy',
        metrics=['accuracy']  # ['accuracy']
    )

    # when and how to save model
    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',  # 'auto',
        period=1
    )

elif model_type == "2DHMIL":
    model = resnet(input_shape=(*input_shape[1:], 1), num_classes=nb_classes, ds=2)  # TODO: NOTE THIS HAS BEEN CHANGED!!!

elif model_type == "VGGNet2D":
    network = VGGNet2D(input_shape=(*input_shape[1:], 1), nb_classes=nb_classes)
    network.nb_dense_layers = nb_dense_layers
    network.dense_size = dense_val
    network.L_dim = L_dim
    network.set_convolutions(convs)
    network.set_spatial_dropout(spatial_dropout)
    network.set_dense_dropout(dense_dropout)
    #network.set_stride = 1
    network.set_bn(use_bn)
    network.set_weight_decay(weight_decay)
    model = network.create()

else:
    raise ("Please choose a valid model type among these options: " + str(valid_model_types))

print(model.summary())

# define generators for sampling of data
if model_type == "3DAMIL":
    print()
    print(input_shape)
    print(slab_shape)
    print()
    from batch_generator_amil import batch_gen3_amil
    #print(train_dirs)
    #print("\n", train_dirs[0], len(train_dirs[0]))
    #print("\n", train_dirs[1], len(train_dirs[1]))
    #exit()
    train_gen = batch_gen3_amil(train_dirs, N=int(np.round((len(train_dirs[0]) * 2) / batch_size)), batch_size=batch_size, aug=train_aug, epochs=epochs,
                           nb_classes=nb_classes, input_shape=input_shape, slab_shape=slab_shape, data_path=data_path,
                           mask_flag=mask_flag, bag_size=bag_size, model_type=model_type)
    val_gen = batch_gen3_amil(val_dirs, N=int(np.round((len(val_dirs[0]) * 2) / batch_size)), batch_size=batch_size, aug=val_aug, epochs=epochs,
                         nb_classes=nb_classes, input_shape=input_shape, slab_shape=slab_shape, data_path=data_path,
                         mask_flag=mask_flag, bag_size=bag_size, model_type=model_type)
elif model_type == "MLP":
    train_gen = batch_gen_features3(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs,
                                    nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                                    mask_flag=mask_flag, bag_size=bag_size)
    val_gen = batch_gen_features3(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs,
                                  nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                                  mask_flag=mask_flag, bag_size=bag_size)
else:
    train_gen = batch_gen3(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs,
                           nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                           mask_flag=mask_flag, bag_size=bag_size, model_type=model_type)
    val_gen = batch_gen3(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs,
                         nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                         mask_flag=mask_flag, bag_size=bag_size, model_type=model_type)

if model_type == "2DHMIL" or model_type == "VGGNet2D":  # "hema" in model_type:
    # metrics
    train_accuracy = tf.keras.metrics.Accuracy(name='train_acc')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    opt = tf.optimizers.Adam(lr=lr)  # tf.optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=False)
    #opt = tf.optimizers.Adadelta()

    ######### TRAINING #########
    best_val_loss = 100000
    best_val_acc = 0
    convergence_epoch_counter = 0
    nb_instances_mov_avg = 5
    train_color_code = "\033[0;0m"  # "\033[0;32m"
    val_color_code = "\033[0;0m"  # "\033[0;36m"

    grads = [tf.zeros_like(l) for l in model.trainable_variables]

    train_n = len(train_dir)
    train_steps = train_n/batch_size
    val_n = len(val_dir)
    val_steps = val_n/batch_size

    best_epoch = 1
    for cur_epoch in range(epochs):

        time_list_train = []

        # epoch start time
        epoch_start = timer()

        # curr time for train
        curr_time = timer()

        # new line for each epoch
        print("\n")
        print("Epoch %d/%d" % (cur_epoch + 1, epochs))

        stop_flag = False

        #train_dataset = train_dataset.take(num_instances["train_{}".format(cur_fold)]).shuffle(BUFFER_SIZE)

        for j, (x, y) in enumerate(train_gen):
            if stop_flag:
                break
            for i, (x_curr, y_curr) in enumerate(zip(x, y)):
                y_curr = tf.convert_to_tensor([int(y_curr[0])])  # FIXME (?)
                y_curr = tf.one_hot(y_curr, 2)
                grad, loss, pred = step_bag_gradient((x_curr, y_curr), model)
                for g in range(len(grads)):
                    grads[g] = running_average(grads[g], grad[g], i + 1)

                # TODO: CHECK IF THE PROBLEM IS WITH SOFTMAX OR SIMILAR
                train_accuracy.update_state(
                    tf.argmax(y_curr, axis=1),
                    tf.argmax(pred, axis=1),
                )
                train_loss.update_state(loss)

                if ((i+1)%batch_size==0) or (j+1==int(train_steps)):
                    opt.apply_gradients(zip(grads, model.trainable_variables))

                    time_list_train.append(timer() - curr_time)
                    curr_time = timer()
                    time_avg = np.mean(time_list_train[::-1][:nb_instances_mov_avg])  # average across k last

                    show_progbar(
                        (j + 1),
                        train_n,
                        train_loss.result(),
                        train_accuracy.result(),
                        train_color_code,
                        batch_size,
                        time_avg
                    )

                if (j+1 == int(train_steps)):
                    stop_flag = True
                    break

        # epoch start time for validation set
        time_list_val = []
        curr_time = timer()

        final_j = j
        stop_flag = False

        # validation metrics
        #print("\n")  # {}Validating...\033[0;0m".format(val_color_code))
        for j, (x, y) in enumerate(val_gen):
            if stop_flag:
                break
            for i, (x_curr, y_curr) in enumerate(zip(x, y)):
                y_curr = tf.convert_to_tensor([int(y_curr[0])])
                y_curr = tf.one_hot(y_curr, 2)
                loss, pred = step_bag_val((x_curr, y_curr), model)

                val_accuracy.update_state(
                    tf.argmax(y_curr, axis=1),
                    tf.argmax(pred, axis=1),
                )
                val_loss.update_state(loss)

                if ((i+1)%batch_size==0) or (j+1 == int(val_steps)):

                    time_list_val.append(timer() - curr_time)
                    curr_time = timer()
                    time_avg = np.mean(time_list_val[::-1][:nb_instances_mov_avg])  # average across k last

                    show_progbar(
                        (j + 1),
                        val_n,
                        val_loss.result(),
                        val_accuracy.result(),
                        val_color_code,
                        batch_size,
                        time_avg
                    )

                if (j+1 == int(val_steps)):
                    stop_flag = True
                    break

        # epoch end time
        epoch_end = timer()

        # remove and merge the progbars into one
        show_progbar_merged(
            (final_j + 1),
            train_n,
            train_loss.result(),
            val_loss.result(),
            train_accuracy.result(),
            val_accuracy.result(),
            "\033[0;0m",
            batch_size,
            np.mean(time_list_train),
            epoch_end - epoch_start
        )



        with open(history_path + 'history_' + name + '.txt', 'a') as f:
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                cur_epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                val_loss.result(),
                val_accuracy.result(),
            ))


        if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
            print("\nCurrent Fold: {}\
                    \nNo improvement in {} epochs, model has converged.\
                    \nModel achieved best val loss at epoch {}.\
                    \nTrain Loss: {:.4f} Train Acc: {:.2%}\
                    \nVal   Loss: {:.4f} Val   Acc: {:.2%}".format(
                cur_fold,
                CONVERGENCE_EPOCH_LIMIT,
                best_epoch,
                train_loss.result(),
                train_accuracy.result(),
                val_loss.result(),
                val_accuracy.result(),
            ))
            break

        #if val_loss.result() > best_val_loss and\
        #        np.abs(val_loss.result() - best_val_loss) > epsilon:
        if val_accuracy.result() < best_val_acc and\
                np.abs(val_accuracy.result() - best_val_acc) > epsilon:
            convergence_epoch_counter += 1
        else:
            convergence_epoch_counter = 0

        #if val_loss.result() < best_val_loss:
        if val_accuracy.result() > best_val_acc:  # Save on best validation accuracy instead
            best_epoch = cur_epoch + 1
            best_val_loss = val_loss.result()
            best_val_acc = val_accuracy.result()
            model.save_weights(
                save_model_path + 'model_' + name + '.h5'
            )

else:
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            if "MIL" in model_type:
                self.losses.append(['loss', 'val_loss',
                                    'bag_accuracy', 'val_bag_accuracy'])
            else:
                self.losses.append(['loss', 'val_loss',
                                    'acc', 'val_acc'])

        def on_epoch_end(self, batch, logs={}):
            if "MIL" in model_type:
                self.losses.append([logs.get('loss'), logs.get('val_loss'),
                                    logs.get('bag_accuracy'), logs.get('val_bag_accuracy')])
            else:
                self.losses.append([logs.get('loss'), logs.get('val_loss'),
                                    logs.get('acc'), logs.get('val_acc')])
            # save history:
            ff = h5py.File((history_path + 'history_' + name + '.h5'), 'w')
            ff.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip", compression_opts=4)
            ff.close()


    # make history logger (logs loss and specifiec metrics at each epoch)
    history_log = LossHistory()

    if "AMIL" in model_type:
        w = 2**batch_size
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=int(ceil(len(train_dir) / batch_size)),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=int(ceil(len(val_dir) / batch_size)),
            callbacks=[save_best, history_log],
            use_multiprocessing=False,
            workers=1,
            #class_weight={0: (1 - w), 1: w},  # TODO: (NOT SUPPORTED IN AMIL!) Tried to add weights to the loss and accuracy
        )
    else:
        history = model.fit_generator(
            train_gen,
            steps_per_epoch=int(ceil(len(train_dir) / batch_size)),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=int(ceil(len(val_dir) / batch_size)),
            callbacks=[save_best, history_log],
            use_multiprocessing=False,
            workers=1
        )
