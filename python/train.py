import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD,Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten,\
                                           multiply, BatchNormalization, Conv3D, MaxPooling3D
import numpy as np
from metrics import bag_accuracy, bag_loss
from custom_layers import Mil_Attention, Last_Sigmoid
from batch_generator import batch_gen3
import os
from math import ceil
import h5py
from datetime import date
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
import shutil
from models import *


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


def getClassDistribution(x):
    cc = []
    for ll in x:
        cc.append(len(ll))
    return cc

#if __name__ == '__main__':  # <- TODO: (fix this!) currently just did this because I am obese AF and didn't bother to make actual input variables to the model functions above

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras


# current date
curr_date = "_".join(date.today().strftime("%d/%m/%Y").split("/")[:2]) + "_"

weight_decay = 0 #0.0005 #0.0005
useGated = False #False # Default: Frue
lr = 5e-5 #5e-5
batch_size = 1 #16
nb_classes = 2
slices = 1
#window = 256
mask_flag = True #True
epochs = 200
#bag_size = 50  # TODO: This is dynamic, which results in me being forced to use batch size = 1, fix this! I want both dynamic bag_size & bigger batch size

old_date = "040320"
input_shape = (1, 256, 256)
hu_clip = [-1024, 1024]
new_spacing = [1., 1., 2.]
test = 1

# augmentations
train_aug = {} #{'flip': 1, 'rotate': 20, 'shift': int(np.round(window * 0.1))}  # , 'zoom':[0.75, 1.25]}
val_aug = {}

data_name = old_date + "_binary_healthy_sick" + \
           "_shape_" + str(input_shape).replace(" ", "") + \
           "_huclip_" + str(hu_clip).replace(" ", "") + \
           "_spacing_" + str(new_spacing).replace(" ", "")

# paths
data_path = "/home/andrep/workspace/DeepMIL/data/"
save_model_path = '/home/andrep/workspace/DeepMIL/output/models/'
history_path = '/home/andrep/workspace/DeepMIL/output/history/'
datasets_path = '/home/andrep/workspace/DeepMIL/output/datasets/'

# path to training data
data_path += data_name + "/"

# name of output (to save everything as
name = data_name + \
    "_lr_" + str(lr) + \
    "_bs_" + str(batch_size) + \
    "_nb_classes_" + str(nb_classes) + \
    "_weight_decay_" + str(weight_decay) + \
    "_gated_" + str(useGated) + \
    "_mask_" + str(mask_flag) + \
    "_test_" + str(test) +\
    "_eps_" + str(epochs)

print("\n\n\n")

# prepare data -> balance classes, and CTs
classes = np.array([0, 1])

tmps = [[], []]
for path in os.listdir(data_path):
    tmps[int(path[0])].append(path)

# shuffle each class
for i, tmp in enumerate(tmps):
    np.random.shuffle(tmp)
    tmps[i] = tmp

# 80, 20 split => merge test and validation set
val = (0.8 * np.array([len(x) for x in tmps])).astype(int)
train_dir = [[], []]
val_dir = [[], []]
test_dir = [[], []]
for i, tmp in enumerate(tmps):
    length = len(tmp)
    val = int(0.8 * length)
    val2 = int(0.9 * length)
    train_dir[i] = tmp[:val]
    val_dir[i] = tmp[val:val2]
    test_dir[i] = tmp[val2:]

# merge val and test dirs
for i, c in enumerate(test_dir):
    val_dir[i] += c
test_dir = val_dir.copy()

# distribution before balancing
print("Class distribution on all sets before balancing: ")
print(getClassDistribution(train_dir))
print(getClassDistribution(val_dir))
print(getClassDistribution(test_dir))

# balance
train_dir = upsample_balance(train_dir)
val_dir = upsample_balance(val_dir)

#print(test_dir)

# sets
print("Sets (train, val, test): ")
print(train_dir)
print(val_dir)
print(test_dir)

# distribution after
print("Class distribution on all sets after balancing: ")
print(getClassDistribution(train_dir))
print(getClassDistribution(val_dir))
print(getClassDistribution(test_dir))

test_dir = [item for sublist in test_dir for item in sublist]
val_dir = [item for sublist in val_dir for item in sublist]
train_dir = [item for sublist in train_dir for item in sublist]

# save random generated data sets
if os.path.exists(datasets_path + 'dataset_' + name + '.h5'):
    os.remove(datasets_path + 'dataset_' + name + '.h5')

f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
f.create_dataset("test", data=np.array(test_dir).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("val", data=np.array(val_dir).astype('S200'), compression="gzip", compression_opts=4)
f.create_dataset("train", data=np.array(train_dir).astype('S200'), compression="gzip", compression_opts=4)
f.close()

print("Model name and configs: ")
print(name)

# define model
#model = model2D()
network = DeepMIL2D(input_shape=input_shape[1:] + (1,), nb_classes=2)
model = network.create()
print(model.summary())

# optimization setup
model.compile(
    optimizer=Adam(lr=lr,beta_1=0.9, beta_2=0.999),
    loss=bag_loss,
    metrics=[bag_accuracy]
)

# define generators for sampling of data
train_gen = batch_gen3(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs,
                       nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                       mask_flag=mask_flag)
val_gen = batch_gen3(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs,
                     nb_classes=nb_classes, input_shape=input_shape, data_path=data_path,
                     mask_flag=mask_flag)

train_length = len(train_dir)
val_length = len(val_dir)

save_best = ModelCheckpoint(
    save_model_path + 'model_' + name + '.h5',
    monitor='loss',  # TODO: WANTED TO MONITOR TRAIN LOSS TO STUDY OVERFITTING, BUT SHOULD HAVE || VAL_LOSS || !!!!!
    verbose=0,
    save_best_only=True,
    save_weights_only=True, # <-TODO: DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
    mode='auto',  # 'auto',
    period=1
)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses.append(['loss', 'val_loss',
                            'bag_accuracy', 'val_bag_accuracy'])

    def on_epoch_end(self, batch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('val_loss'),
                            logs.get('bag_accuracy'), logs.get('val_bag_accuracy')])
        # save history:
        ff = h5py.File((history_path + 'history_' + name + '.h5'), 'w')
        ff.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip", compression_opts=4)
        ff.close()


# make history logger (logs loss and specifiec metrics at each epoch)
history_log = LossHistory()

history = model.fit_generator(
    train_gen,
    steps_per_epoch=int(ceil(train_length / batch_size)),
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=int(ceil(val_length / batch_size)),
    callbacks=[save_best, history_log],
    use_multiprocessing=False,
    workers=1
)



















