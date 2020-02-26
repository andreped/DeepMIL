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
import h5py
from datetime import date
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback


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
    
    # current date
    curr_date = "_".join(date.today().strftime("%d/%m/%Y").split("/")[:2]) + "_"
    
    img_size = 256
    input_dim = (img_size, img_size, 1)
    weight_decay = 0.0005
    useGated = False
    lr = 5e-4
    batch_size = 1
    nb_classes = 2
    slices = 1
    window = 256

    name = curr_date + "binary_healthy_sick_cancer_" + str(img_size)

    # paths
    data_path = "/home/andrep/workspace/DeepMIL/data/260220_dim_2_binary_healthy_sick_lungmask_False/"
    save_model_path = '/home/andrep/workspace/DeepMIL/output/models/'
    history_path = '/home/andrep/workspace/DeepMIL/output/history/'
    datasets_path = '/home/andrep/workspace/DeepMIL/output/datasets/'

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

    # balance
    train_dir = upsample_balance(train_dir)
    val_dir = upsample_balance(val_dir)

    #print(test_dir)

    test_dir = [item for sublist in test_dir for item in sublist]
    val_dir = [item for sublist in val_dir for item in sublist]
    train_dir = [item for sublist in train_dir for item in sublist]

    # save random generated data sets
    f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
    f.create_dataset("test", data=np.array(test_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("val", data=np.array(val_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_dir).astype('S200'), compression="gzip", compression_opts=4)
    f.close()


    # define model
    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1 = Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv4 = MaxPooling2D((2, 2))(conv4)


    x = Flatten()(conv3)

    fc1 = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)

    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=useGated)(fc1)
    x_mul = multiply([alpha, fc1])

    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(x_mul)
    model = Model(inputs=[data_input], outputs=[out])

    # optimization setup
    model.compile(
        optimizer=Adam(lr=lr,beta_1=0.9, beta_2=0.999),
        loss=bag_loss,
        metrics=[bag_accuracy]
    )

    # augmentation
    bag_size = 50  # 8
    epochs = 1000

    train_aug = {} #{'flip': 1, 'rotate': 20, 'shift': int(np.round(window * 0.1))}  # , 'zoom':[0.75, 1.25]}
    val_aug = {}

    # define generators for sampling of data
    train_gen = batch_gen3(train_dir, batch_size=batch_size, aug=train_aug, epochs=epochs, nb_classes=nb_classes, input_shape=(slices, window, window, 1), data_path = data_path)
    val_gen = batch_gen3(val_dir, batch_size=batch_size, aug=val_aug, epochs=epochs, nb_classes=nb_classes, input_shape=(slices, window, window, 1), data_path = data_path)

    train_length = len(train_dir)
    val_length = len(val_dir)

    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',  # 'auto',
        period=1
    )


    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.losses.append(['loss', 'val_loss',
                                'acc', 'val_acc'])

        def on_epoch_end(self, batch, logs={}):
            self.losses.append([logs.get('loss'), logs.get('val_loss'),
                                logs.get('acc'), logs.get('val_acc')])
            # save history:
            f = h5py.File((history_path + 'history_' + name + '.h5'), 'w')
            f.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip",
                             compression_opts=4)
            f.close()


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



















