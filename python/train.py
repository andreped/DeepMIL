import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD,Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply, BatchNormalization
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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    #set_session(sess)  # set this TensorFlow session as the default session for Keras


    # current date
    curr_date = "_".join(date.today().strftime("%d/%m/%Y").split("/")[:2]) + "_"
    
    img_size = 256
    input_dim = (img_size, img_size, 1)
    weight_decay = 0 #0.0005 #0.0005
    useGated = False #False # Default: Frue
    lr = 1e-4 #5e-5
    batch_size = 1#16
    nb_classes = 2
    slices = 1
    window = 256
    lung_flag = False #True

    name = curr_date + "binary_healthy_sick_cancer_" + str(img_size)

    # paths
    data_path = "/home/andrep/workspace/DeepMIL/data/280220_dim_2_binary_healthy_sick_lungmask_" + str(lung_flag) + "/"
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
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D((2, 2))(conv4)
    
    conv5 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(weight_decay), activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = MaxPooling2D((2, 2))(conv5)

    x = Flatten()(conv5)
    
    # fully-connected layers
    fc1 = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), name='fc1')(x)
    #fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(64, activation='relu', kernel_regularizer=l2(weight_decay), name="fc2")(fc1)
    #fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(0.5)(fc2)

    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=useGated)(fc2) # fc1 or fc2?
    x_mul = multiply([alpha, fc2])

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
        save_weights_only=True, # <- DONT REALLY WANT THIS TO BE TRUE, BUT GETS PICKLE ISSUES(?)
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



















