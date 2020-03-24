import numpy as np
import os
import sys
import json

from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from utils.augmentations import *
from utils.tfrecord_utils import *
from utils.pad import *
from models_resnet.resnet import *

from timeit import default_timer as timer  # for timing stuff

# for float16 optimization
from tensorflow.keras import backend as K
K.set_floatx('float32')  # 'float16')
K.set_epsilon(1e-7)  # 1e-4)

#from time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def mil_prediction(pred, n=1):
    i = tf.argsort(pred[:, 1], axis=0)
    i = i[len(pred) - n : len(pred)]
    return (tf.gather(pred, i), i)

def show_progbar(cur_step, num_instances, loss, acc, color_code, batch_size, time_per_step):
    TEMPLATE = "\r{}{}/{} [{:{}<{}}] - ETA: {}:{:02d} ({:>3.1f}s/step) - loss: {:>3.4f} - acc: {:>3.4f} \033[0;0m"

    progbar_length = 20

    curr_batch = int(cur_step // batch_size)
    nb_batches = int(np.ceil(num_instances / batch_size))
    ETA = (nb_batches - curr_batch) * time_per_step

    sys.stdout.write(TEMPLATE.format(
        color_code,
        curr_batch,
        nb_batches,
        "=" * min(int(progbar_length*(cur_step/num_instances)), progbar_length),
        "-",
        progbar_length,
        int(ETA // 60),
        int(np.round(ETA % 60)),
        time_per_step,
        loss,
        acc
    ))
    sys.stdout.flush()


def show_progbar_merged(cur_step, num_instances, loss, val_loss, acc, val_acc, color_code, batch_size, time_per_step, time_per_epoch):
    TEMPLATE = "\r{}{}/{} [{:{}<{}}] - {}:{:02d} ({:>3.1f}s/step) - loss: {:>3.4f} - acc: {:>3.4f} - val_loss: {:>3.4f} - val_acc: {:>3.4f}\033[0;0m"
    progbar_length = 20

    sys.stdout.write(TEMPLATE.format(
        color_code,
        int(cur_step // batch_size),
        int(np.ceil(num_instances / batch_size)),
        "=" * min(int(progbar_length*(cur_step/num_instances)), progbar_length),
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


def step_bag_gradient(inputs, model):
    x, y = inputs

    with tf.GradientTape() as tape:
        logits = model(x, training=True)  # TODO: NO. tf.nn.softmax(logits) is here, not in model
        pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)  # n=1 # TODO: Only keep largest attention?
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(tf.tile(y, [1]), (1, len(y))),  # TODO: Perhaps y is uint8 here (?)
            logits=tf.gather(logits, top_idx),
        )
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.trainable_variables)

    return grad, loss, pred

def step_bag_val(inputs, model):
    x, y = inputs

    logits = model(x, training=True)  # TODO: Do I want to have training=True here? It was set as True (as for train)
    pred, top_idx = mil_prediction(tf.nn.softmax(logits), n=1)  # n=1 # TODO: Only keep largest attention?
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(tf.tile(y, [1]), (1, len(y))),
        logits=tf.gather(logits, top_idx),
    )
    loss = tf.reduce_mean(loss)

    return loss, pred



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
                ("Missing cmd line arguments; please first run `make_tfrecord.py`\n")
                ("First argument: path to TFRecord directory")
             )
        sys.exit()

    # whether to use GPU or not
    GPU = "0"  # "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # "0"

    # dynamically grow the memory used on the GPU (FOR TF==2.*)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
        #tf.config.gpu.set_per_process_memory_fraction(device, 0.75) # <- use fraction of GPU instead

    ########## HYPERPARAMETER SETUP ##########

    N_EPOCHS = 200  # 10000
    BATCH_SIZE = 1  # 2**7
    BUFFER_SIZE = 2**2
    ds = 2  # 4
    instance_size = (512, 512)  # (256, 256) # TODO: BUG HERE SOMEWHERE? SOMETHING HARDCODED?
    num_classes = 2
    learning_rate = 1e-4  # 1e-4
    train_color_code = "\033[0;0m"  # "\033[0;32m"
    val_color_code = "\033[0;0m"  # "\033[0;36m"
    CONVERGENCE_EPOCH_LIMIT = 50
    epsilon = 1e-4
    nb_instances_mov_avg = 5

    ########## DIRECTORY SETUP ##########


    MODEL_NAME = "resnetish_ds_{}".format(ds)
    WEIGHT_DIR = Path("models_resnet/weights") / MODEL_NAME
    RESULTS_DIR = Path("results") / MODEL_NAME
    DATA_DIR = Path(sys.argv[1])
    NUM_INSTANCES_FILE = DATA_DIR / "count.txt"
    with open(NUM_INSTANCES_FILE, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    num_instances = {}
    for l in lines:
        cur_name, cur_fold, n = l.split()
        num_instances["{}_{}".format(cur_name, cur_fold)] = int(n)

    # files and paths
    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=Path('.'))

    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    HISTORY_PATH = WEIGHT_DIR / (MODEL_NAME + "_history.json")


    # TODO: WHETHER TO USE PRE-TRAINED RESNET OR TRAIN ENCODER-CLASSIFIER-NET FROM SCRATCH
    model_type = "resnetish"  # "2DFCN"  # "resnetish"  # TODO: I get OOM using 2DFCN

    # Actual instantiation happens for each fold
    if model_type == "resnetish":
        model = resnet(input_shape=(*instance_size, 1), num_classes=num_classes, ds=ds)  # TODO: NOTE THIS HAS BEEN CHANGED!!!
    elif model_type == "2DFCN":  # TODO: Doesnt work. Does network have to be FCNs for these to be used in pipeline?
        import sys
        sys.path.append("/home/andrep/workspace/DeepMIL/python")
        from models import Benchline3DFCN
        nb_dense_layers = 1
        dense_val = 128
        convs = [16, 32, 32, 64, 64, 128, 128]
        dense_dropout = 0

        print((*instance_size, 1))
        network = Benchline3DFCN(input_shape=(*instance_size, 1), nb_classes=num_classes)
        network.nb_dense_layers = nb_dense_layers
        network.dense_size = dense_val
        network.set_convolutions(convs)
        network.set_dense_dropout(dense_dropout)
        network.set_final_dense(num_classes)
        network.set_use_bn(False)
        model = network.create()

    INIT_WEIGHT_PATH = WEIGHT_DIR / "init_weights.h5"
    model.save_weights(str(INIT_WEIGHT_PATH))
    json_string = model.to_json()
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(json_string, f)

    print(model.summary())  # line_length=75))

    ######### FIVE FOLD CROSS VALIDATION #########

    TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve_fold_{}.csv"

    TRAIN_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_train.tfrecord"
    VAL_TF_RECORD_FILENAME = DATA_DIR / "dataset_fold_{}_val.tfrecord"

    for cur_fold in range(5):

        with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        ######### MODEL AND CALLBACKS #########
        model.load_weights(str(INIT_WEIGHT_PATH))

        # training ADAM with float16 precision (mixed)
        #tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        opt = tf.optimizers.Adam(learning_rate=learning_rate)  #, epsilon=1e-4)  # TODO: Need a slightly larger epsilon for it to work, default was not set here
        #opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        #opt = tf.optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=False)


        ######### DATA IMPORT #########
        train_dataset = tf.data.TFRecordDataset(
                str(TRAIN_TF_RECORD_FILENAME).format(cur_fold))
        train_dataset = train_dataset.map(lambda record: parse_bag(record,instance_size,num_labels=num_classes))
        train_dataset = train_dataset.take(num_instances["train_{}".format(cur_fold)]).shuffle(BUFFER_SIZE)


        val_dataset = tf.data.TFRecordDataset(
                str(VAL_TF_RECORD_FILENAME).format(cur_fold))
        val_dataset = val_dataset.map(lambda record: parse_bag(
                record,
                instance_size,
                num_labels=num_classes))
        """
        augmentations = [flip_dim1, flip_dim2]  # Removed rotate2d
        for f in augmentations:
            train_dataset = train_dataset.map(
                lambda x, y:
                tf.cond(tf.random.uniform([], 0, 1) > 0.9,  # with 90% chance, call first `lambda`:
                        lambda: (f(x), y),  # apply augmentation `f`, don't touch `y`
                        lambda: (x, y),     # don't apply any aug
                        ), num_parallel_calls=4,
                )
        """
        # metrics
        train_accuracy = tf.keras.metrics.Accuracy(name='train_acc')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        ######### TRAINING #########
        best_val_loss = 100000
        best_val_acc = 0
        convergence_epoch_counter = 0

        grads = [tf.zeros_like(l) for l in model.trainable_variables]

        train_n = num_instances["train_{}".format(cur_fold)]
        val_n = num_instances["val_{}".format(cur_fold)]

        best_epoch = 1
        for cur_epoch in range(N_EPOCHS):

            time_list_train = []

            # epoch start time
            epoch_start = timer()

            # curr time for train
            curr_time = timer()

            # new line for each epoch
            print("\n")
            print("Epoch %d/%d" % (cur_epoch + 1, N_EPOCHS))

            # for each epoch, shuffle dataset  # TODO: Is this necessary? Maybe tfrecord does this in the background
            train_dataset = train_dataset.take(num_instances["train_{}".format(cur_fold)]).shuffle(BUFFER_SIZE)

            #print("\n")  # {}Training...\033[0;0m".format(train_color_code))
            for i, (x, y) in enumerate(train_dataset):

                #print(x.shape)
                #print(x)
                grad, loss, pred = step_bag_gradient((x,y), model)
                for g in range(len(grads)):
                    grads[g] = running_average(grads[g], grad[g], i + 1)

                # TODO: CHECK IF THE PROBLEM IS WITH SOFTMAX OR SIMILAR
                #print()
                #print(y, pred)
                #print(tf.argmax(tf.convert_to_tensor([y]), axis=1), tf.argmax(pred, axis=1))
                train_accuracy.update_state(
                    tf.argmax(tf.convert_to_tensor([y]), axis=1),
                    tf.argmax(pred, axis=1),
                )
                train_loss.update_state(loss)

                if (i+1)%BATCH_SIZE==0 or (i+1)==train_n:
                    opt.apply_gradients(zip(grads, model.trainable_variables))

                    time_list_train.append(timer() - curr_time)
                    curr_time = timer()
                    time_avg = np.mean(time_list_train[::-1][:nb_instances_mov_avg])  # average across k last

                    show_progbar(
                        (i + 1),
                        train_n,
                        train_loss.result(),
                        train_accuracy.result(),
                        train_color_code,
                        BATCH_SIZE,
                        time_avg
                    )
            final_i = i

            # epoch start time for validation set
            time_list_val = []
            curr_time = timer()

            # validation metrics
            #print("\n")  # {}Validating...\033[0;0m".format(val_color_code))
            for i, (x, y) in enumerate(val_dataset):

                loss, pred = step_bag_val((x,y), model)

                val_accuracy.update_state(
                    tf.argmax(tf.convert_to_tensor([y]), axis=1),
                    tf.argmax(pred, axis=1),
                )
                val_loss.update_state(loss)

                if (i+1)%BATCH_SIZE==0 or (i+1)==val_n:

                    time_list_val.append(timer() - curr_time)
                    curr_time = timer()
                    time_avg = np.mean(time_list_val[::-1][:nb_instances_mov_avg])  # average across k last

                    show_progbar(
                        (i + 1),
                        val_n,
                        val_loss.result(),
                        val_accuracy.result(),
                        val_color_code,
                        BATCH_SIZE,
                        time_avg
                    )

            # epoch end time
            epoch_end = timer()

            # remove and merge the progbars into one
            show_progbar_merged(
                (final_i + 1),
                train_n,
                train_loss.result(),
                val_loss.result(),
                train_accuracy.result(),
                val_accuracy.result(),
                "\033[0;0m",
                BATCH_SIZE,
                np.mean(time_list_train),
                epoch_end - epoch_start
            )



            with open(str(TRAIN_CURVE_FILENAME).format(cur_fold), 'a') as f:
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    cur_epoch + 1,
                    train_loss.result(),
                    train_accuracy.result(),
                    val_loss.result(),
                    val_accuracy.result(),
                ))


            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nCurrent Fold: {}\
                        \nNo improvement in {} epochs, model is converged.\
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
                    nb.abs(val_accuracy.result() - best_val_acc) > epsilon:
                convergence_epoch_counter += 1
            else:
                convergence_epoch_counter = 0

            #if val_loss.result() < best_val_loss:
            if val_accuracy.result() > best_val_acc:  # Save on best validation accuracy instead
                best_epoch = cur_epoch + 1
                best_val_loss = val_loss.result()
                best_val_acc = val_accuracy.result()
                model.save_weights(
                    str(WEIGHT_DIR / "best_weights_fold_{}.h5".format(cur_fold))
                )
