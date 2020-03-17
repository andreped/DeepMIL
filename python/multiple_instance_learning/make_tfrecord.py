import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import tensorflow as tf
import nibabel as nib

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from utils.tfrecord_utils import *
from utils.patch_ops import *



def maxminscale(x):
    if len(np.unique(x)) > 1:
        x = (x - x.min()) / (x.max() - x.min())
    return x

def intensity_normalize(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def clip_negatives(x):
    x[np.where(x <= 0)] = 0
    return x

def prepare_data(x_filename, y_label, num_classes):
    x = nib.load(str(x_filename)).get_fdata()
    #x = clip_negatives(x)
    x[x < -1024] = -1024
    x[x > 1024] = 1024
    x = maxminscale(x)

    x_slices = get_slices(x, TARGET_DIMS)
    x_slices = x_slices.astype(np.float16)  # np.float16, np.float32

    y = np.zeros((num_classes,), dtype=np.uint8)
    y[y_label] = 1

    return x_slices, y


'''
def prepare_data_thorax(x_filename, y_label, num_classes):

    # read CT
    curr_ct = str(x_filename)
    nib_volume = nib.load(curr_ct)
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    data = resampled_volume.get_data().astype('float32')

    # resize to get (512, 512) output images
    from scipy.ndimage import zoom
    img_size = input_shape[1]
    data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

    # pre-processing
    data[data < hu_clip[0]] = hu_clip[0]
    data[data > hu_clip[1]] = hu_clip[1]

    # intensity normalize [0, 1]
    data = minmaxscale(data)

    # fix orientation
    data = np.flip(data, axis=1)  # np.rot90(data, k=1, axes=(0, 1))
    data = np.flip(data, axis=0)

    # get z axis first
    data = np.swapaxes(data, 0, -1)

    # get lung mask, but don't mask the data. Add it in the generated data file, to be used in batchgen if of interest
    gt_nib_volume = nib.load(
        data_path[:-1] + "_lungmask/" + curr_ct.split(data_path)[1].split(".")[0] + "_lungmask.nii.gz")
    resampled_volume = resample_to_output(gt_nib_volume, new_spacing, order=0)
    gt = resampled_volume.get_data().astype('float32')

    # fix orientation
    gt = np.flip(gt, axis=1)
    gt = np.flip(gt, axis=0)

    # get z axis first
    gt = np.swapaxes(gt, 0, -1)

    gt[gt > 0] = 1
    data_shapes = data.shape
    gt_shapes = gt.shape
    gt = zoom(gt, [data_shapes[0] / gt_shapes[0],
                   data_shapes[1] / gt_shapes[1],
                   data_shapes[2] / gt_shapes[2]], order=0)
    data[gt == 0] = 0  # mask CT with lung mask # <- DONT MASK, one can easily do that if of interest in batchgen

    data = data.astype(np.float32)

    return data, y
'''


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
                ("Missing cmd line arguments.\n")
                ("First argument: source of data.\n")
                ("Second argument: destination folder for TFRecords.\n")
             )
        sys.exit()

    ######### DIRECTRY SETUP #########

    # pass the preprocessed data directory here
    IN_DATA_DIR = Path(sys.argv[1])
    OUT_DATA_DIR = Path(sys.argv[2])

    if not OUT_DATA_DIR.exists():
        OUT_DATA_DIR.mkdir(parents=True)

    TF_RECORD_FILENAME = OUT_DATA_DIR / "dataset_fold_{}_{}.tfrecord"
    TEST_FILENAMES_FILE = OUT_DATA_DIR / "test_filenames_labels.txt"

    # write which filenames and classes for train/val/test
    train_filenames_file = OUT_DATA_DIR / "train_filenames_fold_{}.txt"
    val_filenames_file = OUT_DATA_DIR / "val_filenames_fold_{}.txt"
    test_filenames_file = OUT_DATA_DIR /"test_filenames_fold_{}.txt"

    # write the number of instances/bags for progress bar purposes
    count_file = OUT_DATA_DIR / "count.txt"

    TARGET_DIMS = (512, 512)  # (256, 256)

    ######### GET DATA FILENAMES #######
    classes = sorted([d for d in IN_DATA_DIR.iterdir() if d.is_dir()])
    class_mapping = {c.name:i for i, c in enumerate(classes)}
    class_mapping_inv = {v:k for k, v in class_mapping.items()}

    X_names = []
    y = []
    class_counter = {c.name:0 for c in classes}

    for classdir in classes:
        for filename in classdir.iterdir():
            X_names.append(filename)
            cur_class = filename.parts[-2]
            y.append(class_mapping[cur_class])
            class_counter[cur_class] += 1

    print("Initial class distribution:")
    for c, count in class_counter.items():
        print("{}: {}".format(c, count))

    X_names = np.array(X_names)
    y = np.array(y)

    class_indices = [np.where(y == i)[0] for i in range(len(classes))]
    class_indices = [shuffle(c, random_state=4) for c in class_indices]

    ######### TRAIN/TEST SPLIT #########
    LIMIT_TRAIN_SPLIT = int(0.8 * min([len(i) for i in class_indices]))
    print("\nTraining distribution:")
    for i, n in enumerate(class_indices):
        print("Number of train samples for class {}: {}".format(
                class_mapping_inv[i],
                len(n[:LIMIT_TRAIN_SPLIT])
            )
        )
    print("\nTesting distribution:")
    for i, n in enumerate(class_indices):
        print("Number of test samples for class {}: {}".format(
                class_mapping_inv[i],
                len(n[LIMIT_TRAIN_SPLIT:])
            )
        )

    train_idx = np.concatenate([
        c[:LIMIT_TRAIN_SPLIT] for c in class_indices
    ])

    test_idx = np.concatenate([
        c[LIMIT_TRAIN_SPLIT:] for c in class_indices
    ])

    # shuffle indices for randomness
    train_idx = shuffle(train_idx, random_state=4)
    test_idx = shuffle(test_idx, random_state=4)

    # split
    X_names_train = X_names[train_idx]
    y_train = y[train_idx]

    X_names_test = X_names[test_idx]
    y_test = y[test_idx]

    ######### 5-FOLD TRAIN/VAL SPLIT #########
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)


    # K Fold for train and val
    for cur_fold, (train_idx, val_idx) in enumerate(skf.split(X_names_train, y_train)):

        print("Number training samples: {}\nNumber val samples: {}".format(
            len(train_idx),
            len(val_idx),
        ))

        dev_info = [
            (train_idx, "train", train_filenames_file),
            (val_idx, "val", val_filenames_file),
        ]

        for dev_idx, cur_name, fnames_file in dev_info:
            print("\nCreating {} TFRecord...".format(cur_name))
            cur_tfrecord = str(TF_RECORD_FILENAME).format(cur_fold, cur_name)
            count = 0
            with tf.io.TFRecordWriter(cur_tfrecord) as writer:
                for i in tqdm(dev_idx):
                    with open(fnames_file, 'a') as f:
                        f.write("{},{}\n".format(
                            X_names_train[i],
                            y_train[i],
                        ))

                    cur_x_slices, cur_y_label = prepare_data(
                        X_names_train[i],
                        y_train[i],
                        len(classes),
                    )

                    tf_example = image_example(
                        cur_x_slices,
                        cur_y_label,
                        len(cur_x_slices),
                    )

                    count += 1
                    writer.write(tf_example.SerializeToString())

            with open(count_file, 'a') as f:
                f.write("{} {} {}\n".format(cur_name, cur_fold, count))


    print("\nCreating Test TFRecord...")
    count = 0
    with tf.io.TFRecordWriter(str(TF_RECORD_FILENAME).format("_", "test")) as writer:
        for x_name, y_label in tqdm(zip(X_names_test, y_test), total=len(X_names_test)):
            with open(TEST_FILENAMES_FILE, 'a') as f:
                f.write("{},{}\n".format(x_name, y_label))
            x_slices, y_label = prepare_data(
                x_name,
                y_label,
                len(classes),
            )

            count += 1
            tf_example = volume_image_example(x_slices, y_label, len(x_slices))
            writer.write(tf_example.SerializeToString())

    with open(count_file, 'a') as f:
        f.write("{} {} {}\n".format("test", "_", count))
