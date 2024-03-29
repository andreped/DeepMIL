import os
import subprocess as sp
from tqdm import tqdm
import multiprocessing as mp
import tensorflow as tf
from numba import cuda
from tensorflow.keras import backend as K


def func(tmp):
    loc, curr_end_path = tmp
    sp.check_call(["lungmask", loc, curr_end_path], stderr=sp.DEVNULL, stdout=sp.DEVNULL)

    K.clear_session()


if __name__ == '__main__':

    # whether to use GPU or not
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0"

    # dynamically grow the memory used on the GPU (FOR TF==2.*)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    data_path = "/mnt/EncryptedPathology/DeepMIL/healthy_sick/"
    lungmask_path = "/mnt/EncryptedPathology/DeepMIL/healthy_sick_lungmask/"

    if not os.path.exists(lungmask_path):
        os.makedirs(lungmask_path)

    locs = []
    for curr_set in tqdm(os.listdir(data_path), "Set:"):
        path1 = data_path + curr_set + "/"
        curr_endpath1 = lungmask_path + curr_set + "/"
        if not os.path.exists(curr_endpath1):
            os.makedirs(curr_endpath1)

        for ct in tqdm(os.listdir(path1), "CT:"):
            loc = path1 + ct
            curr_id = ct.split(".")[0]
            curr_end_path = lungmask_path + curr_set + "/" + curr_id + "_lungmask.nii.gz"
            locs.append([loc, curr_end_path])

    proc_num = 4
    p = mp.Pool(proc_num)
    num_tasks = len(locs)
    r = list(tqdm(p.imap(func, locs), "CT", total=num_tasks))  # list(tqdm(p.imap(func,gts),total=num_tasks))
    p.close()
    p.join()