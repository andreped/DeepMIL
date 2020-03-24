import os
import subprocess as sp
from tqdm import tqdm
import tensorflow as tf

# whether to use GPU or not
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0"

# dynamically grow the memory used on the GPU (FOR TF==2.*)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

data_path = "/mnt/EncryptedPathology/DeepMIL/MIL_emphysema/"
lungmask_path = "/mnt/EncryptedPathology/DeepMIL/MIL_emphysema_lungmask/"

if not os.path.exists(lungmask_path):
    os.makedirs(lungmask_path)

locs = []
for curr_folder in os.listdir(data_path):
    path1 = data_path + curr_folder
    if os.path.isdir(path1):
        path1 += "/"
        if not os.path.exists(lungmask_path + curr_folder + "/"):
            os.makedirs(lungmask_path + curr_folder + "/")
        for ct in os.listdir(path1):
            loc = path1 + ct
            curr_id = ct.split(".")[0]
            curr_end_path = lungmask_path + curr_folder + "/" + curr_id + "_lungmask.nii.gz"
            locs.append([loc, curr_end_path])

#locs = locs[6923:]

for loc, curr_end_path in tqdm(locs, "CT: "):
    print(loc, curr_end_path)
    if os.path.exists(curr_end_path):
        continue
    #try:
    #sp.check_call(["lungmask", loc, curr_end_path, "--batchsize", "64"], stderr=sp.DEVNULL, stdout=sp.DEVNULL)
    sp.check_call(["lungmask", loc, curr_end_path], stderr=sp.DEVNULL, stdout=sp.DEVNULL)
    #sp.check_call(["lungmask", loc, curr_end_path, "--cpu"], stderr=sp.DEVNULL, stdout=sp.DEVNULL)
    #except Exception:
    #    print("something is wrong with the file: \n" + loc)
    #    del Exception