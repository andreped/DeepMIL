import os
import subprocess as sp
from tqdm import tqdm

data_path = "/mnt/EncryptedPathology/DeepMIL/healthy_sick/"
lungmask_path = "/mnt/EncryptedPathology/DeepMil/healthy_sick_lungmask/"

if not os.path.exists(lungmask_path):
    os.makedirs(lungmask_path)

for curr_set in tqdm(os.listdir(data_path), "Set:"):
    path1 = data_path + curr_set + "/"
    curr_endpath1 = lungmask_path + curr_set + "/"
    if not os.path.exists(curr_endpath1):
        os.makedirs(curr_endpath1)

    for ct in tqdm(os.listdir(path1), "CT:"):
        loc = path1 + ct
        curr_id = ct.split(".")[0]
        curr_end_path = lungmask_path + curr_set + "/" + curr_id + "_lungmask.nii.gz"
        sp.check_call(["lungmask", loc, curr_end_path], stdout=open(os.devnull, 'wb'))
