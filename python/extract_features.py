import h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras import backend as K

def func(ct):
	model = InceptionV3(include_top=False, weights='imagenet', pooling="max")
	#print(model.summary())

	ct = data_path + ct + "/"
	for path in os.listdir(ct):
		path = ct + path

		with h5py.File(path, "a") as f: # Use "a" to add data content to existing file without erasing existing data
			data = np.array(f["data"]).astype(np.float32)
			mask = np.array(f["lungmask"]).astype(np.float32)
			data[mask == 0] = 0
			del mask
			data = np.expand_dims(data, axis=0)
			data = np.stack((data,)*3, axis=-1)
			features = model.predict(data).flatten()
			del data
			if "features" in list(f.keys()):
				del f["features"]
			f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
			del features
	del model
	K.clear_session() # <- VERY IMPORTANT! To avoid memory leak

if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	height = 299
	width = 299
	channels = 3

	dataset_path = "040320_binary_healthy_sick_shape_(1,256,256)_huclip_[-1024,1024]_spacing_[1.0,1.0,2.0]"
	data_path = "/home/andrep/workspace/DeepMIL/data/"

	data_path += dataset_path + "/"

	cts = os.listdir(data_path)
	np.random.shuffle(cts)

	proc_num = 12 # 16
	p = mp.Pool(proc_num)
	r = list(tqdm(p.imap(func, cts), "CT", total=len(cts), smoothing=0.1))  # list(tqdm(p.imap(func,gts),total=num_tasks))
	p.close()
	p.join()