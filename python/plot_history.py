import numpy as np 
import h5py
from prettytable import PrettyTable
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import os, sys

#path = '/hdd/DP/output/history/history_24_02_2.hd5'
#history_path = "/mnt/EncryptedPathology/pathology/output/history/"
#path = "history_25_07_bcgrade_patch_classify_512_10x_inception_v3_augs_flip_rotate_hsv_mult_scale_model_resnet50_test_6.hd5"
#path = "history_25_07_bcgrade_patch_classify_512_10x_inception_v3_augs_flip_rotate_hsv_mult_scale_model_inceptionv3_test_6.hd5"
#path = "history_11-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10_model_inceptionv3_lr_1e-4_bs_8_eps_100_test_1.hd5"
#path = "history_11-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10_model_inceptionv3_lr_1e-4_bs_32_eps_100_test_1.hd5"
#path = "history_11-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10,mult:[0.9,1.1],scale:[1,1.2]_model_inceptionv3_dense_64_drp_0.7_lr_1e-4_bs_32_eps_100_classes_[1,2,3]_test_1.hd5"
#path = "history_11-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10,mult:[0.9,1.1],scale:[1,1.2]_model_simple_vgg_dense_64_drp_0.7_lr_1e-4_bs_32_eps_100_classes_[1,2,3]_test_1.hd5"
#path = "history_14-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10_model_inceptionv3_dense_64_drp_0.7_lr_1e-4_bs_32_eps_200_classes_[1,3]_test_1.hd5"
#path = "history_14-10-20_bcgrade_classify_images_512_10x_vsi_train_augs_flip:1,rotate_ll:1,hsv:10_model_inceptionv3_dense_64_drp_0.7_lr_1e-4_bs_32_eps_200_classes_[1,2,3]_test_1.hd5"
#path = history_path + path

path = sys.argv[1]

f = h5py.File(path, 'r')
data = np.array(f["history"])
f.close()

'''
print(data)
data = data[:,(0,1,4,5)]
print(data)
data[1:,2] = np.array([0.7285, 0.8807, 0.9087, 0.9185, 0.9332, 0.9417, 0.9533, 0.9595, 0.9689, 0.9735,
	0.9758, 0.9744, 0.9810, 0.9807, 0.9797, 0.9856, 0.9260, 0.9718, 0.9818, 0.9818, 0.9817])
data[1:,3] = np.array([0.4598, 0.4183, 0.5018, 0.5988, 0.6231, 0.6426, 0.5194, 0.6338, 0.7472, 0.8083,
	0.6081, 0.7060, 0.6635, 0.5885, 0.6780, 0.8986, 0.7969, 0.6906, 0.6759, 0.7750, 0.8123])
'''

metrics = [data[0][x].decode("UTF-8") for x in range(len(data[0]))]

print(data)

# remove unicode b from all eleiments
data = np.reshape(np.array([data[x,y].decode("UTF-8") for x in range(data.shape[0]) for y in range(data.shape[1])]), data.shape)

# split metrics from results
metrics = data[0,:]
data = np.round(data[1:,:].astype(np.float32), 8)
data = np.array(data).astype(float)

# make table of results
x = PrettyTable()
x.title = 'Training history'
metrics = metrics.tolist()
metrics.insert(0, "epochs")
x.field_names = metrics
epochs = np.array(range(data.shape[0]))
tmp = np.zeros(data.shape[1]+1, dtype=object)
for i in range(data.shape[0]):
    tmp[0] = i+1
    a = np.round(data[i,:], 5)
    tmp[1:] = a
    x.add_row(tmp)

# set epoch-column to ints


print(x)


# quick summary of lowest val_loss and for which epoch
print('Lowest val_loss observed: ')
print(np.amin(data[:,1]))
print('At epoch: ')
print(np.argmin(data[:,1])+1)

print('Highest val_acc observed: ')
print(np.amax(data[:,3]))
print('At epoch: ')
print(np.argmax(data[:, 3]+1))

epochs = np.array(range(1, data.shape[0]+1))

fig, ax = plt.subplots(1,2, figsize=(14,8))
plt.tight_layout()
ax[0].plot(epochs, data[:,0])
ax[0].plot(epochs, data[:,1])
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(['train', 'val'], loc = 'best')

ax[1].plot(epochs, data[:,2])
ax[1].plot(epochs, data[:,3])
ax[1].set_xlim([min(epochs), max(epochs)])
ax[1].set_ylim([0,1])
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(epochs, [max(data[:,3])]*len(epochs))
ax[1].plot(epochs[np.argmax(data[:,3])], max(data[:,3]), 'kx')
ax[1].legend(['train', 'val'], loc = 'best')

plt.show()
