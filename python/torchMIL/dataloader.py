"""Pytorch dataset object that loads MNIST dataset as bags."""

import os
import numpy as np
import torch
from torch.utils import data
import h5py
from torchvision import datasets, transforms

class LungBags(data.Dataset):
    def __init__(self, file_list, aug={}, nb_classes=2, input_shape=(16, 512, 512, 1), slab_shape=(16,512,512,1), mask_flag=False, seed=1, data_path='', train=True, shuffle_bag=False):

        #self.file_list = [item for sublist in file_list for item in sublist]
        self.file_list = file_list
        self.aug = aug
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.mask_flag = mask_flag
        self.bag_size = input_shape[2]
        self.data_path = data_path
        self.shuffle_bag = shuffle_bag

        self.r = np.random.RandomState(seed)

        self.num_in_train = len(self.file_list)

        self.file_list = [self.data_path + x for x in self.file_list]

    def __len__(self):
            return self.num_in_train

    def __getitem__(self, index):
        filename = self.file_list[index]
        f = h5py.File(filename + "/1.h5", "r")
        bag = np.array(f["data"]).astype(np.float32)
        if self.mask_flag:
            mask = np.array(f["lungmask"]).astype(np.float32)
            bag[mask == 0] = 0
        if self.shuffle_bag:
            slice_order = list(range(bag.shape[0]))
            np.random.shuffle(slice_order)
            bag = bag[slice_order]
        #bag = np.stack((bag,)*3,axis=1)
        bag = np.expand_dims(bag, axis=1)
        label = np.array(f["output"]).astype(np.float32)
        f.close()

        return bag, label

class MnistBags(data.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.max_bag_length = 14

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1
            if bag_length > 14:
                bag_length = 14
            if bag_length <= 14:
                pad = 14 - bag_length

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            bags = all_imgs[indices]
            bags = torch.nn.functional.pad(bags,(0,0,0,0,0,0,0,pad)).data

            labels_in_bag = torch.nn.functional.pad(all_labels[indices],(0,pad)).data
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(bags)
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":
    datagen_date = 100620
    input_shape = (128,256,256)
    slab_shape = (128,256,256)
    negative_class = "healthy"
    positive_class = "sick"
    hu_clip = [-1024,1024]
    new_spacing = [1,1,2]
    CNN3D_flag = True
    MIL_type = 2

    datapath = "/home/tan/Documents/PhD/DeepMIL/data/"
    data_name = str(datagen_date) + "_binary_" + negative_class + "_" + positive_class + \
    "_input_" + str(input_shape).replace(" ","") + \
    "_slab_" + str(slab_shape).replace(" ","") + \
    "_huclip_" + str(hu_clip).replace(" ","") + \
    "_spacing_" + str(new_spacing).replace(" ","") + \
    "_3DCNN_" + str(CNN3D_flag) + \
    "_" + str(MIL_type) + "DMIL"

    datapath += data_name + "/"

    tmps = [[],[]]
    for path in os.listdir(datapath):
        tmps[int(path[0])].append(path)
    for i, tmp in enumerate(tmps):
        np.random.shuffle(tmp)
        tmps[i] = tmp

    train_dir = [[],[]]
    train_dir = tmps
    train_dirs = train_dir.copy()


    train_loader = data.DataLoader(LungBags(file_list = train_dirs,data_path=datapath, aug={},
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.min(len_bag_list_train), np.max(len_bag_list_train)))
