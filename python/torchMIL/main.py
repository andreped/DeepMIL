from __future__ import print_function

import sys
import numpy as np
import os
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from timeit import default_timer as timer

from dataloader import LungBags
from model import AtnVGG,Attention, GatedAttention

torch.backends.cudnn.benchmark=True

def show_progbar(cur_step, num_instances, loss, acc, color_code, batch_size, time_per_step):
    TEMPLATE = "\r{}{}/{} [{:{}<{}}] - ETA: {}:{:02d} ({:>3.1f}s/step) - loss: {:>3.4f} - acc: {:>3.4f} \033[0;0m"

    progbar_length = 20

    curr_batch = cur_step  # int(cur_step // batch_size)
    nb_batches = int(num_instances // batch_size)
    ETA = (nb_batches - curr_batch) * time_per_step

    mins = int(ETA // 60)
    secs = int(np.round(ETA % 60))

    sys.stdout.write(TEMPLATE.format(
        color_code,
        curr_batch,
        nb_batches,
        "=" * min(int(progbar_length*(cur_step/nb_batches)), progbar_length),
        "-",
        progbar_length,
        mins,
        secs,
        time_per_step,
        loss,
        acc
    ))
    sys.stdout.flush()


# Training settings
#parser = argparse.Argumentparser(description='PyTorch MNIST bags Example')
#parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                    help='number of epochs to train (default: 10)')
#parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
#                    help='learning rate (default: 0.01)')
#parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
#                    help='weight decay')
#parser.add_argument('--target_number', type=int, default=9, metavar='T',
#                    help='bags have a positive labels if they contain at least one 9')
#parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
#                    help='average bag length')
#parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
#                    help='variance of bag length')
#parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
#                    help='number of bags in training set')
#parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
#                    help='number of bags in test set')
#parser.add_argument('--seed', type=int, default=2, metavar='S',
#                    help='random seed (default: 1)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
#parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

CUDA = True and torch.cuda.is_available()

torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)
    print('\nGPU is ON!')

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
train_dir = [[],[]]
val_dir = [[],[]]
for path in os.listdir(datapath):
    tmps[int(path[0])].append(path)
for i, tmp in enumerate(tmps):
    np.random.shuffle(tmp)
    tmps[i] = tmp
val = (0.8 * np.array([len(x) for x in tmps])).astype(int)
for i,tmp in enumerate(tmps):
    length = len(tmp)
    val = int(0.8 * length)
    val2 = int(0.9 * length)
    train_dir[i] = tmp[:val]
    val_dir[i] = tmp[val:val2]


train_dirs = train_dir.copy()
val_dirs = val_dir.copy()
train_n = len([item for sublist in train_dirs for item in sublist])
val_n = len([item for sublist in val_dirs for item in sublist])
batch_size = 16
nepoch = 30


print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_loader = data_utils.DataLoader(LungBags(file_list=train_dirs, data_path=datapath,
                                               aug={},
                                               seed=1,
                                               train=True),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     **loader_kwargs)

val_loader = data_utils.DataLoader(LungBags(file_list=val_dirs, data_path=datapath,
                                               aug={},
                                               seed=1,
                                               train=True),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     **loader_kwargs)


print('Init Model')
modelchoice = 1
if modelchoice == 1:
    model = AtnVGG()
elif modelchoice == 2:
    model = GatedAttention()
if CUDA:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)


def train(epoch,best_val):
    model.train()
    train_loss = []
    train_acc = []
    print("\n")
    print("Epoch %d/%d" % (epoch,nepoch))

    for batch_idx, (data, label) in enumerate(train_loader):
        time_list_train = []
        val_loss = []
        val_acc = []
        epoch_start = timer()
        curr_time = timer()

        if CUDA:
            data,label = data.cuda(), label.cuda()
        data, bag_label = Variable(data), Variable(label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        l = [(data[i],label[i]) for i in range(data.shape[0])]
        for (datax,labelx) in l:
            loss, _ = model.calculate_objective(datax, labelx)
            train_loss.append(loss.data[0].cpu().numpy())
            acc, _ = model.calculate_classification_error(datax, labelx)
            train_acc.append(acc)

            # backward pass
            loss.backward()
        # step
        optimizer.step()

        time_list_train.append(timer()-curr_time)
        curr_time=timer()
        time_avg = np.mean(time_list_train[::-1][:5])

        show_progbar(batch_idx,train_n,np.mean(train_loss),np.mean(train_acc), "\033[0;0m",batch_size,time_avg)

    with torch.no_grad():
        for (val_data,val_label) in val_loader:
            val_data, val_label = val_data.cuda(), val_label.cuda()
            l = [(val_data[i],val_label[i]) for i in range(val_data.shape[0])]
            for (val_datax,val_labelx) in l:
                loss, _ = model.calculate_objective(val_datax, val_labelx)
                val_loss.append(loss.data[0].cpu().numpy())
                acc, _ = model.calculate_classification_error(val_datax, val_labelx)
                val_acc.append(acc)
        print("val:")
        show_progbar(batch_idx,train_n,np.mean(val_loss),np.mean(val_acc), "\033[0;0m",batch_size,time_avg)
        if np.mean(val_acc) > best_val:
            best_val = np.mean(val_acc)
            torch.save(model.state_dict,"saved/model.pt")

    return best_val

    # calculate loss and error for epoch
    #train_loss /= len(train_loader)/batch_size
    #train_error /= len(train_loader)/batch_size

    #print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


if __name__ == "__main__":
    print('Start Training')
    best_val_acc = 0.0
    for epoch in range(1, nepoch + 1):
        best_val_acc = train(epoch,best_val_acc)
    print('Start Testing')
    test()
