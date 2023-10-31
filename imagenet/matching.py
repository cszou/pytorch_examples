import os

from tqdm import tqdm
import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
import scipy.optimize

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models

from collections import OrderedDict


def load_alexnet(model, path):
    preTrained = torch.load(path)['state_dict']
    param = model.state_dict()
    param['features.0.weight'] = preTrained['features.module.0.weight']
    param['features.0.bias'] = preTrained['features.module.0.bias']
    param['features.3.weight'] = preTrained['features.module.3.weight']
    param['features.3.bias'] = preTrained['features.module.3.bias']
    param['features.6.weight'] = preTrained['features.module.6.weight']
    param['features.6.bias'] = preTrained['features.module.6.bias']
    param['features.8.weight'] = preTrained['features.module.8.weight']
    param['features.8.bias'] = preTrained['features.module.8.bias']
    param['features.10.weight'] = preTrained['features.module.10.weight']
    param['features.10.bias'] = preTrained['features.module.10.bias']
    param['classifier.1.weight'] = preTrained['classifier.1.weight']
    param['classifier.1.bias'] = preTrained['classifier.1.bias']
    param['classifier.4.weight'] = preTrained['classifier.4.weight']
    param['classifier.4.bias'] = preTrained['classifier.4.bias']
    param['classifier.6.weight'] = preTrained['classifier.6.weight']
    param['classifier.6.bias'] = preTrained['classifier.6.bias']
    model.load_state_dict(param)
    return model


criterion = nn.CrossEntropyLoss()

traindir = os.path.join('imagenet', 'train')
valdir = os.path.join('imagenet', 'val')
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

train_dataset = torchvision.datasets.ImageFolder(
    traindir,
    T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]))

val_dataset = torchvision.datasets.ImageFolder(
    valdir,
    T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=False, num_workers=16, )

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=256, shuffle=False, num_workers=16, )


# given two networks net0, net1 which each output a feature map of shape NxCxWxH
# this will reshape both outputs to (N*W*H)xC
# and then compute a CxC correlation matrix between the outputs of the two networks
def run_corr_matrix(net0, net1, epochs=1, loader=train_loader):
    n = epochs * len(loader)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().cuda()
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).double()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).double()

                mean0_b = out0.mean(dim=0)
                mean1_b = out1.mean(dim=0)
                std0_b = out0.std(dim=0)
                std1_b = out1.std(dim=0)
                outer_b = (out0.T @ out1) / out0.shape[0]

                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    std0 = torch.zeros_like(std0_b)
                    std1 = torch.zeros_like(std1_b)
                    outer = torch.zeros_like(outer_b)
                mean0 += mean0_b / n
                mean1 += mean1_b / n
                std0 += std0_b / n
                std1 += std1_b / n
                outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr


def get_layer_perm1(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    corr_mtx_a = np.nan_to_num(corr_mtx_a)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map


# returns the channel-permutation to make layer1's activations most closely
# match layer0's.
def get_layer_perm(net0, net1):
    corr_mtx = run_corr_matrix(net0, net1)
    return get_layer_perm1(corr_mtx)


def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
    for w in pre_weights:
        w.data = w[perm_map]


def permute_input(perm_map, conv):
    w = conv.weight
    w.data = w[:, perm_map, :, :]


model1 = torchvision.models.alexnet()
model2 = torchvision.models.alexnet()

model1 = load_alexnet(model1, 'r1/checkpoint.pth.tar')
model2 = load_alexnet(model2, 'r2/checkpoint.pth.tar')

print(model1.state_dict()['features.0.weight'][0, 0, 0, 0])
print(model2.state_dict()['features.0.weight'][0, 0, 0, 0])

block1 = model1.features
block2 = model2.features
subnet1 = nn.Sequential(block1)
subnet2 = nn.Sequential(block2)
perm_map = get_layer_perm(subnet1, subnet2, val_loader)
print(perm_map)
print(perm_map.shape)
# permute_output(perm_map, block2.conv1, block2.bn1)
# permute_input(perm_map, block2.conv2)
print(model1.state_dict()['features.0.bias'])

print(model1.state_dict()['features.0.weight'][0, 0, 0, 0])
print(model2.state_dict()['features.0.weight'][0, 0, 0, 0])
