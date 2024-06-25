import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import math
import torch.optim as optim
import torch.hub as hub
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10, FashionMNIST, ImageFolder, SVHN, CIFAR100
from imagenetv2_pytorch import ImageNetV2Dataset, ImageNetValDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
# from torchvision.transforms import ToTensor
# import argparse
import scipy.ndimage as nd
from scipy.special import softmax
from collections import OrderedDict, Counter
import copy
import argparse
import time
import json, collections
from matplotlib.image import imread
import sys
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from IPython import display
from PIL import Image
import tarfile
from zipfile import ZipFile
import random
from numpy.random import normal
from torch.utils.data import random_split
from numpy import hstack, vstack
from statsmodels.distributions.empirical_distribution import ECDF
import os
import multiprocessing

from loss import *
# os.environ["NCCL_P2P_DISABLE"]="TRUE"
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.getcwd()

# Allow reproducability
SEED = random.randint(1, 2000) # OR a constant number, e.g. 11
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

##### Setting the Path #####
os.chdir('your path')
root_path = os.path.join(os.getcwd(), 'your inner path')

def data_choice_val(dataset='mnist', data_path=os.path.join(root_path, 'data'), dl=False, num_calib=5000):
    cpu_count = multiprocessing.cpu_count()
    preprocess_transform = transforms.Compose([
                transforms.Resize(256), # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    loader_tuple = None
    
    if dataset == 'mnist':     ## MNIST data
        b_size = 512
        data_train = MNIST(os.path.join(data_path, 'mnist'),
                           download=dl,
                           train=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
        
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = MNIST(os.path.join(data_path, 'mnist'),
                         train=False,
                         download=dl,
                         transform=transforms.Compose([transforms.ToTensor()]))
        data_val = ImageFolder(os.path.join(data_path,'notmnist'), 
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Grayscale(num_output_channels=1)]))
        
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size, num_workers=cpu_count-1, shuffle=True, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size, num_workers=cpu_count-1, shuffle=True, pin_memory=True)
        sample, _ = data_val[5]

    elif dataset == 'fmnist':    ## FashionMNIST data
        b_size = 512
        data_train = FashionMNIST(os.path.join(data_path, 'fashion_mnist'),
                           download=dl,
                           train=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
        
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = FashionMNIST(os.path.join(data_path, 'fashion_mnist'),
                             train=False,
                             download=dl,
                             transform=transforms.Compose([transforms.ToTensor()]))

        data_val = ImageFolder(os.path.join(data_path,'notmnist'), 
                               transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Grayscale(num_output_channels=1)]))
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size, num_workers=cpu_count-1, shuffle=True, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size, num_workers=cpu_count-1, shuffle=True, pin_memory=True)
        sample, _ = data_val[5]

    elif dataset == 'cifar10':
        b_size = 256
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(data_path, 'cifar_orig'), train=True, download=dl, transform=transform_train)
        dataloader_train = DataLoader(
            trainset, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root=os.path.join(data_path, 'cifar_orig'), train=False, download=dl, transform=transform_test)
        dataloader_val_same = DataLoader(
            testset, batch_size=b_size, shuffle=False, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = None
    
    elif dataset == 'imagenet':
        b_size = 128
        transform_imagenet = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

        # Get the conformal calibration dataset
        # imagenet_calib_data, imagenet_val_data = random_split(ImageFolder(imagenet_path, transform_imagenet), [num_calib,50000-num_calib])
        imagenet_val_data = ImageNetValDataset(transform=transform_imagenet, location=data_path)
        # imagenet_val_data = ImageFolder(os.path.join(data_path, 'imagenet_val'), transform_imagenet)
        #### Note that VAL_DATASET_SIZE = 50000 and V2_DATASET_SIZE = 10000
        
        # Initialize loaders 
        # dataloader_train = DataLoader(imagenet_calib_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val_same = DataLoader(imagenet_val_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)

        dataloader_train = None
        dataloader_val = None
        
        # Get the conformal calibration dataset
        calib_data, val_data = torch.utils.data.random_split(imagenet_val_data, [num_calib, 50000 - num_calib])

        # Initialize loaders 
        calib_loader = DataLoader(calib_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        loader_tuple = (calib_loader, val_loader)
    
    elif dataset == 'imagenetv2':
        b_size = 128
        transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])
        ])
        dataloader_train = None
        dataloader_val = None
        
        #### variant="threshold-0.7" and "top-images" and "matched-frequency" can be used.
        imagenetv2_val_data = ImageNetV2Dataset("matched-frequency", transform=transform_imagenet, location=data_path) # supports matched-frequency, threshold-0.7, top-images variants
        #### Note that VAL_DATASET_SIZE = 50000 and V2_DATASET_SIZE = 10000
        dataloader_val_same = DataLoader(imagenetv2_val_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        # Get the conformal calibration dataset
        calib_data_v2, val_data_v2 = torch.utils.data.random_split(imagenetv2_val_data, [num_calib, 10000 - num_calib])

        # Initialize loaders 
        calib_loader_v2 = DataLoader(calib_data_v2, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        val_loader_v2 = DataLoader(val_data_v2, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        loader_tuple = (calib_loader_v2, val_loader_v2)
        
    elif dataset == 'cifar100':    ## CIFAR100 dataset
        b_size = 128
        if dl:
            dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar-100-python.tar.gz', 'r:gz') as tar:
                tar.extractall(path=data_path)
        
        cifar100_val_data = CIFAR100(os.path.join(data_path, 'cifar-100-python'), train=False, 
                       download=True, transform=transforms.Compose([transforms.ToTensor()]))

        dataloader_val_same = DataLoader(cifar100_val_data, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        dataloader_train = None
        dataloader_val = None
        
    elif dataset == 'cifar10-svhn':    ## CIFAR10-SVHN dataset
        b_size = 256
        if dl:
            dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
                tar.extractall(path=data_path)
        
        data_train = ImageFolder(os.path.join(data_path,'cifar10','train'), transform=transforms.ToTensor())
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = ImageFolder(os.path.join(data_path,'cifar10','test'), transform=transforms.ToTensor())
        data_val = SVHN(os.path.join(data_path, 'svhn'), split='test', 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        sample, _ = data_val[11]
        
    elif dataset == 'tiny_imagenet':    ## imagenet-SVHN dataset
        b_size = 60
        if dl:
            dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            download_url(dataset_url, '.')

            with ZipFile('./tiny-imagenet-200.zip', 'r') as zip:
                zip.extractall(path=data_path)
        
        data_train = ImageFolder(os.path.join(data_path,'tiny-imagenet-200','train'), transform=preprocess_transform)
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = ImageFolder(os.path.join(data_path,'tiny-imagenet-200','val','images'), transform=preprocess_transform)
        data_val = SVHN(os.path.join(data_path, 'svhn'), split='test', 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        sample, _ = data_val[11]
        
#     elif dataset == 'cifar10-svhn':    ## CIFAR10-SVHN dataset
#         b_size = 128
        
#         data_train = CIFAR10(os.path.join(data_path, 'cifar10'),
#                            download=dl,
#                            train=True,
#                            transform=transforms.Compose([transforms.ToTensor()]))
        
#         dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=8)
#         if same_data_val:
#             data_val = CIFAR10(os.path.join(data_path, 'cifar10'),
#                              train=False,
#                              download=dl,
#                              transform=transforms.Compose([transforms.ToTensor()]))
#         else:
#             data_val = SVHN(os.path.join(data_path, 'svhn'), split='test', 
#                             download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        
#         dataloader_val = DataLoader(data_val, batch_size=b_size*2, shuffle=True)
#         sample, _ = data_val[11]
        
    elif dataset == 'cifar10-100':    ## CIFAR10-100 dataset
        b_size = 256
        if dl:
            dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar-100-python.tar.gz', 'r:gz') as tar:
                tar.extractall(path=data_path)
            
            dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
                tar.extractall(path=data_path)
        
        data_train = ImageFolder(os.path.join(data_path,'cifar10','train'), transform=transforms.ToTensor())
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = ImageFolder(os.path.join(data_path,'cifar10','test'), transform=transforms.ToTensor())
        data_val = CIFAR100(os.path.join(data_path, 'cifar-100-python'), train=False, 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        data_val_list = []
        for i in range(len(data_val)):
            if data_val[i][1] < 10:
                data_val_list.append(data_val[i])
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = DataLoader(data_val_list, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        sample, _ = data_val_list[1]
    
    elif dataset == 'cifar100-svhn':    ## CIFAR100-SVHN dataset
        b_size = 256
        if dl:
            dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar-100-python.tar.gz', 'r:gz') as tar:
                tar.extractall(path=data_path)
        
        data_train = CIFAR100(os.path.join(data_path, 'cifar-100-python'), train=True, 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        
        data_val_same = CIFAR100(os.path.join(data_path, 'cifar-100-python'), train=False, 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))
        data_val = SVHN(os.path.join(data_path, 'svhn'), split='test', 
                        download=dl, transform=transforms.Compose([transforms.ToTensor()]))        

#        data_train_list = []
#        for i in range(len(data_train)):
#            if data_train[i][1] < 10:
#                data_train_list.append(data_train[i])
        dataloader_val_same = DataLoader(data_val_same, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        sample, _ = data_val[11]
        
#     elif dataset == 'cifar10-100':    ## CIFAR10-100 dataset
#         b_size = 128
        
#         data_train = CIFAR10(os.path.join(data_path, 'cifar10'),
#                            download=dl,
#                            train=True,
#                            transform=transforms.Compose([transforms.ToTensor()]))
        
#         dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=8)
#         if same_data_val:
#             data_val = CIFAR10(os.path.join(data_path, 'cifar10'),
#                              train=False,
#                              download=dl,
#                              transform=transforms.Compose([transforms.ToTensor()]))
#         else:
#             data_val = CIFAR100(os.path.join(data_path, 'cifar100'), train=False, 
#                             download=dl, transform=transforms.Compose([transforms.ToTensor()]))
#             data_val_list = []
#             for i in range(len(data_val)):
#                 if data_val[i][1] < 10:
#                     data_val_list.append(data_val[i])
#         dataloader_val = DataLoader(data_val_list, batch_size=b_size*2, shuffle=True)
#         sample, _ = data_val[11]
    
    elif dataset == 'cifar5':    ## CIFAR5 dataset
        b_size = 256
        if dl:
            dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
            download_url(dataset_url, '.')

            with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
                tar.extractall(path=data_path)

        data_train = ImageFolder(os.path.join(data_path,'cifar5','train'), transform=transforms.ToTensor())
        data_val = ImageFolder(os.path.join(data_path,'cifar5','test'), transform=transforms.ToTensor())
        
        dataloader_train = DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val = DataLoader(data_val, batch_size=b_size*2, shuffle=True, num_workers=cpu_count-1, pin_memory=True)
        dataloader_val_same = None
        sample, _ = data_val[11]
    
    else:
        print('Please enter a valid dataset name, e.g., \'mnist\', \'fmnist\', \'cifar10\', and \'cifar5\'')
    
    dataloaders = {"train": dataloader_train, "val": dataloader_val, "val_same": dataloader_val_same, "loader_tup": loader_tuple}
    return dataloaders

## helpers
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(get_device())
    return y[labels]

def rotate_img(x, deg, data='mnist'):
    if data == 'mnist' or data == 'fmnist':
        return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
    else:
        return nd.rotate(x.reshape(32, 32), deg, reshape=False).ravel()

def apply_dropout(m):
    for each_module in m.children():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
    return m
            
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def uncertainties(p):
    aleatoric = np.mean(p*(1-p), axis=0)
    epistemic = np.mean(p**2, axis=0) - np.mean(p, axis=0)**2
    return aleatoric, epistemic

### utils for evidential computation
def relu(array):
    return torch.maximum(array, torch.tensor(0))

# def evid_prob(array):
#     return array / torch.sum(array, dim=1, keepdim=True)

def adapt_plot(tup_dict):

    groups = ("1", "2-3", "4-6", "7-10", "11-100", "101-1000")
    mean_size = {
        'Base': tup_dict['base'],
        'APS': tup_dict['aps'],
        'RAPS': tup_dict['raps'],
        'ECP': tup_dict['ecp']
    }

    x = np.arange(len(groups))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0

    plt.rcParams['figure.figsize'] = [8, 6]
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in mean_size.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width=width, label=attribute, align='center')
        # ax.bar_label(rects, fontsize=4, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Prediction Set Size', fontsize=19)
    ax.set_xlabel('True Label Rank (Difficulty) of Testing data', fontsize=19)
    # ax.set_title('Adaptivity in ECP')
    ax.set_xticks(x + (3/2)*width, groups, fontsize=16)
    ax.legend(loc='upper left', ncols=4, fontsize=14)
    # ax.set_ylim(1, 100)

    # plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'results', 'adapt_ecp_diff.pdf'), bbox_inches='tight')
    # plt.show()

def adapt_plot_size(tup_dict):

    groups = ("0.90", "0.95", "0.99")
    mean_size = {
        'Base': tup_dict['naive'],
        'APS': tup_dict['aps'],
        'RAPS': tup_dict['raps'],
        'ECP': tup_dict['ecp']
    }

    x = np.arange(len(groups))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0
    
    plt.rcParams['figure.figsize'] = [8, 4]
    fig, ax = plt.subplots(layout='constrained')
    
    for attribute, measurement in mean_size.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width=width, label=attribute, align='center')
        # ax.bar_label(rects, fontsize=4, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Prediction Set Size', fontsize=19)
    ax.set_xlabel('Coverage Level: 1 - \u03B4', fontsize=19)
    # ax.set_title('Adaptivity in ECP')
    ax.set_xticks(x + (3/2)*width, groups, fontsize=18)
    ax.legend(loc='upper left', ncols=4, fontsize=13)
    # ax.set_ylim(1, 100)

    # plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'results', 'adapt_ecp_size.pdf'), bbox_inches='tight')
    # plt.show()

def adapt_plot_cov(tup_dict):

    groups = ("0.90", "0.95", "0.99")
    mean_size = {
        'Base': tup_dict['naive'],
        'APS': tup_dict['aps'],
        'RAPS': tup_dict['raps'],
        'ECP': tup_dict['ecp']
    }

    x = np.arange(len(groups))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0
    
    plt.rcParams['figure.figsize'] = [8, 4]
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in mean_size.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width=width, label=attribute, align='center')
        # ax.bar_label(rects, fontsize=4, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Empirical Coverage', fontsize=19)
    ax.set_xlabel('Coverage Level: 1 - \u03B4', fontsize=19)
    # ax.set_title('Adaptivity in ECP')
    ax.set_xticks(x + (3/2)*width, groups, fontsize=18)
    ax.legend(loc='upper left', ncols=4, fontsize=13)
    ax.set_ylim(0.85, 1.0)

    # plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'results', 'adapt_ecp_cov.pdf'), bbox_inches='tight')
    # plt.show()
    
################ My Conformal Prediction ###############
# Problem setup for conformal prediction
# logits and labels should be numpy arrays...
def cp_generator(logits, labels, temp_scale=1, n=5000, alpha=0.1, method='ecp', lam_reg=0, k_reg=5, 
                 disallow_zero_sets=False, rand=True, label_strings=None, verbose=True):
#     n=1000       # number of calibration points
#     alpha = 0.1       # 1-alpha is the desired coverage
#     # Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
#     lam_reg = 0.01
#     k_reg = 5
#     disallow_zero_sets = False       # Set this to False in order to see the coverage upper bound hold
#     rand = True      # Set this to True in order to see the coverage upper bound hold
    num_classes = logits.shape[1]
    labels = labels.cpu().numpy()
    labels = labels.reshape(-1,1)
    smx_probs = F.softmax(logits/temp_scale, dim=1)
    smx_probs = smx_probs.cpu().numpy()
    if method == 'ecp':
        evidence = relu_evidence(logits)
        dirich_param = evidence + 1      # 1 = 1/K * K
        dirich_sum = torch.sum(dirich_param, dim=1, keepdim=True)
        evidential_probs = dirich_param / dirich_sum
        evidential_uncer = num_classes / dirich_sum
        evidential_belief = evidence / dirich_sum
        uncer = evidential_uncer.cpu().numpy()
        probs = evidential_probs.cpu().numpy()
        belief = evidential_belief.cpu().numpy()
    else:
        probs = smx_probs
        if method == 'aps' or method == 'naive':
            lam_reg = 0
    reg_vec = np.array(k_reg*[0,] + (num_classes - k_reg)*[lam_reg,])[None,:]

    #############################################################################
    ### Split the instances into calibration and validation sets (save the shuffling)

    idx = np.array([1] * n + [0] * (probs.shape[0]-n)) > 0
    np.random.shuffle(idx)
    cal_probs, val_probs = probs[idx,:], probs[~idx,:]
    cal_labels, val_labels = labels[idx], labels[~idx]
    cal_labels = cal_labels.astype(int)
    val_labels = val_labels.astype(int)
#     cal_probs = probs[idx,:]
#     cal_labels = labels[idx]

    #############################################################################
    ### Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_probs.argsort(1)[:,::-1]    # indices of descending sorted calibration probs
    if method == 'ecp':
        cal_smx, val_smx = smx_probs[idx,:], smx_probs[~idx,:]
        cal_belief, val_belief = belief[idx,:], belief[~idx,:]
        cal_uncer, val_uncer = uncer[idx], uncer[~idx]
        # beta_penalty = 2
        cal_rank = np.where(cal_pi == cal_labels)[1] #+ 1
        cal_rank = cal_rank.reshape(-1,1)
        cal_rank_all = cal_pi.argsort(1)
        ###########
        # evid_scores = ((1/num_classes) * np.log(cal_probs)) / ( (1 - (cal_rank/num_classes)) * (cal_probs**2) * (cal_belief) + 1e-8) # * (cal_belief)
        evid_scores = ((1/num_classes) * np.log(cal_probs)) / ( (cal_smx) * (cal_probs**2) * (1 - (cal_rank_all/num_classes)) + 1e-8) #  * (1 - (cal_rank_all/num_classes)**2) 
        ## (1 - (np.maximum(cal_rank - 5, 0) / num_classes))
        cal_scores = ((-1) * evid_scores * cal_uncer) #* (1 - (cal_rank_all/num_classes)**2) #- (np.maximum(cal_rank_all - 100, 0) / num_classes) #+ (cal_rank/num_classes)**(1/2) # * (cal_rank**(1/beta_penalty)) # * ((cal_rank / num_classes)**beta_penalty)
        ###### cal_scores = 1 - np.exp((-1) * cal_scores) # softmax(cal_scores, axis=1)
        cal_scores = (cal_scores / np.max(cal_scores, axis=1).reshape(-1,1)) #- (np.maximum(cal_rank - 100, 0) / num_classes)
        
        cal_scores = np.take_along_axis(cal_scores, cal_pi, axis=1) # cal_scores * (1 - (np.maximum(cal_rank_all - 10, 0) / 1000))
        cal_scores_reg = (cal_scores + reg_vec) #**(1/2) # - (np.maximum(np.arange(num_classes).reshape(1,-1) - 10, 0) / (num_classes)) #
        cal_scores = cal_scores_reg[np.arange(n), (cal_rank).squeeze()] # - cal_uncer * cal_scores_reg[np.arange(n), (cal_rank).squeeze()] #  cal_rank - 1
        ##### cal_scores_reg.cumsum(axis=1)[np.arange(n), (cal_rank).squeeze()]  ### np.random.rand(n) *
    
    elif method == 'lac':
        cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1)
        cal_L = np.where(cal_pi == cal_labels)[1]
        cal_scores = 1 - cal_srt[np.arange(n), cal_L]
        
    elif method == 'naive':
        print('The Naive method is selected for Conformal Prediction\n')
    
    else:
        cal_srt = np.take_along_axis(cal_probs, cal_pi, axis=1)     # cal_srt = torch.take_along_dim(cal_smx, cal_pi, dim=1)
        cal_srt_reg = cal_srt + reg_vec 
        cal_L = np.where(cal_pi == cal_labels)[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n), cal_L]
        
    ### Get the score quantile
    if method != 'naive':
        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    else:
        qhat = 1 - alpha
    
    if verbose:
        print(f"The quantile is: {qhat}")
    
    ### Deploy
    n_val = val_probs.shape[0]
    val_pi = val_probs.argsort(1)[:,::-1]    # indices of descending sorted validation probs
    if method == 'ecp':
        val_rank = val_pi.argsort(1)
        # evid_scores = ((1/num_classes)* np.log(val_probs)) / ( (1 - (val_rank/num_classes)) * (val_probs**2) * (val_belief) + 1e-8) # * (val_belief)
        evid_scores = ((1/num_classes)* np.log(val_probs)) / ((val_smx) * (val_probs**2) * (1 - (val_rank/num_classes)) + 1e-8) #  
        val_scores = ((-1) * evid_scores * val_uncer) #* (1 - (val_rank/num_classes)**2) #- (np.maximum(val_rank - 100, 0) / num_classes) #* (1 - (np.maximum(val_rank - 20, 0) / 1000)) #+ (val_rank/num_classes)**(1/2) # * (val_rank**(1/beta_penalty)) # * ((val_rank / num_classes)**beta_penalty) 
        ###########
        ###### val_scores = 1 - np.exp((-1) * val_scores) # softmax(val_scores, axis=1)
        val_scores = (val_scores / np.max(val_scores, axis=1).reshape(-1,1)) #- (np.maximum(val_rank - 100, 0) / num_classes)
        
        val_scores = np.take_along_axis(val_scores, val_pi, axis=1) # val_scores * (1 - (np.maximum(val_rank - 10, 0) / 1000))
        val_scores_reg = (val_scores + reg_vec)
        val_scores = val_scores_reg #.cumsum(axis=1)
        
        indicators = val_scores <= qhat  
        # indicators = (val_scores - val_uncer * val_scores_reg) <= qhat if rand else (val_scores - val_scores_reg) <= qhat  # np.random.rand(n_val,1) *
        
        if disallow_zero_sets: indicators[:,0] = True 
        ### To revert them in their original form of class labels
        prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
        # prediction_sets = indicators
    
    elif method == 'lac':
        indicators = (1 - val_probs) <= qhat 
        if disallow_zero_sets: indicators[:,0] = True 
        prediction_sets = indicators
    
    elif method == 'naive':
        val_srt = np.take_along_axis(val_probs, val_pi, axis=1)
        val_srt_cumsum = val_srt.cumsum(axis=1)
        L = np.sum(val_srt_cumsum < qhat, axis=1)
        U = np.random.rand(n_val)
        V = (val_srt_cumsum[np.arange(n_val),(L).squeeze()] - (1 - alpha)) / val_srt[np.arange(n_val),(L).squeeze()]
        for i in range(n_val):
            if U[i] <= V[i]: 
                L[i] = L[i] - 1
        indicators = np.array([(L[i] + 1) * [1,] + (num_classes - L[i] - 1) * [0,] for i in range(n_val)])
        if disallow_zero_sets: indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)

    else:
        val_srt = np.take_along_axis(val_probs, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
        indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        if disallow_zero_sets: indicators[:,0] = True 
        ### To revert them in their original form of class labels
        prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    
    ### Finding the true label rank in the descending-sorted set of probs (starting from 1)
    true_label_rank = np.where(val_pi == val_labels)[1] + 1   # 1D np array for all testing data points 
    
    ### Calculate empirical coverage
    empirical_coverage = prediction_sets[np.arange(n_val), val_labels.squeeze()].mean()
    if verbose:
        print(f"The empirical coverage is: {empirical_coverage}")
    
    ### Create the prediction sets with their sizes and coverages (with/without label strings)
    p_size, p_size_1, p_size_2, p_size_3, p_size_4, p_size_5, p_size_6 = [], [], [], [], [], [], []
    counts = np.array([0, 0, 0, 0, 0, 0])
    sums = np.array([0, 0, 0, 0, 0, 0])
#     p_set_list = []
    for i, p_set in enumerate(prediction_sets):
        p_size.append(np.sum(p_set))
        if true_label_rank[i] <= 1:
            counts[0] += 1
            p_size_1.append(np.sum(p_set))
            sums[0] += p_set[val_labels.squeeze()[i]]
        elif true_label_rank[i] >= 2 and true_label_rank[i] <= 3:
            counts[1] += 1
            p_size_2.append(np.sum(p_set))
            sums[1] += p_set[val_labels.squeeze()[i]]
        elif true_label_rank[i] >= 4 and true_label_rank[i] <= 6:
            counts[2] += 1
            p_size_3.append(np.sum(p_set))
            sums[2] += p_set[val_labels.squeeze()[i]]
        elif true_label_rank[i] >= 7 and true_label_rank[i] <= 10:
            counts[3] += 1
            p_size_4.append(np.sum(p_set))
            sums[3] += p_set[val_labels.squeeze()[i]]
        elif true_label_rank[i] >= 11 and true_label_rank[i] <= 100:
            counts[4] += 1
            p_size_5.append(np.sum(p_set))
            sums[4] += p_set[val_labels.squeeze()[i]]
        else:
            counts[5] += 1
            p_size_6.append(np.sum(p_set))
            sums[5] += p_set[val_labels.squeeze()[i]]
        
        
        # if true_label_rank[i] >= 3 and true_label_rank[i] <= 10:
            # counts[5] += 1
            # p_size_6.append(np.sum(p_set))
            # sums[5] += p_set[val_labels.squeeze()[i]]
        # elif true_label_rank[i] >= 11 and true_label_rank[i] <= 1000:
            # counts[6] += 1
            # p_size_7.append(np.sum(p_set))
            # sums[6] += p_set[val_labels.squeeze()[i]]
    
    counts_strat = np.array([0, 0, 0, 0, 0, 0])
    sums_strat = np.array([0, 0, 0, 0, 0, 0])
    counts_bench = np.array([0, 0, 0, 0, 0])
    sums_bench = np.array([0, 0, 0, 0, 0])
    for i, p_set in enumerate(prediction_sets):
        size_strat = np.sum(p_set)
        if size_strat <= 1:
            counts_strat[0] += 1
            sums_strat[0] += p_set[val_labels.squeeze()[i]]
            counts_bench[0] += 1
            sums_bench[0] += p_set[val_labels.squeeze()[i]]
        elif size_strat >= 2 and size_strat <= 3:
            counts_strat[1] += 1
            sums_strat[1] += p_set[val_labels.squeeze()[i]]
            counts_bench[1] += 1
            sums_bench[1] += p_set[val_labels.squeeze()[i]]
        elif size_strat >= 4 and size_strat <= 6:
            counts_strat[2] += 1
            sums_strat[2] += p_set[val_labels.squeeze()[i]]
            counts_bench[2] += 1
            sums_bench[2] += p_set[val_labels.squeeze()[i]]
        elif size_strat >= 7 and size_strat <= 10:
            counts_strat[3] += 1
            sums_strat[3] += p_set[val_labels.squeeze()[i]]
            counts_bench[2] += 1
            sums_bench[2] += p_set[val_labels.squeeze()[i]]
        elif size_strat >= 11 and size_strat <= 100:
            counts_strat[4] += 1
            sums_strat[4] += p_set[val_labels.squeeze()[i]]
            counts_bench[3] += 1
            sums_bench[3] += p_set[val_labels.squeeze()[i]]
        else:
            counts_strat[5] += 1
            sums_strat[5] += p_set[val_labels.squeeze()[i]]
            counts_bench[4] += 1
            sums_bench[4] += p_set[val_labels.squeeze()[i]]
            
        #### if label_strings is not None:
#         p_list = list(label_strings[p_set])
#         p_set_list.append(p_list)
#         len_avg += len(p_list)
#     len_avg /= len(prediction_sets)
    
    
    #############################################################################
    p_size_tuple = ( np.round(np.array(p_size_1).mean(),2), np.round(np.array(p_size_2).mean(),2), 
                    np.round(np.array(p_size_3).mean(),2), np.round(np.array(p_size_4).mean(),2), 
                    np.round(np.array(p_size_5).mean(),2), np.round(np.array(p_size_6).mean(),2) )
    
    p_size_std = ( np.round(np.array(p_size_1).std(),2), np.round(np.array(p_size_2).std(),2), 
                    np.round(np.array(p_size_3).std(),2), np.round(np.array(p_size_4).std(),2), 
                    np.round(np.array(p_size_5).std(),2), np.round(np.array(p_size_6).std(),2) )
    
    p_cov = np.round(sums / counts, 4)
    cov_tuple = (p_cov, counts)
    
    cov_strat = sums_strat[counts_strat > 0] / counts_strat[counts_strat > 0]
    strat_tuple = (cov_strat, counts_strat)
    
    cov_bench = sums_bench[counts_bench > 0] / counts_bench[counts_bench > 0]
    sscv = np.max(np.absolute(cov_bench - 1 + alpha))
    
    p_size_arr = np.array(p_size)
    mean_size = p_size_arr[p_size_arr != 0].mean()
    size_adapt_trade = (1 - sscv) / (mean_size + 1e-10)
    
    sscv = np.round(sscv, 5)
    size_adapt_trade = np.round(size_adapt_trade, 5)
    
    return p_size_arr, p_size_tuple, p_size_std, cov_tuple, true_label_rank, n, alpha, strat_tuple, sscv, size_adapt_trade


######### Training ###
def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    logits = torch.zeros((len(loader.dataset), 1000)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Finding the optimal temperature:')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits
    
def train_cp(epoch, trainloader, net_model, criterion, optimizer, device):
    print('\nEpoch: {}'.format(epoch+1))
    net_model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
    #    print('batch number: {}'.format(batch_idx))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        max_vals, predicted = outputs.max(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Training -> Loss: %.4f | Acc: %.4f%% (%d/%d)'
                 % (train_loss/len(trainloader), 100.*correct/total, correct, total))

    
########## Testing ###
def test_cp(epoch, num_classes, cal_loader, testloader, net_model, best_acc, criterion, device, data_name='cifar10'):
    full_outputs = torch.empty(0, num_classes, device=device)
    full_labels = torch.empty(0, 1, device=device)
    net_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader): # using the validation dataloader
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_model(inputs)
            full_outputs = torch.vstack((full_outputs, outputs))
            full_labels = torch.vstack((full_labels, targets.reshape(-1,1)))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            max_vals, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Testing -> Loss: %.4f | Acc: %.4f%% (%d/%d)'
             % (test_loss/len(testloader), 100.*correct/total, correct, total))
    
    print("Temperature Scaling...")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(net_model, cal_loader)
    logits_loader = DataLoader(logits_dataset, batch_size=cal_loader.batch_size, shuffle=False, pin_memory=True)
    temp = platt_logits(logits_loader)
    print('Optimal Temprature: {}'.format(temp.item()))
    
    # Save checkpoint.
    acc = 100.*correct/total
    loss = test_loss/len(testloader)
    if acc > best_acc:
        print('Saving the model...')
        state = {
            'model': net_model.state_dict(),
            'acc': acc,
            'loss': loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join(root_path, 'results','ecp_checkpoint')):
            os.mkdir(os.path.join(root_path, 'results','ecp_checkpoint'))
        torch.save(state, os.path.join(root_path, 'results', 'ecp_checkpoint', 'ckpt_{}.pth'.format(data_name)))
        best_acc = acc
    return full_outputs, full_labels, best_acc, temp.item()
    
    
def patch_hub_with_proxy():
    download_url_to_file = hub.download_url_to_file

    def _proxy_download_url_to_file(
        url: str,
        *args,
        **kwargs
    ):
        if url.startswith("https://github.com"):
            cdn_url = "https://ghproxy.com/" + url
            return download_url_to_file(cdn_url, *args, **kwargs)
    hub.download_url_to_file = _proxy_download_url_to_file

    def _git_archive_link(repo_owner, repo_name, ref):
        return f"https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/{ref}.zip"
    hub._git_archive_link = _git_archive_link
    
