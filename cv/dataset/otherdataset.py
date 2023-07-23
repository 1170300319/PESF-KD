from torchvision import transforms, datasets
import numpy as np
import torch
import json
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset.cifar100 import Tmpdataset
import torch


class Dataset():
    def __init__(self, data, targets, transforms=[], batch_size=128, shuffle=True,
                 ):
        assert (len(data) == len(targets))
        self.length = len(data)
        self.batch_size = batch_size
        self.transforms = transforms
        self.permutation = torch.arange(self.length)
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.permutation = torch.randperm(self.length)
        for i in range(self.n_batches):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), \
                      self.targets[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in
                                   self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), self.targets[
                          self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]

    def __len__(self):
        return self.n_batches



def iterator(data, target, transforms, forcecpu=False, shuffle=True, use_hd=False, is_instance=False):
    #return Dataset(data, target, transforms, shuffle=shuffle)
    return Tmpdataset(data, target, transforms, is_instance=is_instance)

import random


from PIL import Image


def cifarfs(use_hd=True, data_augmentation=True, dataset_path=None):
    """
    CIFAR FS dataset
    Number of classes :
    - train: 64
    - val  : 16
    - novel: 20
    Number of samples per class: exactly 600
    Total number of images: 60000
    Images size : 32x32
    """
    datasets = {}
    classes = []
    total = 60000
    buffer = {'train': 0, 'val': 64, 'test': 64 + 16}
    for metaSub in ["meta-train", "meta-val", "meta-test"]:
        subset = metaSub.split('-')[-1]
        data = []
        target = []
        subset_path = os.path.join(dataset_path, 'cifar_fs', metaSub)
        classe_files = os.listdir(subset_path)

        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            count = 0
            for file in files:
                count += 1
                target.append(c + buffer[subset])

                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                else:
                    data.append(path)

        datasets[subset] = [data, torch.LongTensor(target)]

    assert (len(datasets['train'][0]) + len(datasets['val'][0]) + len(
        datasets['test'][0]) == total), 'Total number of sample per class is not 600'

    image_size = 32
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(image_size),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(),
                                           norm)

    all_transforms = torch.nn.Sequential(transforms.Resize([int(1.15 * image_size), int(1.15 * image_size)]),
                                         transforms.CenterCrop(image_size),
                                         norm)

    train_loader = iterator(datasets['train'][0], datasets['train'][1], transforms=train_transforms, forcecpu=True,
                                use_hd=use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)

    return (train_loader, train_clean, val_loader, test_loader), [3, image_size, image_size], (
    64, 16, 20, 600), True, False


def fc100(rank, worldsize, use_hd=False, num_workers=8, batch_size=128, dataset_path=None):
    """
    fc100 dataset
    Number of classes :
    - train: 60
    - val  : 20
    - novel: 20
    Number of samples per class: exactly 600
    Total number of images: 60000
    Images size : 84x84
    """
    datasets = {}
    total = 60000
    buffer = {'train': 0, 'val': 60, 'test': 60 + 20}
    for subset in ['train', 'val', 'test']:
        data = []
        target = []
        subset_path = os.path.join(dataset_path, subset)
        classe_files = os.listdir(subset_path)

        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            for file in files:
                target.append(c + buffer[subset])
                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    #image = np.asarray(Image.open(path).convert('RGB'))
                    data.append(image)
                else:
                    data.append(path)
        datasets[subset] = [data, torch.LongTensor(target)]

    assert (len(datasets['train'][0]) + len(datasets['val'][0]) + len(
        datasets['test'][0]) == total), 'Total number of sample per class is not 1300'
    print('len, ', len(datasets["train"][0]))
    print('shape, ', datasets["train"][0][0].shape)
    image_size = 32
    #norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           norm])

    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        norm])
    train_set = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=False,
                                use_hd=use_hd, shuffle=False, is_instance=True)
    train_sampler = DistributedSampler(train_set, shuffle=True, num_replicas=int(worldsize), rank=rank)
    train_loader = DataLoader(train_set, sampler=train_sampler,
                              batch_size=int(batch_size),
                              shuffle=False,
                              num_workers=int(num_workers))

    test_set = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=False,
                           shuffle=False, use_hd=use_hd, is_instance=False)
    test_sampler = DistributedSampler(test_set, shuffle=True, num_replicas=int(worldsize), rank=rank)
    test_loader = DataLoader(test_set, sampler=test_sampler,
                              batch_size=int(batch_size/2),
                              shuffle=False,
                              num_workers=int(num_workers/2))
    # [3, 84, 84], (60, 20, 20, 600), True, False
    # return (train_loader, val_loader, test_loader)
    return train_loader, test_loader, 600*12, train_sampler, test_sampler


def get_dataset(dataset_name):
    if dataset_name.lower() == "cifarfs":
        return cifarfs(data_augmentation=True)
    elif dataset_name.lower() == "fc100":
        return fc100()
    else:
        print("Unknown dataset!")


print("datasets, ", end='')
