import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path
import pandas as pd
import json
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return {k: torch.from_numpy(sample[k]).float().unsqueeze(0) for k in sample.keys()}
        

class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        path = Path(args.dataset_dir)
        meta = pd.read_csv(path / 'train-ref.csv')
        self.input_list = list(meta['inp'])
        self.gt_list = list(meta['gt'])
        self.ref_list = list(meta['ref'])
        self.refsr_list = list(meta['refsr'])

        self.input_list = [str(path / 'train' / Path(item).name)
                           for item in self.input_list]
        self.gt_list = [str(path / 'train' / Path(item).name) for item in self.gt_list]
        self.ref_list = [str(path / 'ref' / Path(item).name) for item in self.ref_list]
        self.refsr_list = [str(path / 'ref' / Path(item).name)
                           for item in self.refsr_list]
        self.transform = transform

    def __len__(self):
        # return 9
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.gt_list[idx])
        h, w = HR.shape[:2]
        ### LR and LR_sr
        LR = imread(self.input_list[idx])
        LR_sr = np.array(Image.fromarray(
            LR).resize((w//2, h//2), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        Ref_sr = imread(self.refsr_list[idx])
        h2, w2 = Ref_sr.shape[:2]
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize(
            (w2//2, h2//2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ValSet(TrainSet):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        path = Path(args.dataset_dir)
        meta = pd.read_csv(path / 'test-ref.csv')
        self.input_list = list(meta['inp'])
        self.gt_list = list(meta['gt'])

        self.input_list = [str(path / 'test' / Path(item).name)
                           for item in self.input_list]
        self.gt_list = [str(path / 'test' / Path(item).name) for item in self.gt_list]

        self.transform = transform

    def __len__(self):
        # return 9
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.gt_list[idx])
        h, w = HR.shape[:2]
        ### LR and LR_sr
        LR = imread(self.input_list[idx])      
        LR_sr = np.array(Image.fromarray(
            LR).resize((w//2, h//2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

class TestSet(Dataset):
    def __init__(self, dataset_dir=None, meta=None, has_gt=True, transform=transforms.Compose([ToTensor()])):
        self.path = Path(dataset_dir)    
        self.has_gt = has_gt    
        with open(Path(meta)) as json_file:
            data = json.load(json_file)
        self.input_list = []
        for ID, lsts in data.items():
            for lst in lsts:
                x1, x2, y1, y2 = lst
                self.input_list.append([ID, x1, x2, y1, y2])

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ID, x1, x2, y1, y2 = self.input_list[idx]

        LR = imread(self.path / (ID+"_lr.png"))
        
        LR = LR[x1:x2, y1:y2]
        if self.has_gt:
            HR = imread(self.path / (ID+"_hr.png"))
            HR = HR[x1:x2, y1:y2]
        assert LR.shape == (160, 160), '{}'.format(LR.shape)
        w, h = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(LR).resize((w//2, h//2), Image.BICUBIC))

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr}
        if self.has_gt:
            HR = HR / 127.5 - 1.
            sample['HR'] = HR
        if self.transform:
            sample = self.transform(sample)
        return sample