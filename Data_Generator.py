import torch.nn.functional as F
import torch
import numpy as np
import os
import torch.utils.data as data_util
from IO import *
import cv2
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import numpy as np
import timeit  #may not use in future
class Dataset(data_util.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, input_left, input_right,output_left,output_right):
        self.input_left_id = path_gen(input_left, os.listdir(input_left))
        self.input_right_id = path_gen(input_right, os.listdir(input_right))
        self.output_left_id = path_gen(output_left, os.listdir(output_left)) #initially not using
        self.output_right_id = path_gen(output_right, os.listdir(output_right))
        self.input_ID = list(zip(self.input_left_id,self.input_right_id))
        self.output_ID = list(zip(self.output_left_id,self.output_right_id))

  def __len__(self):
        return len(self.input_left_id)

  def normalizer(self,data):
      return cv2.normalize(data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  def resize(self,img,dim):
      return cv2.resize(img,dim)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        input_left_ID, input_right_ID = self.input_ID[index]
        output_left_ID, output_right_ID = self.output_ID[index]

        # Load data and get label

        X_left, X_right = self.normalizer(np.transpose(self.resize(read(input_left_ID),(768,384)), (2, 0, 1))), self.normalizer(np.transpose(self.resize(read(input_right_ID),(768,384)), (2, 0, 1)))
        #y_left = self.normalizer(read(output_left_ID))
        y_right = self.resize(self.normalizer(read(output_right_ID)),(384,192))
        X = np.concatenate((X_left, X_right),axis=0)
        # if is_null(X):                                              #for debugging
        #     print('null value on index:{}'.format(input_left_ID))
        #     assert False
        # if is_null(y_right):
        #     print('null value on index:{}'.format(input_left_ID))   #for debugging
        #     assert False
        return X, y_right

def path_gen(path,paths):
    new_paths = []
    for i in paths:
        new_paths.append(os.path.join(path,i))
    return new_paths
def  is_null(a):
        np.isnan(a).any()
class Data_Generator():
    def __init__(self,Dataset, params, tpu, device):
        self.Dataset = Dataset
        self.device = device
        self.tpu = tpu
        self.Data_loader = data_util.DataLoader(self.Dataset,**params)
    def reset_generator(self):
        if self.tpu:
            self.loader = pl.ParallelLoader(self.Data_loader, [self.device])
            self.generator = self.loader.per_device_loader(self.device)
        else:
            self.loader = self.Data_loader
            self.generator = self.loader
    def next_batch(self):
        for i,k in enumerate(self.generator):
            yield k

        # try:
        #     return next(self.generator)
        # except StopIteration:
        #     if self.tpu:
        #         self.generator = iter(self.loader.per_device_loader(self.device))
        #     else:
        #         self.generator = iter(self.loader)
        #     return next(self.generator)
