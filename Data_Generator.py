import torch.nn.functional as F
import torch
import numpy as np
import os
import torch.utils.data as data_util
from IO import *
import cv2
import numpy as np
import timeit  #may not use in future
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pass
class Dataset(data_util.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, input_left, input_right,output_left,output_right,validate=False,gama_range=[0.8,2.2],bright_range=[0.5,2],color_range=[0.8,1.2]):
        self.input_left_id = path_gen(input_left, os.listdir(input_left))
        self.input_right_id = path_gen(input_right, os.listdir(input_right))
        self.output_left_id = path_gen(output_left, os.listdir(output_left)) #initially not using
        self.output_right_id = path_gen(output_right, os.listdir(output_right))
        self.input_ID = list(zip(self.input_left_id,self.input_right_id))
        self.output_ID = list(zip(self.output_left_id,self.output_right_id))
        self.validate = validate
        self.gama_range = gama_range
        self.bright_range = bright_range
        self.color_range = color_range
  def __len__(self):
        return len(self.input_left_id)

  def normalizer(self,data):
      return cv2.normalize(data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  def resize(self,img,dim):
      return cv2.resize(img,dim)

  def profile_aug(self,imgL,imgR):
      r = np.random.uniform(low=0,high=1)
      if r>=0.5:
          gama = np.random.uniform(low = self.gama_range[0], high = self.gama_range[1])
          inv_gama = 1/gama
          imgL,imgR = imgL**inv_gama,imgR**inv_gama

          b = np.random.uniform(low = self.bright_range[0], high = self.bright_range[1])
          imgL,imgR = b*imgL,b*imgR

          shift = np.random.uniform(low = self.color_range[0],high = self.color_range[1],size=3)
          for i in range(3):
              imgL[:,:,i],imgR[:,:,i] = shift[i]*imgL[:,:,i],shift[i]*imgR[:,:,i]

      return np.clip(imgL,0,1),np.clip(imgR,0,1)

  def spatial_aug(self,imgL,imgR):
      r = np.random.uniform(low=0,high=1)
      if r>=0.5:
          ri = np.random.uniform(low=0,high=1)
          rr = np.random.uniform(low=0,high=1)
          if ri>=0.5:
              imgL,imgR = np.rot90(imgR,2),np.rot90(imgL,2)
          if rr>0.5:
              imgL,imgR, = np.flip(imgR,1),np.flip(imgL,1)
      return imgL,imgR

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        input_left_ID, input_right_ID = self.input_ID[index]
        output_left_ID, output_right_ID = self.output_ID[index]

        # Load data and get label
        imgL,imgR = cv2.imread(input_left_ID),cv2.imread(input_right_ID)
        # data augmentation

        imgL,imgR = self.resize(imgL,(768,384)),self.resize(imgR,(768,384))
        if not self.validate:
            imgL,imgR = self.spatial_aug(imgL,imgR)
        imgL,imgR = self.normalizer(imgL), self.normalizer(imgR)
        if not self.validate:
            imgL,imgR = self.profile_aug(imgL,imgR)
        imgL,imgR = np.transpose(imgL, (2, 0, 1)),np.transpose(imgR, (2, 0, 1))

        #y_left = self.normalizer(read(output_left_ID))
        #y_right = self.resize(self.normalizer(read(output_right_ID)),(384,192))

        # if is_null(X):                                              #for debugging
        #     print('null value on index:{}'.format(input_left_ID))
        #     assert False
        # if is_null(y_right):
        #     print('null value on index:{}'.format(input_left_ID))   #for debugging
        #     assert False
        return imgL,imgR

def path_gen(path,paths):
    new_paths = []
    for i in paths:
        new_paths.append(os.path.join(path,i))
    return new_paths
def  is_null(a):
        np.isnan(a).any()
class Data_Generator():
    def __init__(self,Dataset, params, tpu, device, tpu_params = {}):
        self.Dataset = Dataset
        self.device = device
        self.tpu = tpu

        self.tpu_params = tpu_params
        self.Data_loader = data_util.DataLoader(self.Dataset,**params)
    def reset_generator(self):
        if self.tpu:
            self.loader = pl.ParallelLoader(self.Data_loader, [self.device],**self.tpu_params)
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
