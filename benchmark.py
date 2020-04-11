import torch.nn.functional as F
import torch
import numpy as np
import os
import torch.utils.data as data_util
from IO import *
import cv2
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser(description='For benchmarking the disk')
#benchmarking
benchmark = parser.add_argument_group('Benchmark', 'provide location for all the files and set number of process to use etc')
benchmark.add_argument('--n_path',default=None, help='base path for all the files except for hdf5 files')
benchmark.add_argument('--h5_path',default=None, help='location of hdf5 files')
benchmark.add_argument('--batch_size', type=int, help='Bathc size')  #edit
benchmark.add_argument('--num_workers', type=int,default=0 ,help='Number of workers to use for data loader')
args = parser.parse_args()
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

        X_left, X_right = self.normalizer(np.transpose(self.resize(cv2.imread(input_left_ID),(768,384)), (2, 0, 1))), self.normalizer(np.transpose(self.resize(cv2.imread(input_right_ID),(768,384)), (2, 0, 1)))
        y_left = self.resize(self.normalizer(read(output_left_ID)),(384,192))
        y_right = self.resize(self.normalizer(read(output_right_ID)),(384,192))
        X = np.concatenate((X_left, X_right),axis=0)
        # if is_null(X):                                              #for debugging
        #     print('null value on index:{}'.format(input_left_ID))
        #     assert False
        # if is_null(y_right):
        #     print('null value on index:{}'.format(input_left_ID))   #for debugging
        #     assert False
        return X, [y_right,y_left]

def path_gen(path,paths):
    new_paths = []
    for i in paths:
        new_paths.append(os.path.join(path,i))
    return new_paths
def  is_null(a):
        np.isnan(a).any()
class Data_Generator():
    def __init__(self,Dataset, params):
        self.Dataset = Dataset
        self.Data_loader = data_util.DataLoader(self.Dataset,**params)
    def reset_generator(self):
            self.loader = self.Data_loader
            self.generator = self.loader
    def next_batch(self):
        for i,k in enumerate(self.generator):
            yield k
class Dataset2(data_util.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, hdf5_path):
          self.paths = self.path_gen(hdf5_path)

  def __len__(self):
        return len(self.paths)

  def normalizer(self,data):
      return cv2.normalize(data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  def resize(self,img,dim):
      return cv2.resize(img,dim)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.paths[index]

        # Load data and get label
        with h5py.File(file_path,"r") as h:
            X_left, X_right = h["image"].value
            y_left,y_right = h["label"].value
        X = np.concatenate((X_left, X_right),axis=0)
        # if is_null(X):                                              #for debugging
        #     print('null value on index:{}'.format(input_left_ID))
        #     assert False
        # if is_null(y_right):
        #     print('null value on index:{}'.format(input_left_ID))   #for debugging
        #     assert False
        return X, [y_right,y_left]
  def path_gen(self,path):
        paths = os.listdir(path)
        new_paths = []
        for i in paths:
            new_paths.append(os.path.join(path,i))
        return new_paths
if __name__ == "__main__":
    params_training = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers,
              'drop_last': True,
              "sampler":None}
    if args.n_path:
        img_l = "/mnt/disks/dataset/FlyingThings3D_subset_webp_clean.tar.bz2/mnt/f/datasets/disp/new/Flyings3D_subset_webp_clean/train/image_clean/left_webp"
        img_r = "/mnt/disks/dataset/FlyingThings3D_subset_webp_clean.tar.bz2/mnt/f/datasets/disp/new/Flyings3D_subset_webp_clean/train/image_clean/left_webp"
        dis_l = "/mnt/disks/dataset/FlyingThings3D_subset_disparity.tar.bz2/FlyingThings3D_subset/train/disparity/left"
        dis_r = "/mnt/disks/dataset/FlyingThings3D_subset_disparity.tar.bz2/FlyingThings3D_subset/train/disparity/right"
        dataset = Dataset(img_l,img_r,dis_l,dis_r)
        g = Data_Generator(dataset,params_training)
        g.reset_generator()
        l = g.next_batch()
        next(l)
        tot=0
        from time import time
        for i in range(10):
            t = time()
            k = next(l)
            tt = time() - t
            print(i,":",tt)
            tot+=tt
        print("average time for bath of {0} without hdf5:{1}".format(args.batch_size,tot/10))
    if args.h5_path:
        dataset2 = Dataset2(args.h5_path)
        g = Data_Generator(dataset,params_training)
        g.reset_generator()
        l = g.next_batch()
        next(l)
        tot=0
        from time import time
        for i in range(10):
            t = time()
            k = next(l)
            tt = time() - t
            print(i,":",tt)
            tot+=tt
        print("average time for bath of {0} with hdf5:{1}".format(args.batch_size,tot/10))