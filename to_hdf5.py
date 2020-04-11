import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import h5py
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from IO import *
import cv2
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description='for converting to hdf5')
benchmark = parser.add_argument_group('convert', 'provide location for all the files and set number of process to use etc')
benchmark.add_argument('--h5_path',default=None, help='location of hdf5 files')
benchmark.add_argument('--num_workers', type=int,default=0 ,help='Number of workers to use for data loader')
args = parser.parse_args()
def normalizer(data):
      return cv2.normalize(data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
def resize(img,dim):
      return cv2.resize(img,dim)
def path_gen(path):
    paths = os.listdir(path)
    new_paths = []
    for i in paths:
        new_paths.append(os.path.join(path,i))
    return new_paths
l_l = path_gen(r"/mnt/disks/dataset/FlyingThings3D_subset_webp_clean.tar.bz2/mnt/f/datasets/disp/new/Flyings3D_subset_webp_clean/train/image_clean/left_webp")
l_r = path_gen(r"/mnt/disks/dataset/FlyingThings3D_subset_webp_clean.tar.bz2/mnt/f/datasets/disp/new/Flyings3D_subset_webp_clean/train/image_clean/left_webp")
ld_l = path_gen(r"/mnt/disks/dataset/FlyingThings3D_subset_disparity.tar.bz2/FlyingThings3D_subset/train/disparity/left")
ld_r = path_gen(r"/mnt/disks/dataset/FlyingThings3D_subset_disparity.tar.bz2/FlyingThings3D_subset/train/disparity/right")
l = list(enumerate(zip(l_l,l_r,ld_l,ld_r)))
def hdf(im):
    index = im[0]
    print(index)
    img_l = normalizer(np.transpose(resize(cv2.imread(im[1][0]),(768,384)), (2, 0, 1)))
    img_r = normalizer(np.transpose(resize(cv2.imread(im[1][1]),(768,384)), (2, 0, 1)))
    X = [img_l,img_r]
    disp_l = resize(normalizer(read(im[1][2])),(384,192))
    disp_r = resize(normalizer(read(im[1][3])),(384,192))
    y = [disp_l,disp_r]
    with h5py.File("F:{0}/{1}.h5".format(args.h5_path,index),"w") as h:
        h.create_dataset('image',data = X)
        h.create_dataset('label',data = y)

from multiprocessing import Pool


if __name__ == '__main__':
    p = Pool(args.num_workers)
    print(p.map(hdf, l[:500]))