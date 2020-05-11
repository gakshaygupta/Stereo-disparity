import torch
from encoders import *
from depth import *
from Models import *
from time import time
from IO import *
from parameters import parameters
from Data_Generator import *
from misc.utils import SSIM
import matplotlib.pyplot as plt
import cv2
import argparse

def main():

    parser = argparse.ArgumentParser(description='Test a Disparity Network Model')

    # Testing Dataset
    corpora_group = parser.add_argument_group('testing dataset', 'provide location for each real image (left and right) and Disparity map(left and right)')
    corpora_group.add_argument('--real_left',default="", help='The location of the folder containing the left real images.')
    corpora_group.add_argument('--real_right',default="", help='The location of the folder containing the right real images.')
    corpora_group.add_argument('--disp_left',default="", help='The location of the folder containing the left disparity map')  #edit
    corpora_group.add_argument('--disp_right',default="", help='The location of the folder containing the right disparity map')
    corpora_group.add_argument('--left_img',default="", help='The location of the folder containing the left disparity map')  #edit
    corpora_group.add_argument('--right_img',default="", help='The location of the folder containing the right disparity map')
    # Data Pipieline Arguments
    data_pipeline_group = parser.add_argument_group("Data Pipeline","Let the user change the data pipeline arguments")
    data_pipeline_group.add_argument("--num_workers", type=int, default=1, help="number of workers to use to fetch the data into the ram")
    data_pipeline_group.add_argument("--batch_size", type=int, default=10, help="batch size")

    # Model restoring
    model_restorin_group = parser.add_argument_group("Model Restoring","Let the user restore the saved model")
    model_restorin_group.add_argument("--model_pth", type=str, default="", help="Location  of the saved model")

    # Architecture (up_kernel, up_stride, in_channels, out_channels, padding)
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--corr', default=False,action='store_true', help="Enables the correlation layer")
    architecture_group.add_argument('--D', type=int, default=40, help="Hyperparameter depicting the max dispacement the correlation layer can perform")

    # Other
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--tpu', default=False, action='store_true', help='use tpu')
    parser.add_argument('--c1', type=float, default=1, help='smooth loss')
    parser.add_argument('--c2', type=float, default=1, help='recon loss')
    parser.add_argument('--c3', type=float, default=1, help='dipsmi loss')
    parser.add_argument('--c4', type=float, default=1, help='edge loss')
    # Parse arguments
    args = parser.parse_args()
    try:
        dev = xm.xla_device()
    except:
        pass
    def cpu(x):
        return x.cpu() if x is not None else None

    def tpu(x):
        return x.to(dev)

    def gpu(x):
        return x.cuda() if x is not None else None
    # Select device
    device = gpu if args.cuda else cpu
    device = tpu if args.tpu else device

    D = device(SDMU_Encoder(num_filters = parameters["num_filters"]
                                , kernels = parameters["kernels"]
                                , strides = parameters["strides"]
                                , corr=args.corr
                                , D=args.D)) #parms to be filled
    U = device(SDMU_Depth(up_kernels = parameters["up_kernels"]
                        , i_kernels = parameters["i_kernels"]
                        , pr_kernels = parameters["pr_kernels"]
                        , up_filters = parameters["up_filters"]
                        , down_filters = parameters["down_filters"]
                        , index = parameters["index"]
                        , pr_filters = parameters["pr_filters"]))          #parms to be filled
    DispNet_ = device(SDMU(D, U,device,args.c1,args.c2,args.c3,args.c4))
    DispNet_.load_state_dict(torch.load(args.model_pth)["model_state_dict"])
    DispNet_.eval()
    if not args.real_left=="":
        params_validation = {'batch_size': args.batch_size,
              'shuffle': True ,
              'num_workers':  args.num_workers,
              'drop_last': True }
        val_dataset = Dataset(input_left = args.real_left
                              , input_right = args.real_right
                              , output_left = args.disp_left
                              , output_right = args.disp_right
                              , validate = True)
        data["validation"] = Data_Generator(val_dataset,params_validation,tpu=args.tpu,device= dev if args.tpu else None)
        data["validation"].reset_generator()
        disp = None
        for imgL,imgR in data["validation"].generator:
            if disp!=None:
                with torch.no_grad():
                    disp = torch.cat([dispL,DispNet_.predict(imgL,imgR)],0)
            else:
                    disp = DispNet_.predict(imgR,imgL)
        dispL = disp[:,0,:,:].squeeze(0)
        dispR = disp[:,1,:,:].squeeze(0)
        d = np.array(torch.cat([dispL,dispR],0).cpu())

        np.save("img_tot.npy",d)
    else:

        def normalizer(data):
              return cv2.normalize(data, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        def resize(img,dim):
              return cv2.resize(img,dim)
        imgL,imgR = cv2.imread(args.left_img),cv2.imread(args.right_img)
        imgL,imgR = resize(imgL,(768,384)),resize(imgR,(768,384))
        imgL,imgR = normalizer(imgL), normalizer(imgR)
        imgL,imgR = torch.tensor(imgL).permute(2,0,1).unsqueeze(0),torch.tensor(imgR).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            disp = DispNet_.predict(imgL,imgR)
        dispL = disp[:,0,:,:].squeeze(0)
        dispR = disp[:,1,:,:].squeeze(0)
        d = np.array(torch.cat([dispL,dispR],0).cpu())

        np.save("img.npy",d)

if __name__ == '__main__':
    main()
