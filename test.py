import torch
from encoders import *
from depth import *
from Models import *
from time import time
from IO import *
from parameters import parameters
from Data_Generator import *
from misc.utils import SSIM

def main():

        parser = argparse.ArgumentParser(description='Test a Disparity Network Model')

        # Testing Dataset
        corpora_group = parser.add_argument_group('testing dataset', 'provide location for each real image (left and right) and Disparity map(left and right)')
        corpora_group.add_argument('--real_left', help='The location of the folder containing the left real images.')
        corpora_group.add_argument('--real_right', help='The location of the folder containing the right real images.')
        corpora_group.add_argument('--disp_left', help='The location of the folder containing the left disparity map')  #edit
        corpora_group.add_argument('--disp_right', help='The location of the folder containing the right disparity map')

        # Data Pipieline Arguments
        data_pipeline_group = parser.add_argument_group("Data Pipeline","Let the user change the data pipeline arguments")
        data_pipeline_group.add_argument("--num_workers", type=int, default=1, help="number of workers to use to fetch the data into the ram")

        # Model restoring
        model_restorin_group = parser.add_argument_group("Model Restoring","Let the user restore the saved model")
        model_restorin_group.add_argument("--model_pth", type=str, default="", help="Location  of the saved model")

        # Other
        parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
        parser.add_argument('--tpu', default=False, action='store_true', help='use tpu')
        parser.add_argument('--c1', type=float, default=1, help='smooth loss')
        parser.add_argument('--c2', type=float, default=1, help='recon loss')
        parser.add_argument('--c3', type=float, default=1, help='dipsmi loss')
        parser.add_argument('--c4', type=float, default=1, help='edge loss')
        # Parse arguments
        args = parser.parse_args()

        D = device(SDMU_Encoder(num_filters = parameters["num_filters"]
                                    , kernels = parameters["kernels"]
                                    , strides = parameters["strides"]
                                    , corr=args.corr
                                    , D=args.D)) #parms to be filled
        add_optimizer(D,[Up_to_Down],num_workers=world_size)
        U = device(SDMU_Depth(up_kernels = parameters["up_kernels"]
                            , i_kernels = parameters["i_kernels"]
                            , pr_kernels = parameters["pr_kernels"]
                            , up_filters = parameters["up_filters"]
                            , down_filters = parameters["down_filters"]
                            , index = parameters["index"]
                            , pr_filters = parameters["pr_filters"]))          #parms to be filled
        add_optimizer(U,[Up_to_Down],num_workers=world_size)
        DispNet_ = device(SDMU(D, U,device,args.c1,args.c2,args.c3,args.c4))
        DispNet_.load_state_dict(torch.load(args.model_pth))
        DispNet_.eval()
