import train
import argparse

if __name__ == '__main__':
        # Build argument parser
    parser = argparse.ArgumentParser(description='Train a Disparity Network Model')

    # Training Dataset
    corpora_group = parser.add_argument_group('training dataset', 'provide location for each real image (left and right) and Disparity map(left and right)')
    corpora_group.add_argument('--real_left', help='The location of the folder containing the left real images.')
    corpora_group.add_argument('--real_right', help='The location of the folder containing the right real images.')
    corpora_group.add_argument('--disp_left', help='The location of the folder containing the left disparity map')  #edit
    corpora_group.add_argument('--disp_right', help='The location of the folder containing the right disparity map')

    # # Architecture (up_kernel, up_stride, in_channels, out_channels, padding)
    # architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    # architecture_group.add_argument('--arch_param', type=int, default=2, help="The location of the file containing architecure's parameters")
    # Data Pipieline Arguments
    data_pipeline_group = parser.add_argument_group("Data Pipeline","Let the user change the data pipeline arguments")
    data_pipeline_group.add_argument("--num_workers", type=int, default=1, help="number of workers to use to fetch the data into the ram")

    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=300000, help='the number of training iterations for initialization phase (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--model_path',default = "", help='Path of the folder to which the model will be saved ')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=1, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation' , default=False, action='store_true', help='use validation dataset for validation ')
    logging_group.add_argument('--real_left_v', help='The location of the folder containing the left real images.')
    logging_group.add_argument('--real_right_v', help='The location of the folder containing the right real images.')
    logging_group.add_argument('--disp_left_v', help='The location of the folder containing the left disparity map')  #edit
    logging_group.add_argument('--disp_right_v', help='The location of the folder containing the right disparity map')
    logging_group.add_argument('--batch_size_v', type=int, default=50, help='batch size of the validation data')

    # TPU related Arguments
    tpu_group = parser.add_argument_group("TPU","Arguments for TPU training")
    tpu_group.add_argument("--num_cores", type = int, default=8, help="Defines the number of TPU cores to use")
    tpu_group.add_argument("--loader_prefetch_size", type=int, default=8, help='Defines the loader prefetch queue size')
    tpu_group.add_argument("--device_prefetch_size", type=int, default=4, help='Defines the device prefetch size')

    # Other
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--tpu', default=False, action='store_true', help='use tpu')
    parser.add_argument('--c1', type=float, default=1, help='smooth loss')
    parser.add_argument('--c2', type=float, default=1, help='recon loss')
    parser.add_argument('--c3', type=float, default=1, help='dipsmi loss')
    parser.add_argument('--c4', type=float, default=1, help='edge loss')
    # Parse arguments
    args = parser.parse_args()
    if args.tpu:
        print("tpu enabled")
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(train.main_train, args=(args,), nprocs=args.num_cores)#, start_method='fork')
    else:
        train.main_train(0,args)
