import torch
import argparse
from down_conv import *
from up_convolution import *
from DispNet import *
from time import time
from IO import *
from parameters import parameters
from Data_Generator import *
import torch_xla
import torch_xla.core.xla_model as xm
import devices

def main_train():
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
    logging_group.add_argument('--validation' , default=False, help='use validation dataset for validation ')
    logging_group.add_argument('--real_left_v', help='The location of the folder containing the left real images.')
    logging_group.add_argument('--real_right_v', help='The location of the folder containing the right real images.')
    logging_group.add_argument('--disp_left_v', help='The location of the folder containing the left disparity map')  #edit
    logging_group.add_argument('--disp_right_v', help='The location of the folder containing the right disparity map')
    logging_group.add_argument('--batch_size_v', type=int, default=50, help='batch size of the validation data')

    # Other
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--tpu', default=False, action='store_true', help='use tpu')
    # Parse arguments
    args = parser.parse_args()
    # Select device
    device = devices.gpu if args.cuda else devices.cpu
    device = devices.tpu if args.tpu else device
    # optimizer list
    Up_to_Down = []
    # Method which creates a module optimizer and add it to the given list
    def add_optimizer(module,model_optimizers=()):
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
        for direction in model_optimizers:
            direction.append(optimizer)
        return optimizer
    #Datasets
    params_training = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers,
          'drop_last': True }
    params_validation = {'batch_size': args.batch_size_v,
          'shuffle': True,
          'num_workers':  args.num_workers,
          'drop_last': True }
    data = {"training":Data_Generator(Dataset(input_left = args.real_left
            ,input_right = args.real_right, output_left = args.disp_left
            , output_right = args.disp_right),params_training)}
    if args.validation:
        data["validation"] = Data_Generator(Dataset(input_left = args.real_left_v
                ,input_right = args.real_right_v, output_left = args.disp_left_v
                , output_right = args.disp_right_v),params_validation)
    # model
    D = device(Down_Convolution(num_filters = parameters["num_filters"]
                                , kernels = parameters["kernels"]
                                , strides = parameters["strides"])) #parms to be filled
    add_optimizer(D,[Up_to_Down])
    U = device(Up_Conv(up_kernels = parameters["up_kernels"]
                        , i_kernels = parameters["i_kernels"]
                        , pr_kernels = parameters["pr_kernels"]
                        , up_filters = parameters["up_filters"]
                        , down_filters = parameters["down_filters"]
                        , index = parameters["index"]
                        , pr_filters = parameters["pr_filters"]))          #parms to be filled
    add_optimizer(U,[Up_to_Down])
    DispNet_ = device(DispNet(D, U))
    #dataset
    DispNet_trainer = Trainer(Data_Generator = data["training"]
                            , optimizers = Up_to_Down
                            , Network = DispNet_
                            , schedule_coeff = parameters["schedule_coeff"]
                            ,batch_size = args.batch_size)
    trainers = [DispNet_trainer]
    def save_models(name,step):
        torch.save(DispNet_, '{0}{1}:DispNet_step_size_{2}.pth'.format(name, args.save,step))
        torch.save(D, '{0}{1}:Down_step_size_{2}.pth'.format(name, args.save,step))
        torch.save(U, '{0}{1}:UP_step_size_{2}.pth'.format(name, args.save,step))

    def training(loggers): # this may be used as intput to the swamp tpu
        for steps in range(1, args.iterations + 1):
            for trainer in trainers:
                trainer.step(steps)

            if steps % args.log_interval == 0:
                print('STEP {0} x {1}'.format(steps, args.batch_size))
                for logger in loggers:
                    logger.log()
            if steps%args.save_interval==0 and args.model_path!="":
                save_models(args.model_path,steps)
    #defining loggers
    DispNet_logger = Logger("DispNet", DispNet_trainer, log_interval = args.log_interval)
    #starting training
    training(loggers = [DispNet_logger]) #loggers to be defined

class Trainer:

    def __init__(self, Data_Generator, optimizers, Network, schedule_coeff, batch_size):
        self.optimizers = optimizers
        self.Data_Generator = Data_Generator
        self.Network = Network
        self.schedule_coeff = schedule_coeff
        self.batch_size = batch_size
        self.EPE = 0
        self.IO_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.which = len(self.schedule_coeff)
        self.l = len(self.schedule_coeff)
        self.i = 0

    def step(self,curr):
        #selects the loss
        if self.schedule_coeff[self.i][0]<curr:
            self.i = self.i+1
        self.which = self.l-self.i
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        t = time()
        data = self.Data_Generator.next_batch()
        self.IO_time += time() - t
        t = time()
        loss = self.schedule_coeff[self.i]*self.Network.score(input = data[0], output = data[1], which=self.which, train = True)
        self.forward_time += time() - t

        t = time()
        loss.backward()
        for optimizer in self.optimizers:
             xm.optimizer_step(optimizer, barrier=True)
        self.backward_time += time() - t

    def total_time(self):
        return self.io_time+self.forward_time+self.backward_time

    def reset_stats(self):
        self.EPE = 0
        self.IO_time = 0
        self.forward_time = 0
        self.backward_time = 0

class Logger:

    def __init__(self,name, trainer, log_interval, validators = ()):
        self.trainer = trainer
        self.validators = validators
        self.log_interval = log_interval
        self.name = name

    def log(self):

        if self.trainer is not None or len(self.validator) > 0:
            print("{0}".format(self.name))

        if self.trainer is not None:
            loss = self.trainer.EPE
            io_time = self.trainer.IO_time
            forward_time = self.trainer.forward_time
            backward_time = self.trainer.backward_time
            which = self.trainer.which
            print(" -Training_loss: {0:10.2f} -IO_time: {1:.2f}s -forward_time: {2:.2f}s -backward_time: {3:.2f}s -which_loss: pr_loss{4}".format(loss,io_time,forward_time,backward_time,which))
            self.trainer.reset_stats()

        for id, validator in enumerate(self.validators):
            t = time()
            EPE = validator.EPE
            validator.reset_stats()
            print(" - Validator_EPE: {0:10.2f}   total_time:({1:.2f}s)".format(EPE,time()-t))

class Validator:

    def __init__(self, Network, Data_Generator, batch_size):
        self.Network = Network
        self.batch_size = batch_size
        self.loader = Data_Generator.loader
        self.EPE = 0

    def validation(self):
        count = 0
        for i,data in enumerate(self.loader):
                self.EPE+= torch.sum(torch.tensor(self.Network.score(data[0],data[1]))).item()
                count+= 1
        return self.EPE/count

    def reset_stats(self):
        self.EPE = 0
