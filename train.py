import torch
from encoders import *
from depth import *
from Models import *
from time import time
from IO import *
from parameters import parameters
from Data_Generator import *
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    pass
# import devices

def main_train(index, args):
    print("Running main on TPU:{}".format(index))
    torch.manual_seed(2)
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
    # optimizer list
    Up_to_Down = []
    # Method which creates a module optimizer and add it to the given list
    def add_optimizer(module,model_optimizers=(),num_workers = 1): #needs a change for lr tpu
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adadelta(module.parameters(), lr=args.learning_rate*num_workers, weight_decay = 0.001)
        for direction in model_optimizers:
            direction.append(optimizer)
        return optimizer
    #Dataset
    train_dataset =  Dataset(input_left = args.real_left
            ,input_right = args.real_right, output_left = args.disp_left
            , output_right = args.disp_right)
    # data Sampler
    if args.tpu :
        world_size = xm.xrt_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=xm.get_ordinal(),
            shuffle=True)
        tpu_params = {'loader_prefetch_size':args.loader_prefetch_size,
                      'device_prefetch_size':args.device_prefetch_size}
    else:
        world_size = 1
        train_sampler = None
        tpu_params = {}
    #Dataset loader
    params_training = {'batch_size': args.batch_size,
          'shuffle': not train_sampler,
          'num_workers': args.num_workers,
          'drop_last': True,
          "sampler":train_sampler}
    params_validation = {'batch_size': args.batch_size_v,
          'shuffle': True ,
          'num_workers':  args.num_workers,
          'drop_last': True }
    data = {"training":Data_Generator(train_dataset,params_training,tpu=args.tpu,device= dev if args.tpu else None,tpu_params=tpu_params)}
    if args.validation:
        data["validation"] = Data_Generator(Dataset(input_left = args.real_left_v
                ,input_right = args.real_right_v, output_left = args.disp_left_v
                , output_right = args.disp_right_v),params_validation,tpu=args.tpu,device= dev if args.tpu else None,tpu_params=tpu_params)
    # model
    D = device(SDMU_Encoder(num_filters = parameters["num_filters"]
                                , kernels = parameters["kernels"]
                                , strides = parameters["strides"])) #parms to be filled
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
    start = time()
    training(loggers = [DispNet_logger]) #loggers to be defined
    print("Total training time taken:{0:.2f}s".format(time()-start))

class Trainer:

    def __init__(self, Data_Generator, optimizers, Network, schedule_coeff, batch_size):
        self.optimizers = optimizers
        self.Data_Generator = Data_Generator
        self.Network = Network
        self.schedule_coeff = schedule_coeff
        self.batch_size = batch_size
        self.EPE = 0
        self.sm = 0
        self.re = 0
        self.ds = 0
        self.em = 0
        self.IO_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.which = len(self.schedule_coeff)
        self.l = len(self.schedule_coeff)
        self.i = 0
        self.Data_Generator.reset_generator()
        self.data = self.Data_Generator.next_batch()
        self.tpu = self.Data_Generator.tpu
    def step(self,curr):
        #selects the loss
        if self.schedule_coeff[self.i][0]<curr:
            self.i = self.i+1
        self.which = self.l-self.i
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        t = time()
        try:
            data = next(self.data)
        except StopIteration:
            self.Data_Generator.reset_generator()
            self.data = self.Data_Generator.next_batch()
            data = next(self.data)
        self.IO_time += time() - t
        t = time()
        loss = self.Network.score(imgL = data[0], imgR = data[1], which=self.which, lp=self.i+1, train = True)
        self.forward_time += time() - t
        t = time()
        self.sm+= loss[0].detach()
        self.re+= loss[2].detach()
        self.ds+= loss[1].detach()
        self.em+= loss[3].detach()
        l = 0
        for i in loss:
            l+=i
        l = l.mul(self.schedule_coeff[self.i][1])
        l.backward()
        if self.tpu:
            for optimizer in self.optimizers:
                xm.optimizer_step(optimizer)#, barrier=True)
        else:
            for optimizer in self.optimizers:
                optimizer.step()

        self.backward_time += time() - t

    def total_time(self):
        return self.io_time+self.forward_time+self.backward_time

    def reset_stats(self):
        self.EPE = 0
        self.IO_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.sm = 0
        self.re = 0
        self.ds = 0
        self.em = 0

class Logger:

    def __init__(self,name, trainer, log_interval, validators = (), TPU_index = None):
        self.trainer = trainer
        self.validators = validators
        self.log_interval = log_interval
        self.name = name
        self.TPU_index = TPU_index

    def log(self):

        if self.trainer is not None or len(self.validator) > 0:
            print("{0}".format(self.name))

        if self.trainer is not None:
            io_time = self.trainer.IO_time
            forward_time = self.trainer.forward_time
            backward_time = self.trainer.backward_time
            which = self.trainer.which
            sm = self.trainer.sm.item()/self.log_interval
            re = self.trainer.re.item()/self.log_interval
            ds = self.trainer.ds.item()/self.log_interval
            em = self.trainer.em.item()/self.log_interval
            tot = sm+re+ds+em
            print(" -Training_loss: {0} -S: {1} -R: {2} -D: {3} -E: {4} -IO_time: {5:.2f}s -forward_time: {6:.2f}s -backward_time: {7:.2f}s -which_loss: pr_loss{8}".format(tot,sm,re,ds,em,io_time,forward_time,backward_time,which))
            self.trainer.reset_stats()

        for id, validator in enumerate(self.validators):
            t = time()
            EPE = validator.EPE
            validator.reset_stats()         #might need to change
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
