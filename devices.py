import torch_xla
import torch_xla.core.xla_model as xm

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


default = cpu
