import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_xla
import torch_xla.core.xla_model as xm

class Down_Convolution(nn.Module):

    def __init__(self, num_filters, kernels,strides):
        super().__init__()
        self.input_output = [(num_filters[x],num_filters[x+1]) for x in range(0,len(num_filters)-1)]
        self.kernels = kernels
        self.layer_parm = list(zip(self.input_output,self.kernels,strides))
        self.layer_list = nn.ModuleList([nn.Conv2d(x[0][0],x[0][1],x[1],x[2],padding = x[1]//2) for x in self.layer_parm])
        self.relu = nn.ReLU()

    def forward(self,input):
        conv_out = [self.relu(self.layer_list[0](input))]
        for conv in self.layer_list[1:]:
            conv_out.append(self.relu(conv(conv_out[-1])))
        return conv_out
