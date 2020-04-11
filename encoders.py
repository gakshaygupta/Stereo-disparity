import torch.nn as nn
import torch.nn.functional as F
import torch
from misc.utils import Corr
def SeparableConv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=True):
    conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
    pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    return nn.Sequential(conv1,pointwise)
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

class SDMU_Encoder(nn.Module):
    def __init__(self, num_filters, kernels,strides,D=None):
        super().__init__()
        self.input_output = [(num_filters[x],num_filters[x+1]) for x in range(0,len(num_filters)-1)]
        self.kernels = kernels
        self.layer_parm = list(zip(self.input_output,self.kernels,strides))
        self.layer_parm[2] = ((2*self.layer_parm[2][0][0],self.layer_parm[2][0][1]),self.layer_parm[2][1],self.layer_parm[2][2])
        self.layer_list = nn.ModuleList([nn.Conv2d(x[0][0],x[0][1],x[1],x[2],padding=x[1]//2) for x in self.layer_parm])
        self.corr = Corr()
        self.relu = nn.ReLU()
    def forward(self,imgL,imgR):
        conv1L = self.layer_list[0](imgL)
        conv1L = self.relu(conv1L)
        conv2L = self.layer_list[1](conv1L)
        conv2L = self.relu(conv2L)
        conv1R = self.layer_list[0](imgR)
        conv1R = self.relu(conv1R)
        conv2R = self.layer_list[1](conv1R)
        conv2R = self.relu(conv2R)
        #corr = self.corr(conv2L,conv2R)
        corr = torch.cat([conv2L,conv2R],dim=1)
        conv_out = [torch.cat([conv1L,conv1R],1),torch.cat([conv2L,conv2R],1),corr]
        for conv in self.layer_list[2:]:
            conv_out.append(self.relu(conv(conv_out[-1])))
        return conv_out
