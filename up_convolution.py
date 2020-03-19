import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_xla
import torch_xla.core.xla_model as xm

def up_conv(up_kernel, up_stride, in_channels, out_channels, padding ):
      up_c = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = up_kernel, stride = up_stride, padding = padding)
      return nn.Sequential(up_c,nn.ReLU(inplace=True))

def conv(kernel,stride,in_channels,out_channels, padding):
      cnn  = nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = stride, padding = padding)
      return nn.Sequential(cnn,nn.ReLU(inplace=True))


def pre_disp(in_channels, kernel_size, stride, padding):
      prl = nn.Conv2d(in_channels, 1, kernel_size, stride = stride, padding = padding)
      return nn.Sequential(prl,nn.Sigmoid())

def interpolate(input, scale=2):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)

class Up_Conv(nn.Module):

    def __init__(self, up_kernels, i_kernels, pr_kernels, up_filters, down_filters, index, pr_filters, up_stride = 2, i_stride = 1, pr_stride = 1):
            super().__init__()
            self.index = index
            self.in_out_up = [(up_filters[x],up_filters[x+1]) for x in range(0,len(up_filters)-1)]
            self.up_conv_list = nn.ModuleList([up_conv(x[1], up_stride, x[0][0], x[0][1],padding = 1) for x in zip(self.in_out_up,up_kernels)])
            self.in_out_i = [(up_filters[x]+down_filters[-(x)]+1,up_filters[x]) for x in range(1,len(up_filters))]
            self.i_conv_list = nn.ModuleList([conv(x[1], i_stride, x[0][0], x[0][1],padding = 1) for x in zip(self.in_out_i,i_kernels)])
            self.prl_list = nn.ModuleList([pre_disp(x[1],x[0],stride = pr_stride, padding = 1) for x in zip(pr_kernels,pr_filters)])

    def forward(self,down_out,index):
            prob6 = self.prl_list[0](down_out[-1])
            final_list = list(zip(self.up_conv_list,self.i_conv_list,self.prl_list[1:]))
            out = prob6
            in_ = down_out[-1]
            for i,l in enumerate(final_list):
                if i==(index-1):
                    break
                # print(interpolate(out[-1]).shape,out[-1].shape ,l[0](in_).shape,down_out[self.index[i]].shape,i)
                out = out.detach()
                in_ = l[1](torch.cat([interpolate(out),l[0](in_),down_out[self.index[i]]],1))
                out = l[2](in_)
            out = out.squeeze(1)
            return out
