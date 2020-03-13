import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_xla
import torch_xla.core.xla_model as xm

class DispNet(nn.Module):
    def __init__(self,down,up,device):
        super().__init__()
        self.D = down
        self.U = up
        self.BCE_criterion = nn.BCELoss(size_average=False)
        self.MSE = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.device = device
    def _train(self, mode):
        self.D.train(mode)
        self.U.train(mode)

    def predict(self,input):  #data must come from dataloader
        output = self.U(self.D(input))
        return output[-1]

    def EPE(self,input_disp, target_disp):
        return torch.norm(target_disp-input_disp,p=2,dim=(1,2)).mean()

    def L2(self,input,target):
        return self.MSE(input,target)

    def resize(self,input,factor,reduce=True):
        return F.interpolate(input.unsqueeze(1), size=None, scale_factor=1/factor if reduce else factor, mode='bilinear', align_corners=False).clamp(min=0, max=1).squeeze(1)

    def score(self, input, output, which, lp, train=False):
        self._train(train)
        intermediate = self.D(self.device(input))
        prob_output = self.U(intermediate, lp) #B*H*W
        output = self.resize(output,factor= 2**(which-1))
        return self.EPE(self.device(output),prob_output)
