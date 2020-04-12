import torch.nn as nn
import torch.nn.functional as F
import torch
from misc.utils import Corr,SSIM,BiLinear
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
        output = self.U(self.D(input),6)
        return output

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

class SDMU(nn.Module):
    def __init__(self,D,U,device,c1,c2,c3,c4=0,edge_loss_b=True,max_disp=0.4):
        super().__init__()
        self.D = D
        self.U = U
        self.device = device
        self.SSIM = SSIM()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.edge_loss_b = edge_loss_b
        self.max_disp = max_disp
        self.image_warp = BiLinear(device = device)
        self.zero = torch.zeros([1])
    def _train(self,mode):
        self.D.train(mode)
        self.U.train(mode)

    def predict(self,input):  #data must come from dataloader
        output = self.U(self.D(input),6)
        return output

    def resize(self,input,factor,reduce=True):
        return F.interpolate(input, size=None, scale_factor=1/factor if reduce else factor, mode='bilinear', align_corners=False).clamp(min=0, max=1)

    def gradient_x(self, img):
        gx = img[:,:,:,:-1] - img[:,:,:,1:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:,:-1,:] - img[:,:,1:,:]
        return gy

    def generate_image_left(self, img, disp):
        return self.image_warp(img, -disp)

    def generate_image_right(self, img, disp):
        return self.image_warp(img, disp)

    def smooth_loss(self,dispL,dispR,imgL,imgR):
        imgL_gx = self.gradient_x(imgL)
        imgL_gy = self.gradient_y(imgL)
        imgR_gx = self.gradient_x(imgR)
        imgR_gy = self.gradient_y(imgR)
        dispL_gx = self.gradient_x(dispL).squeeze(1)
        dispL_gy = self.gradient_y(dispL).squeeze(1)
        dispR_gx = self.gradient_x(dispR).squeeze(1)
        dispR_gy = self.gradient_y(dispR).squeeze(1)
        with torch.no_grad():
            weightL_x = torch.exp(-torch.mean(torch.abs(imgL_gx),dim=1))
            weightL_y = torch.exp(-torch.mean(torch.abs(imgL_gy),dim=1))
            weightR_x = torch.exp(-torch.mean(torch.abs(imgR_gx),dim=1))
            weightR_y = torch.exp(-torch.mean(torch.abs(imgR_gy),dim=1))
        #print(dispR_gx.shape,weightR_x.shape)
        lossL = torch.mean(weightL_x*torch.abs(dispL_gx))+torch.mean(weightL_y*torch.abs(dispL_gy))
        lossR = torch.mean(weightR_x*torch.abs(dispR_gx))+torch.mean(weightR_y*torch.abs(dispR_gy))
        #print("smooth",lossL,lossR)
        return lossL+lossR

    def recon_loss(self,dispL,dispR,imgL,imgR,alpha = 0.85):
        gen_left = self.generate_image_left(imgR,dispL)
        gen_right = self.generate_image_right(imgL,dispR)
        lossL = torch.mean(alpha*self.SSIM(imgL,gen_left)+(1-alpha)*torch.abs(imgL-gen_left))
        lossR = torch.mean(alpha*self.SSIM(imgR,gen_right)+(1-alpha)*torch.abs(imgR-gen_right))
        #print("recon",lossL,lossR)
        return lossL+lossR

    def disp_sim(self,dispL,dispR):
        gen_dispL = self.generate_image_left(dispR,dispL)
        gen_dispR = self.generate_image_right(dispL,dispR)
        lossL = torch.mean(torch.abs(dispL-gen_dispL))
        lossR = torch.mean(torch.abs(dispR-gen_dispR))
        #print("dispsmi",lossL,lossR)
        return lossL+lossR

    def edge_loss(self,dispL,dispR,imgL,imgR,alpha=0.01):
        #add sgausian smoothning
        imgL_gx = self.gradient_x(imgL)
        imgL_gy = self.gradient_y(imgL)
        imgR_gx = self.gradient_x(imgR)
        imgR_gy = self.gradient_y(imgR)
        dispL_gx = self.gradient_x(dispL).squeeze(1)
        dispL_gy = self.gradient_y(dispL).squeeze(1)
        dispR_gx = self.gradient_x(dispR).squeeze(1)
        dispR_gy = self.gradient_y(dispR).squeeze(1)
        with torch.no_grad():
            weightL_x = torch.exp(-1/torch.mean(torch.abs(imgL_gx),dim=1))
            weightL_y = torch.exp(-1/torch.mean(torch.abs(imgL_gy),dim=1))
            weightR_x = torch.exp(-1/torch.mean(torch.abs(imgR_gx),dim=1))
            weightR_y = torch.exp(-1/torch.mean(torch.abs(imgR_gy),dim=1))
        lossL = torch.mean(weightL_x*torch.abs(1/(dispL_gx+alpha)))+torch.mean(weightL_y*torch.abs(1/(dispL_gy+alpha)))
        lossR = torch.mean(weightR_x*torch.abs(1/(dispR_gx+alpha)))+torch.mean(weightR_y*torch.abs(1/(dispR_gy+alpha)))
        #print("edge",lossL,lossR)
        return lossL+lossR

    def compute_loss(self,disp,imgL,imgR,r):
        dispL = disp[:,0,:,:].unsqueeze(1)
        dispR = disp[:,1,:,:].unsqueeze(1)
        smooth_loss = self.c1*self.smooth_loss(dispL,dispR,imgL,imgR)/r if self.c1>0 else self.zero
        recon_loss = self.c2*self.recon_loss(dispL,dispR,imgL,imgR) if self.c2>0 else self.zero
        disp_sim = self.c3*self.disp_sim(dispL,dispR) if self.c3>0 else self.zero
        edge_loss = self.c4*self.edge_loss(dispL,dispR,imgL,imgR)/r if self.edge_loss_b else self.zero
        return [smooth_loss,disp_sim,recon_loss,edge_loss]

    def score(self, imgL, imgR, which, lp, train=False):
        self._train(train)
        with torch.no_grad():
            imgL = self.device(imgL)
            imgR = self.device(imgR)
        intermediate = self.D(imgL,imgR)
        a = 1/0.6
        b = -1/3
        disp = [self.max_disp*(a*i+b) for i in self.U(intermediate, lp)] #B*H*W
        loss = [0]*4
        for i in range(0,6):
            imgLK,imgRK = self.resize(imgL,factor= 2**(6-i)), self.resize(imgR,factor= 2**(6-i))
            l = self.compute_loss(disp[i],imgLK,imgRK,r=2**(i))
            for j in range(0,4):
                loss[j]+=l[j]
        return loss
