import torch.nn as nn
from torch.functional import F
import torch
class BiLinear(nn.Module):
    def __init__(self,device, padding_mode='reflection'):
        super().__init__()
        self.padding_mode = padding_mode
        self.device = device
        self.shape = []
        self.X = []
        self.Y = []
    def forward(self, img, depth):

        # img: the source image (where to sample pixels) -- [B, 3, H, W]
        # depth: depth map of the target image -- [B, 1, H, W]
        # Returns: Source image warped to the target image
        if not self.shape ==depth.size():
            b, _, h, w = depth.size()
            i_range = self.device(torch.linspace(-1.0, 1.0,h,requires_grad = False)) # [1, H, W]  copy 0-height for w times : y coord
            j_range = self.device(torch.linspace(-1.0, 1.0,w,requires_grad = False)) # [1, H, W]  copy 0-width for h times  : x coord

            # pixel_coords = device(torch.stack((j_range, i_range), dim=1).float())  # [1, 2, H, W]
            # batch_pixel_coords = pixel_coords[:,:,:,:].expand(b,2,h,w).contiguous().view(b, 2, -1)  # [B, 2, H*W]
            X, Y = torch.meshgrid([i_range,j_range])
            X = X.expand(b,1,-1,-1) # [B, H*W]
            Y = Y.expand(b,1,-1,-1)
            self.X = X
            self.Y = Y
            X = X+depth
            X = X.squeeze(1)
            Y = Y.squeeze(1)
        else:
            X = self.X
            X+=depth
            Y = self.Y
                                                    # [B, H*W, 2]
        pixel_coords = torch.stack([X,Y],dim=3)  # [B, H, W, 2]

        projected_img = torch.nn.functional.grid_sample(img, pixel_coords, padding_mode=self.padding_mode)

        return projected_img
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
       From https://github.com/nianticlabs/monodepth2
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
# class Corr(nn.Module):
#     #source https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/main.py
#     def __init__(self,max_disp=40):
#         super(Corr, self).__init__()
#         self.max_disp = max_disp
#     def forward(self,x,y):
#         corr_tensors = []
#         for i in range(-self.max_disp, 0, 1):
#             s1 = y.narrow(3, 0, y.shape[3] + i)
#             shifted = F.pad(s1,(-i, 0, 0, 0), "constant", 0.0)
#             corr = torch.mean(shifted * x, 1)
#             corr_tensors.append(corr)
#         for i in range(self.max_disp + 1):
#             s2 = x.narrow(3, i, x.shape[3] - i)
#             shifted = F.pad(s2,(0, i, 0, 0), "constant", 0.0)
#             corr = torch.mean(shifted * y, 1)
#             corr_tensors.append(corr)
#
#         temp = torch.stack(corr_tensors)
#         out = torch.transpose(temp, 0, 1)
#         return out
class Corr(nn.Module):
    # Modified version of source https://github.com/wyf2017/DSMnet/blob/b61652dfb3ee84b996f0ad4055eaf527dc6b965f/models/util_conv.py#L56
    def __init__(self, stride=1, D=40, simfun=None):
        super(Corr, self).__init__()
        self.stride = stride
        self.D = D
        if(simfun is None):
            self.simfun = self.simfun_default
        else: # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun
    def simfun_default(self, fL, fR):
        return torch.sum(fL*fR,dim=1)

    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        corrmap = torch.zeros(bn, self.D*2+1, h, w).type_as(fL.data)
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, self.D):
            if(i >= w): break
            idx = i*self.stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
            corrmap[:, -i, :, idx:] = self.simfun(fR[:, :, :, idx:], fL[:, :, :, :-idx])
        return corrmap

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
       From https://github.com/nianticlabs/monodepth2
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
