from  devices import *
from up_convolution import *
from down_conv import *
from parameters import *
device = gpu
U = device(Up_Conv(up_kernels = parameters["up_kernels"], i_kernels = parameters["i_kernels"], pr_kernels = parameters["pr_kernels"], up_filters = parameters["up_filters"], down_filters = parameters["down_filters"], index = parameters["index"], pr_filters = parameters["pr_filters"]))          #parms to be filled
U = device(Up_Conv(up_kernels = parameters["up_kernels"]
                    , i_kernels = parameters["i_kernels"]
                    , pr_kernels = parameters["pr_kernels"]
                    , up_filters = parameters["up_filters"]
                    , down_filters = parameters["down_filters"]
                    , index = parameters["index"]
                    , pr_filters = parameters["pr_filters"]))
D = device(Down_Convolution(num_filters = parameters["num_filters"]
                            , kernels = parameters["kernels"]
                            , strides = parameters["strides"])) #parms to be filled
data2, scale2 = readPFM(r"F:\datasets\disp\FlyingThings3D_subset\train\disparity\right\0000006.pfm")
from IO import *
import cv2
img, scale  = read("F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left\0000006.png")
img, scale  = read(r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left\0000006.png")
img, scale  = read(r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left\0006.png")
img= read(r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left\0006.png")
cv2.imshow("frame2",np.transpose(img, (1, 0, 2)).shape)
cv2,imshow("f",img)
cv2.imshow("f",img)
cv2.imshow("frame2",np.transpose(img, (1, 0, 2)).shape)
cv2.imshow("frame2",np.transpose(img, (1, 0, 2)))
cv2.imshow("frame2",np.transpose(img, (2, 1, 0)))
from Data_Generator import *
data = Dataset("F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left","F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\right","F:\datasets\disp\FlyingThings3D_subset\train\disparity\right","F:\datasets\disp\FlyingThings3D_subset\train\disparity\right")
data = Dataset("F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left",r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\right","F:\datasets\disp\FlyingThings3D_subset\train\disparity\right","F:\datasets\disp\FlyingThings3D_subset\train\disparity\right")
data = Dataset("F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left",r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\right",r"F:\datasets\disp\FlyingThings3D_subset\train\disparity\right",r"F:\datasets\disp\FlyingThings3D_subset\train\disparity\right")
params_training = {'batch_size': 1,
      'shuffle': True,
      'num_workers': 0,
      'drop_last': True }
lo = Data_Generator(data,params_training)
k = lo.next_batch()
k[0].shape
k[1][0]
k[1][0].shape
k[1][1].shape
from Data_Generator import *
data = Dataset("F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\left",r"F:\datasets\disp\Sampler\FlyingThings3D\RGB_cleanpass\right",r"F:\datasets\disp\FlyingThings3D_subset\train\disparity\right",r"F:\datasets\disp\FlyingThings3D_subset\train\disparity\right")
params_training = {'batch_size': 1,
      'shuffle': True,
      'num_workers': 0,
      'drop_last': True }
lo = Data_Generator(data,params_training)
k = lo.next_batch()
k[0].shape
out = U(D(device(k[0])))
cv2.imshow("f",255*out[-1].squeeze(0).cpu().detach().numpy())