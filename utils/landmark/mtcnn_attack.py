import math
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch_mtcnn.get_nets import PNet
model_urls = dict(pnet='https://github.com/khrlimam/mtcnn-pytorch/releases/download/0.0.1/pnet-6b6ef92b.pth')

def load_state(arch, progress=True):
    state = load_state_dict_from_url(model_urls.get(arch), progress=progress)
    return state

class PNet_attack(PNet):
    def __init__(self, device):
        super(PNet_attack, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))
        
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
        
        state_dict = load_state('pnet')
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}
        self.load_state_dict(filtered_state_dict, strict=False)
        self.float()
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.features(x)
        a = self.conv4_1(x)
        return F.softmax(a, dim=1)

def run_first_stage(image, net, scale):
    _, _, height, width = image.shape
    
    sw = math.ceil(width*scale)
    sh = math.ceil(height*scale)
    
    # =============== bilinear =============== #
    bilinear_img = F.interpolate(image, size=(sh, sw), mode='bilinear', align_corners=False)
    bilinear_probs = net(bilinear_img)
    
    # ============== inter_area ============== #
    lcm_height = int(height * sh)
    lcm_width = int(width * sw)
    upsampled_tensor = F.interpolate(image, size=(lcm_height, lcm_width), mode='nearest')
    kernel_size = (height, width)
    stride_size = (height, width)
    area_img = F.avg_pool2d(upsampled_tensor, kernel_size=kernel_size, stride=stride_size)    
    area_probs = net(area_img)
    
    return bilinear_probs, area_probs

def mtcnn_attack(image, loss_fn=None, loss=0, min_face_size=20.0, idx=7, device='cuda'):
    if loss_fn:
        pnet_attack = PNet_attack(device=device)
        
        _, _, height, width = image.shape
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.709
        
        scales = []
        m = min_detection_size/min_face_size
        min_length *= m
        factor_count = 0
        
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1
            
        for i, s in enumerate(scales):
            if i >= idx:
                probs = run_first_stage(image, pnet_attack, scale=s)
                for prob in probs:
                    mask = prob[:, 1, :, :] > 0.6
                    gt_soft_prob = prob.clone().detach()
                    gt_soft_prob[:, 1, :, :][mask] = 0.4
                    gt_soft_prob[:, 0, :, :][mask] = 0.6
                    scale_loss = loss_fn(prob, gt_soft_prob)
                    
                    if i == 6:
                        weight = 6
                    elif i == 7:
                        weight = 8
                    elif i == 8:
                        weight = 10
                    elif i == 9:
                        weight = 20
                    loss += weight * (scale_loss)
    return loss