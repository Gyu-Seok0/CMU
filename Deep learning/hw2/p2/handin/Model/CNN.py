from torchvision.ops import StochasticDepth
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision #This library is used for image-based operations (Augmentations)
import torchvision.transforms as transforms 
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb
from torch import Tensor
from typing import List
import math
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class ConvNormAct(nn.Sequential):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: int,
                 norm = nn.BatchNorm2d,
                 act = nn.ReLU,
                 **kwargs
                ):
        super().__init__(
            nn.Conv2d(in_features,
                      out_features,
                      kernel_size = kernel_size,
                      padding = kernel_size // 2,
                      **kwargs),
            norm(out_features),
            act(),
        )
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features:int):
        super().__init__(
            ConvNormAct(
                in_features, out_features, kernel_size=7, stride = 2
            ),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )


class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 2,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x
    
class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )
        
class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )
        

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class Network(torch.nn.Module):
    """
    The Very Low early deadline architecture is a 4-layer CNN.

    The first Conv layer has 64 channels, kernel size 7, and stride 4.
    The next three have 128, 256, and 512 channels. Each have kernel size 3 and stride 2.
    
    Think about strided convolutions from the lecture, as convolutioin with stride= 1 and downsampling.
    For stride 1 convolution, what padding do you need for preserving the spatial resolution? 
    (Hint => padding = kernel_size // 2) - Why?)

    Each Conv layer is accompanied by a Batchnorm and ReLU layer.
    Finally, you want to average pool over the spatial dimensions to reduce them to 1 x 1. Use AdaptiveAvgPool2d.
    Then, remove (Flatten?) these trivial 1x1 dimensions away.
    Look through https://pytorch.org/docs/stable/nn.html 
    
    TODO: Fill out the model definition below! 

    Why does a very simple network have 4 convolutions?
    Input images are 224x224. Note that each of these convolutions downsample.
    Downsampling 2x effectively doubles the receptive field, increasing the spatial
    region each pixel extracts features from. Downsampling 32x is standard
    for most image models.

    Why does a very simple network have high channel sizes?
    Every time you downsample 2x, you do 4x less computation (at same channel size).
    To maintain the same level of computation, you 2x increase # of channels, which 
    increases computation by 4x. So, balances out to same computation.
    Another intuition is - as you downsample, you lose spatial information. We want
    to preserve some of it in the channel dimension.
    """

    def __init__(self, num_classes=7000, in_channels = None, stem_features = None, depths = None, widths = None):
        super().__init__()

        self.backbone = ConvNextEncoder(in_channels, stem_features, depths, widths)

        self.cls_layer = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(widths[-1]),
                                        nn.Linear(widths[-1], num_classes),
        )
        
        self.initalize_weights()
            
    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            s = feats.reshape((feats.shape[0],-1))
            return s
        else:
            return out
    
    def initalize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight initalization")

class NewNetwork(torch.nn.Module):
    def __init__(self, num_classes=7000, in_channels = None, stem_features = None, depths = None, widths = None):
        super().__init__()

        self.backbone = ConvNextEncoder(in_channels, stem_features, depths, widths)



        self.feat_layer = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(widths[-1]),
        )

        self.fc = nn.Sequential(
            nn.Linear(widths[-1], 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(np.random.uniform(0,0.5)),
            
            nn.Linear(512,num_classes)
        )


    def forward(self, x, return_feats=False):
    
        feats = self.backbone(x)
        feats = self.feat_layer(feats)

        if return_feats:
            return feats
        else:
            out = self.cls_layer(feats)
            return out

    def initalize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight initalization")

#https://github.com/deepinsight/insightface/blob/0b714727e02b4330008d95bdf98fff8d87372740/recognition/arcface_torch/losses.py#L63
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s = 16.0, margin = 0.15):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        #sin_theta = torch.sqrt(1.000001 - torch.pow(target_logit, 2))
        sin_theta = torch.sqrt(torch.clamp((1.000001 - torch.pow(target_logit, 2)),1e-9,1))
        
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits

class Arc_Network(torch.nn.Module):
    def __init__(self, num_classes=7000, in_channels = None, stem_features = None, depths = None, widths = None):
        super().__init__()

        self.backbone = ConvNextEncoder(in_channels, stem_features, depths, widths)
        self.cls_layer = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(widths[-1]),
                                        nn.Linear(widths[-1], 512),
        )
        self.fc = nn.Linear(512, num_classes)
        self.ArcFace = ArcFace()
        self.initalize_weights()


    def forward(self, x, labels = None, return_feats=False, mode = "Train"):
    
        x = self.backbone(x)
        feats = self.cls_layer(x)
        out = self.fc(feats) 

        if mode == "Train": #ArcFace적용
            out = self.ArcFace(out, labels)
        elif mode == "Test":
            pass
        
        if return_feats: # for verification
            s = feats.reshape((feats.shape[0],-1))
            return s
        else:
            return out

    def initalize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight initalization")




class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

class Conv_Sphere(torch.nn.Module):
    def __init__(self, num_classes=7000, in_channels = None, stem_features = None, depths = None, widths = None):
        super().__init__()

        self.backbone = ConvNextEncoder(in_channels, stem_features, depths, widths)

        self.feat_layer = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(1),
                                        nn.LayerNorm(widths[-1]),
                                        nn.Linear(widths[-1], 512)
        )

        self.fc = AngleLinear(512, num_classes)

        self.initalize_weights()
        
    def forward(self, x, return_feats=False):
    
        out1 = self.backbone(x)
        out2 = self.feat_layer(out1)

        if return_feats:
            return out2
        
        out3 = self.fc(out2) 
        return out3

    def initalize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight initalization")




class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 16, m = 0.15):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels = None):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels is None:
            return cos_th_m 
      

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs



class New_Arc_Network(torch.nn.Module):
    def __init__(self, num_classes=7000, in_channels = None, stem_features = None, depths = None, widths = None):
        super().__init__()

        self.backbone = ConvNextEncoder(in_channels, stem_features, depths, widths)
        self.feat_layer = nn.Sequential(
                                        nn.Linear(widths[-1]*3*3, 512),
                                        nn.BatchNorm1d(512)
        )
        self.initalize_weights()
        self.margin = ArcModule(in_features = 512, 
                                out_features = num_classes)



    def forward(self, x, labels = None, return_feats=False, mode = "Train"):
    
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feat_layer(features)
        features = F.normalize(features)

        if return_feats: # Batch x 512
            return features

        elif mode == "Train": # Batch x 7000
            return self.margin(features, labels)
        
        elif mode == "Test": # # Batch x 7000
            return self.margin(features)


    def initalize_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight initalization")

class ResNet_Arc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.margin = ArcModule(in_features = 512, 
                                out_features = 7000)

    def forward(self, x, labels = None, return_feats=False, mode = "Train"):
        features = self.backbone(x)
        features = F.normalize(features)

        if return_feats: # Batch x 512
            return features

        elif mode == "Train": # Batch x 7000
            return self.margin(features, labels)
        
        elif mode == "Test": # # Batch x 7000
            return self.margin(features)


class Efficient_Arc(torch.nn.Module):
    def __init__(self):
        super().__init__(model_name = "efficientnet_b0",weight_name = "EfficientNet_B0_Weights")

        weights = getattr(models, weight_name).DEFAULT
        self.transforms = weights.transforms()
        base_model = getattr(models, model_name)(weights=weights)
            
        self.model = create_feature_extractor(
                base_model, {"avgpool": 'feature'})


        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.margin = ArcModule(in_features = 512, 
                                out_features = 7000,
                                s = 32.0,
                                m = 0.25)

    def forward(self, x, labels = None, return_feats=False, mode = "Train"):
        features = self.backbone(x)
        features = F.normalize(features)

        if return_feats: # Batch x 512
            return features

        elif mode == "Train": # Batch x 7000
            return self.margin(features, labels)
        
        elif mode == "Test": # # Batch x 7000
            return self.margin(features)