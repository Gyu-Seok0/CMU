
import torch
import torch.nn as nn
import numpy as np


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        dims = [self.input_size] + [64] #[1024]*1 + [1024]*1
        in_out_dims = list(zip(dims[:-1], dims[1:]))
        for i in range(len(in_out_dims)):
           in_dim, out_dim = in_out_dims[i]
           layers += self.make_layer(in_dim,out_dim)

        # classfier 
        layers += [nn.Linear(out_dim, self.output_size)]
        self.layers = nn.Sequential(*layers)
        self.initalize_weights()
        
    def make_layer(self, in_dim, out_dim):
        return [nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(np.random.uniform(0.1,0.6))]
        
    def forward(self,x):
        return self.layers(x)
    
    def initalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight Initalization")


class Simple_MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        layers = self.make_layer(input_size, output_size)
       
        # classfier 
        self.layers = nn.Sequential(*layers)
        self.initalize_weights()
        
    def make_layer(self, in_dim, out_dim):
        return [nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU()]
        
    def forward(self,x):
        return self.layers(x)
    
    def initalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        print("[Done] Weight Initalization")

class LF_Model(torch.nn.Module):
    def __init__(self,input_size, output_size, models):
        super().__init__()
        self.smodel = models[0]
        self.dmodel = models[1]
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        outputs = [self.smodel(x[0]), self.dmodel(x[1])]
        outputs = torch.cat(outputs, dim = 1)
        return self.fc(outputs)

class DF_Model(torch.nn.Module):
    def __init__(self,input_size, output_size, models):
        super().__init__()
        self.smodel = models[0]
        self.dmodel = models[1]
        self.cmodel = models[2]
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        outputs = [self.smodel(x[0]), self.dmodel(x[1]), self.cmodel(x[2])]
        outputs = torch.cat(outputs, dim = 1)
        return self.fc(outputs)




