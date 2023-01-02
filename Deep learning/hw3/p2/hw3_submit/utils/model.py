from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def DS_block(in_channel, out_channel, stride):
    return nn.Sequential(
                         nn.Conv1d(in_channel, in_channel, kernel_size = 3,  stride = stride, groups = in_channel, padding = 2),
                         nn.BatchNorm1d(in_channel),
                         nn.GELU(), #nn.ReLU(),
        
                         nn.Conv1d(in_channel, out_channel, kernel_size = 1, stride = 1),
                         nn.BatchNorm1d(out_channel),
                         nn.GELU() #nn.ReLU(),
                        )

def Res_Block(in_channel, out_channel):
    return nn.Sequential(
                         nn.Conv1d(in_channel, out_channel, kernel_size = 1, stride = 1),
                         nn.BatchNorm1d(out_channel),
                         nn.GELU(), #nn.ReLU(),
        
                         nn.Conv1d(out_channel, in_channel, kernel_size = 1, stride = 1),
                         nn.BatchNorm1d(in_channel),
                         nn.GELU(), #nn.ReLU(),
                         )

def make_DS_Res_Block(dims, strides):
    ds_dims = dims[:-1]
    rs_dims = dims[1:]

    ds_layer = []
    ds_dims = list(zip(ds_dims[:-1], ds_dims[1:]))
    for idx, (in_dim, out_dim) in enumerate(ds_dims):
        ds_layer += [DS_block(in_dim, out_dim, strides[idx])]

    rs_layer = []
    rs_dims = list(zip(rs_dims[:-1], rs_dims[1:]))
    for in_dim, out_dim in rs_dims:
        rs_layer += [Res_Block(in_dim, out_dim)]
    
    return ds_layer, rs_layer

class Network(nn.Module):

    def __init__(self, dims = None):

        super(Network, self).__init__()
        self.OUT_SIZE = 41 # 43

        # Adding some sort of embedding layer or feature extractor might help performance.
        if dims is None:
            #dims = [15, 1024, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 512]
            #dims = [15, 1024, 1024, 1024, 512, 512, 512, ]
            #dims = [15, 512, 512, 512, 256, 256, 256, ]
            self.dims = [15, 1024, 1024, 1024, 512, 512, 512]
            self.strides = [2,1,1,2,1,1]
        
        self.lx_scale = 2 ** Counter(self.strides)[2]
        
        ds_layer, rs_layer = make_DS_Res_Block(self.dims, self.strides)
        self.ds_layer = nn.ModuleList(ds_layer)
        self.rs_layer = nn.ModuleList(rs_layer)
 
        # TODO : look up the documentation. You might need to pass some additional parameters.
        #self.lstm1 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 2, batch_first = True, dropout = 0.25) 
        #self.lstm2 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 2, batch_first = True, dropout = 0.25)
        
        #self.lstm1 = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2, batch_first = True, dropout = 0.25) 
        #self.lstm2 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 2, batch_first = True, dropout = 0.25)
        
        self.lstm = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 3, batch_first = True, dropout = 0.25, bidirectional = True)
        
        
        self.classification = nn.Sequential(
            #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
            #nn.Linear(256, OUT_SIZE)
            nn.Linear(512, self.OUT_SIZE)
        )

        #TODO: Apply a log softmax here. Which dimension would apply it on ?

        self.logSoftmax = nn.LogSoftmax(dim = 2)
        
        
        
    def forward(self, x, lx):
        out = x.permute(0,2,1)
        for i in range(len(self.ds_layer)):
            out = self.ds_layer[i](out)
            out = self.rs_layer[i](out) + out
        out = out.permute(0,2,1)
        
        #out = pack_padded_sequence(out, lx.cpu().numpy(), batch_first=True, enforce_sorted=False)
        
#         out, (hn, cn) = self.lstm1(out)
#         out, (hn, cn) = self.lstm2(out)
        out, _ = self.lstm(out)
        
        #out, out_length = pad_packed_sequence(out, batch_first=True)
        
        out = self.classification(out)
        out = self.logSoftmax(out)
        
        return out, lx // self.lx_scale  #out_length
    

class New_Network(nn.Module):

    def __init__(self, dims = None):
        super(New_Network, self).__init__()
        self.OUT_SIZE = 41

        self.backbone = nn.Sequential(
            nn.Conv1d(15, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        self.lstm = nn.LSTM(input_size = 256, hidden_size = 712, num_layers = 5, batch_first = True, dropout = 0.15, bidirectional = True)

        
        self.classifier = nn.Sequential(
            #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
            #nn.Linear(256, OUT_SIZE)
            nn.Linear(1424, 1424),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(1424, self.OUT_SIZE)
        )

        self.logSoftmax = nn.LogSoftmax(dim = 2)


    
    def forward(self, x, lx):
       out = x.permute(0,2,1)
       out = self.backbone(out) 
       out = out.permute(0,2,1)
       
       out = pack_padded_sequence(out, lx.cpu().numpy(), batch_first=True, enforce_sorted=False)
       out, (hn,cn) = self.lstm(out)
       out, out_length = pad_packed_sequence(out, batch_first=True)

       out = self.classifier(out)
       out = self.logSoftmax(out)

       return out, out_length 