import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import numpy as np
import torch

# VOCAB
VOCAB = ['<sos>',   
        'A',   'B',    'C',    'D',    
        'E',   'F',    'G',    'H',    
        'I',   'J',    'K',    'L',       
        'M',   'N',    'O',    'P',    
        'Q',   'R',    'S',    'T', 
        'U',   'V',    'W',    'X', 
        'Y',   'Z',    "'",    ' ', 
        '<eos>']
        
VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, origin_path, train = True):
        
        # test
        self.train = train
        
        # preprocess
        self.mfcc_dir = os.path.join(origin_path, "mfcc")
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        
        self.transcript_dir = os.path.join(origin_path, "transcript", "raw")
        self.transcript_files = sorted(os.listdir(self.transcript_dir))
        
        self.VOCAB_MAP = VOCAB_MAP
        
        # length
        self.length = len(self.mfcc_files)
        
        # save
        self.mfccs = []
        self.transcripts = []
        
        for i in range(self.length):
            mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[i])           
            mfcc = np.load(mfcc_path)
            self.mfccs.append(mfcc)

            tran_path = os.path.join(self.transcript_dir, self.transcript_files[i])
            tran = np.load(tran_path)
            tran = np.vectorize(self.VOCAB_MAP.get)(tran)
            self.transcripts.append(tran)
        
    def __len__(self):
        return self.length
        
    
    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]
        mfcc = torch.FloatTensor(mfcc)
    
        tran = self.transcripts[ind]
        tran = torch.LongTensor(tran)
        return mfcc, tran
        
    def collate_fn(self, batch):
        batch_mfcc = []
        batch_tran = []
        lengths_mfcc = []
        lengths_tran = []
        
        for x,y in batch:
            batch_mfcc.append(x)
            lengths_mfcc.append(x.shape[0])

            batch_tran.append(y)
            lengths_tran.append(y.shape[0])
                
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first = True)
        batch_tran_pad = pad_sequence(batch_tran, batch_first = True)

        if self.train:
            T_mask = torchaudio.transforms.TimeMasking(time_mask_param = batch_mfcc_pad.shape[1]//8) # sequence
            F_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param = 4) #batch_mfcc_pad.shape[2]//2) # dim
            batch_mfcc_pad = F_mask(T_mask(batch_mfcc_pad.permute(0,2,1))).permute(0,2,1)

        return batch_mfcc_pad, batch_tran_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_tran)

class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self, origin_path):
        
        # preprocess
        self.mfcc_dir = os.path.join(origin_path, "mfcc")
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        
        self.VOCAB_MAP = VOCAB_MAP
        
        # length
        self.length = len(self.mfcc_files)
        
        # save
        self.mfccs = []
        self.transcripts = []
        
        for i in range(self.length):
            mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[i])           
            mfcc = np.load(mfcc_path)
            self.mfccs.append(mfcc)
  
    def __len__(self):
        return self.length
        
    
    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]
        mfcc = torch.FloatTensor(mfcc)
        return mfcc
        
    def collate_fn(self, batch):
        batch_mfcc = []
        batch_tran = []
        lengths_mfcc = []
        lengths_tran = []
        
        for x in batch:
            batch_mfcc.append(x)
            lengths_mfcc.append(x.shape[0])

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first = True)
        
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)