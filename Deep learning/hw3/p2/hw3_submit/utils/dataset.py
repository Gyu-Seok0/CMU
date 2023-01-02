
import torch
import torchaudio
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ARPABET PHONEME MAPPING
# DO NOT CHANGE
# This overwrites the phonetics.py file.

CMUdict_ARPAbet = {
    "" : " ", # BLANK TOKEN
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@", 
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W", 
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R", 
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w", 
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y", 
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D", 
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O", 
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    }

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict
mapping = CMUdict_ARPAbet
LABELS = ARPAbet

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, origin_path, train = True): 
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''
        self.train = train
        # Load the directory and all files in them
        
        self.origin_path = origin_path #"/home/gyuseok/CMU/HW3/hw3p2/train-clean-360"
        
        self.mfcc_dir = os.path.join(self.origin_path,"mfcc")
        self.transcript_dir = os.path.join(self.origin_path, "transcript","raw")

        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO
        self.transcript_files = sorted(os.listdir(self.transcript_dir)) #TODO

        self.PHONEMES = PHONEMES

        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(self.mfcc_files)
        
        #TODO
        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        PHONEMES_dict = {letter:idx for idx,letter in enumerate(self.PHONEMES)}


        #TODO
        # CREATE AN ARRAY OF ALL FEATUERS AND LABELS
        # WHAT NORMALIZATION TECHNIQUE DID YOU USE IN HW1? CAN WE USE IT HERE?
        self.mfccs = []
        self.transcripts = []
        
        for i in range(self.length):
            mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[i])
            label_path = os.path.join(self.transcript_dir, self.transcript_files[i])
            
            mfcc = np.load(mfcc_path)
            label = np.load(label_path)[1:-1] 
            label = np.vectorize(PHONEMES_dict.get)(label) # transform into number
            
            self.mfccs.append(mfcc)
            self.transcripts.append(label)
        
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
       

    def __len__(self):
        
        '''
        TODO: What do we return here?
        '''
        return self.length

    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        mfcc = self.mfccs[ind] # TODO
        transcript = self.transcripts[ind] # TODO
        
        mfcc = torch.FloatTensor(mfcc)
        transcript = torch.LongTensor(transcript)
        return mfcc, transcript


    def collate_fn(self, batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        batch_mfcc = []
        batch_transcript = []
        lengths_mfcc = []
        lengths_transcript = []
        for x,y in batch:
            batch_mfcc.append(x) # TODO
            batch_transcript.append(y) # TODO
            lengths_mfcc.append(x.shape[0])
            lengths_transcript.append(y.shape[0])
            
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first = True) # TODO
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first = True) # TODO

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        if self.train:
            T_mask = torchaudio.transforms.TimeMasking(time_mask_param = batch_mfcc_pad.shape[1]//8) # sequence
            F_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param = 4) #batch_mfcc_pad.shape[2]//2) # dim
            batch_mfcc_pad = F_mask(T_mask(batch_mfcc_pad.permute(0,2,1))).permute(0,2,1)
        
        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


# Test Dataloader
#TODO
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self, origin_path): 
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        # Load the directory and all files in them
        self.origin_path = origin_path #"/home/gyuseok/CMU/HW3/hw3p2/train-clean-360"
        self.mfcc_dir = os.path.join(self.origin_path,"mfcc")
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)) #TODO


        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(self.mfcc_files)
   
        self.mfccs = []        
        for i in range(self.length):
            mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[i])            
            mfcc = np.load(mfcc_path)
            self.mfccs.append(mfcc)
            
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        mfcc = self.mfccs[ind] # TODO
        mfcc = torch.FloatTensor(mfcc)
        return mfcc


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        batch_mfcc = []
        lengths_mfcc = []

        for x in batch:
            batch_mfcc.append(x) # TODO
            lengths_mfcc.append(x.shape[0])

            
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first = True) # TODO

        # You may apply some transformation, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        
        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)
