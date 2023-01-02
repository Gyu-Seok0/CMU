import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torchmetrics

# 1. Discourse Markers

# sentence is a string, discourse_list is a list of strings
def discourse_markers(sentence, discourse_list):
    in_features = # TODO: this should be the length of the discourse list 
    out_features = # TODO: define number of output features
    markers = #TODO: initialize as list of 0s of size input_features

    sentence = sentence.split()
    for word in sentence:
        for j, marker in enumerate(discourse_list):
            if word == marker:
                #TODO: store presence or absence of word or its count in sentence

    markers = # TODO: convert list to a tensor
    return markers



# 2. Cosine Similarity
cos = # define torch cosine similarity function, be sure to define dim as 0
def cosine_sim(sentence_1_embed, sentence_2_embed): # sentence_1 and sentence_2 are embeddings of shape (embedding_size)
    cos_sim = # TODO: find cosine similarity between two input sentences
    #print the shape of this tensor. Since this is just one item, we may need to reshape this tensor

    cos_sim = #TODO: reshape the tensor to size 1
    return cos_sim



# 3. Coreference 
def get_clusters(combined_sent):
    preds = #TODO: call predict function of coref_model and pass the sentence as specified in the documentation (list of a single concatenated sentence)
    clusters = #TODO: obtain the coref cluster indices (Note: these indices are character indices)
    return clusters

def num_corefs(prev_sentences, cur_sentence, next_sentences):
    combined_sent = ""
    len_prev_sentences = # you may use this variable to check if a point in a cluster belongs in the prev_sentences or not

    # TODO: combine the previous, current and next sentences and store as a single string 'combined_sent'
    
    clusters = #TODO: obtain clusers from the combined sentence

    count = 0

    #TODO: based on the clusters, count the number of coreferences that exist from previous sentences to current OR the next sentences
    # such a coref will exist if a cluster consists of at least one point from prev_sentences and another point from cur_sentence OR next_sentences

    count = #convert to tensor of size 1
    return count
    pass



# Pool Embeddings
def pool_word_embeddings(sent_embedding):
    #input dimensions: (num_words, embedding_size)
    #output dimensions: (embedding_size)

    sent_embedding = #TODO: we expect you to sum each word embedding to obtain a sentence embedding
    return sent_embedding



# Get sentence embeddings
def get_sentence_embeddings(docs):
    sentence_embeddings = []
    labels = []
    prev = None
    para_len = 0    # 4. a very important feature 

    for i, doc in enumerate(docs):
        for j,row in enumerate(doc):
            sentence = row[0] 
            label = #TODO: 'B' indicates 1, otherwise 0
            labels.append(label)

            if row[1] == 'B':
                # set para_len to 0
            else:
                # increment para_len
            
            para_len = # TODO: convert to tensor of shape 1

            markers = #TODO: obtain discourse markers using the current sentence and the discourse list you defined
            
            sent_tensor = word_to_index.tensorFromSentence(sentence)
            sent_embedding = embed_layer(sent_tensor) 

            sent_embedding = #TODO: pool word embeddings to obtain sentence embedding

            cos_sim = torch.FloatTensor([0])
            if prev!=None:
                cos_sim = #TODO: find cosine similarity between current and previous sentence
            prev = sent_embedding 

            #uncomment the lines below to calculate number of corefs
            # num_coref = torch.FloatTensor([0])
            # if j>0 and j<len(doc)-1:
            #     prev_sentences = [doc[j-1][0]] 
            #     next_sentences = [doc[j+1][0]]
            #     num_coref = num_corefs(prev_sentences, sentence, next_sentences)

            sent_embedding = #TODO: concatenate the sentence embedding with the features obtained from discourse markers, cosine similarity, coreference and para_length
            
            sentence_embeddings.append(sent_embedding)

    sentence_embeddings  = pad_sequence(sentence_embeddings, batch_first=True, padding_value=0) #pad the embeddings
    labels = torch.tensor(labels, dtype=torch.float)
    return sentence_embeddings, labels



# Model
class MLP(nn.Module):
    def __init__(self, input_size, output_size = 1): #output size is 1 for binary classification
        super(MLP,self).__init__()
        hidden_size = #TODO: define size of hidden layer
        self.linear1 = # TODO: define hidden linear layer
        self.relu1 = #TODO: define activation layer
        self.linear2 = # TODO: define output layer 
        self.sigmoid = # TODO: define a sigmoid layer
         
    def forward(self, input):
        # implement the forward function of the MLP
        pass



