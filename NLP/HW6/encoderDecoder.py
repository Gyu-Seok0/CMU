# -*- coding: utf-8 -*-
"""
# Neural Machine Translation using Seq2Seq Models

In HW06, we will be training our own machine translation system.
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchmetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""Object to store our language data. In this object, we assign a unique integer ID to every word in the vocab. We also assign new `<SOS>`, `<EOS>` and `<OOV>` tokens for start-of-sentence, end-of-sentence and out-of-vocab tokens respectively"""

SOS_token = 0
EOS_token = 1
OOV_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 3  # Count SOS, EOS and OOV

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

"""Standard text pre-processing and string normalization. We convert unicode to ASCII and remove all punctuations"""

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove punctuations
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

def readLangs(lang1, lang2, path_to_corpus):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path_to_corpus, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('|||')] for l in lines]
        
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

"""We only consider sentences that are max 10 words in length. Think about why we might need to restrict the maximum number of words in a sentence."""

MAX_LENGTH = 10


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

"""Read in the data. Ensure that the path to the data files are correct"""

def prepareData(lang1, lang2, path_to_corpus):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path_to_corpus)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


"""Design your encoder model here"""

# Set your hidden size here
HIDDEN_SIZE = 256

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # Assign hidden_size
        self.hidden_size = hidden_size ##YOUR CODE HERE

        # Create nn.Embedding layer with (input_size, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size) ##YOUR CODE HERE

        # Create nn.GRU layer with (hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 1)##YOUR CODE HERE

    def forward(self, input, hidden):
        # Run the input through the embedding layer
        embedded = self.embedding(input) ##YOUR CODE HERE: torch.Size([1, 256])

        # Reshape with (1, 1, -1)
        embedded = embedded.reshape(1, 1, -1) ##YOUR CODE HERE: torch.Size([1, 1, 256])


        # Run both the embedded and hidden through GRU
        output, hidden = self.gru(embedded, hidden) ##YOUR CODE HERE: (1, 1, HIDDEN_SIZE), (1, 1, HIDDEN_SIZE)

        # Return both output and hidden
        return output, hidden

    def initHidden(self):
        # Create a torch tensor of zeros of shape (1, 1, HIDDEN_SIZE)
        return torch.zeros(1,1,self.hidden_size).to(device) ##YOUR CODE HERE

dummy_in = torch.randint(1, 10, (1,), device=device)
dummy_encoder = EncoderRNN(100, HIDDEN_SIZE).to(device)
assert dummy_encoder.initHidden().shape == (1, 1, HIDDEN_SIZE)
dummy_out, dummy_hid = dummy_encoder.forward(dummy_in, dummy_encoder.initHidden())
assert dummy_out.shape == (1, 1, HIDDEN_SIZE)
assert dummy_hid.shape == (1, 1, HIDDEN_SIZE)

"""Design your decoder model here"""
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        # Assign hidden_size
        self.hidden_size = hidden_size ##YOUR CODE HERE

        # Create nn.Embedding layer with (output_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size) ##YOUR CODE HERE

        # Create nn.GRU layer with (hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 1) ##YOUR CODE HERE

        # Create a nn.Linear layer with (hidden_size, output_size)
        self.out = nn.Linear(hidden_size, output_size) ##YOUR CODE HERE

        # Create a nn.LogSoftmax layer with (dim=1)
        self.softmax = nn.LogSoftmax(dim = 1) ##YOUR CODE HERE

    def forward(self, input, hidden):
        # Run the input through the embedding layer
        input = self.embedding(input) ##YOUR CODE HERE

        # Reshape the input with (1, 1, -1)
        input = input.reshape(1,1,-1) ##YOUR CODE HERE

        # Use relu activation 
        input = F.relu(input)

        # Run both the input and hidden through GRU
        output, hidden = self.gru(input, hidden) ##YOUR CODE HERE
        
        # Reshape the output with (1, -1)
        output = output.reshape(1,-1) ##YOUR CODE HERE

        # Run the output through the linear layer
        output = self.out(output) ##YOUR CODE HERE

        # Get softmax scores
        output = self.softmax(output) ##YOUR CODE HERE

        # Return both output and hidden
        return output, hidden

    def initHidden(self):
        # Create a torch tensor of zeros of shape (1, 1, HIDDEN_SIZE)
        return torch.zeros(1,1,self.hidden_size).to(device) ##YOUR CODE HERE

dummy_in = torch.randint(1, 10, (1,), device=device)
dummy_decoder = DecoderRNN(HIDDEN_SIZE, 100).to(device)
assert dummy_decoder.initHidden().shape == (1, 1, HIDDEN_SIZE)
dummy_out, dummy_hid = dummy_decoder.forward(dummy_in, dummy_decoder.initHidden())
assert dummy_out.shape == (1, 100)
assert dummy_hid.shape == (1, 1, HIDDEN_SIZE)

"""Helper functions to convert the sentences into vector. Given a sentence, we convert it into a vector of word IDs"""

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, OOV_token) for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

"""Implement the main training loop here. This function takes in one input tensor and one output tensor and does a forward pass, backward pass and weight updates. """

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, only_forward_pass=False):
    # Initialize the hidden layer for encoder
    encoder_hidden = encoder.initHidden() ##YOUR CODE HERE

    # Reset gradients for both encoder_optimizer and decoder_optimizer
    encoder_optimizer.zero_grad() ##YOUR CODE HERE
    decoder_optimizer.zero_grad() ##YOUR CODE HERE

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        # Run each input word through the encoder
        # You can access the current input as input_tensor[ei]
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden) ##YOUR CODE HERE)

    # First input to the decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Assign the last encoder_hidden to decoder_hidden
    decoder_hidden = encoder_hidden ##YOUR CODE HERE

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # Run decoder by providing decoder_input and decoder_hidden as input
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) ##YOUR CODE HERE)

            # Calculate loss
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # Run decoder by providing decoder_input and decoder_hidden as input
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) ##YOUR CODE HERE)

            # Take the top output of current timestep of decoder. This will be input to next timestep
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    if not only_forward_pass:
        # Backprop by calling backward() function on loss
        loss.backward() ##YOUR CODE HERE

        # Update weights using step() on both encoder_optimizer and decoder_optimizer
        encoder_optimizer.step()##YOUR CODE HERE
        decoder_optimizer.step() ##YOUR CODE HERE

    return loss.item() / target_length
    
def trainIters(encoder, decoder, n_iters, input_lang, output_lang, pairs, learning_rate=0.01):
    # Initialize SGD optimizers for both encoder and decoder
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # Convert words to tensors
    all_training_pairs = [tensorsFromPair(input_lang, output_lang, pair) for pair in pairs]

    # Create training and valid datasets
    random.shuffle(all_training_pairs)
    valid_pairs = all_training_pairs[:500]
    training_pairs = all_training_pairs[500:]
    
    # We will be using NLLLoss as criterion
    criterion = nn.NLLLoss()

    epoch_train_losses = []
    epoch_valid_losses = []

    # In each epoch, we go through all training examples
    for iter in range(1, n_iters + 1):

        # Train
        train_loss = 0.0
        for training_pair in training_pairs:
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                      decoder, encoder_optimizer, decoder_optimizer, criterion)
            train_loss += loss

        # Validate
        valid_loss = 0.0
        for val_pair in valid_pairs:
            input_tensor = val_pair[0]
            output_tensor = val_pair[1]
            loss = train(input_tensor, target_tensor, encoder,
                      decoder, encoder_optimizer, decoder_optimizer, criterion, only_forward_pass=True)
            valid_loss += loss

        avg_train_loss = train_loss / len(training_pairs)
        avg_valid_loss = valid_loss / len(valid_pairs)

        print("Epoch: {}/{}. Avg Train Loss: {}. Avg Valid Loss: {}".format(iter, n_iters, avg_train_loss, avg_valid_loss))

        epoch_train_losses.append(avg_train_loss)
        epoch_valid_losses.append(avg_valid_loss)

    return epoch_train_losses, epoch_valid_losses


"""Function to perform inference. Given an input sentence, this function returns the translated sentence"""

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    # Give your input path here
    HI_PATH = "/content/drive/MyDrive/hw06_data/eng_hin/ted-train.orig.eng-hin"
    ZH_PATH = "/content/drive/MyDrive/hw06_data/eng_zh/ted-train.orig.eng-zh"
    input_lang_train_en_hi, output_lang_train_hi, train_pairs_hi = prepareData('en', 'hi', HI_PATH)
    input_lang_train_en_zh, output_lang_train_zh, train_pairs_zh = prepareData('en', 'zh', ZH_PATH)
    print(random.choice(train_pairs_hi))
    print(random.choice(train_pairs_zh))

    """Train the model. What do you notice about the training and validation losses?"""

    # ENG - HIN
    encoder_eng_hi = EncoderRNN(input_lang_train_en_hi.n_words, HIDDEN_SIZE).to(device)
    decoder_eng_hi = DecoderRNN(HIDDEN_SIZE, output_lang_train_hi.n_words).to(device)

    avg_train_losses_hi, avg_valid_losses_hi = trainIters(encoder_eng_hi, decoder_eng_hi, 30, input_lang_train_en_hi, output_lang_train_hi, train_pairs_hi)

    # ENG - ZH
    encoder_eng_zh = EncoderRNN(input_lang_train_en_zh.n_words, HIDDEN_SIZE).to(device)
    decoder_eng_zh = DecoderRNN(HIDDEN_SIZE, output_lang_train_zh.n_words).to(device)

    avg_train_losses_zh, avg_valid_losses_zh = trainIters(encoder_eng_zh, decoder_eng_zh, 30, input_lang_train_en_zh, output_lang_train_zh, train_pairs_zh)

    """Calculate CHRF score"""

    hyp_hi = []
    ref_hi = []

    for sample in train_pairs_hi:
        x = sample[0]
        y_true = sample[1]
        y_pred_tokens = evaluate(encoder_eng_hi, decoder_eng_hi, x, input_lang_train_en_hi, output_lang_train_hi)

        if "<EOS>" in y_pred_tokens:
            y_pred_tokens.remove("<EOS>")
        y_pred = " ".join(y_pred_tokens)

        hyp_hi.append(y_pred)
        ref_hi.append([y_true])

    hyp_zh = []
    ref_zh = []

    for sample in train_pairs_zh:
        x = sample[0]
        y_true = sample[1]
        y_pred_tokens = evaluate(encoder_eng_zh, decoder_eng_zh, x, input_lang_train_en_zh, output_lang_train_zh)

        if "<EOS>" in y_pred_tokens:
            y_pred_tokens.remove("<EOS>")
        y_pred = " ".join(y_pred_tokens)

        hyp_zh.append(y_pred)
        ref_zh.append([y_true])

    metric = torchmetrics.CHRFScore()
    print("Hindi CHRF Score", metric(hyp_hi, ref_hi))
    print("Chinese CHRF Score", metric(hyp_zh, ref_zh))

