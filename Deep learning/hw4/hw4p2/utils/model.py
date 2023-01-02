import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import math
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_attention(attention): 
    # Function for plotting attention
    # You need to get a diagonal plot
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

def create_loss_mask(lens, DEVICE):
    mask = torch.arange(max(lens))
    mask = torch.tile(mask, (len(lens), 1))
    t = torch.tile( lens.reshape((len(lens), 1)) , (1, mask.shape[1]))
    mask = mask >= t # 주의하기.
    return mask.to(DEVICE)

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dropout = 0.25):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size, hidden_size,
                             num_layers = 1,
                             bidirectional = True,
                             batch_first = True) # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        pad_x, pad_x_length = pad_packed_sequence(x_packed, batch_first= True)
        
        # Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above        
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x, x_lens = self.trunc_reshape(pad_x, pad_x_length)

        # TODO: Pack Padded Sequence. What output(s) would you get?
        pack_x = pack_padded_sequence(x, x_lens.cpu().numpy(),
                                      batch_first = True,
                                      enforce_sorted = False)
        
        # TODO: Pass the sequence through bLSTM
        out, _ = self.blstm(pack_x)

        # What do you return?
        out, out_lens = pad_packed_sequence(out, batch_first = True)

        return out, out_lens

    def trunc_reshape(self, x, x_lens): 
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        batch_size, timestep, feature_dim = x.size()
        
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        if timestep % 2 == 1:
            x = x[:, :-1, :]
            timestep -= 1
        
        trun_x = x.contiguous().view(batch_size, 
                                     int(timestep/2),
                                     feature_dim * 2)
        
        # TODO: Reduce lengths by the same downsampling factor
        turn_x_lens = torch.div(x_lens, 2, rounding_mode= "trunc")
        
        return trun_x, turn_x_lens
    

class Listener(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Listener, self).__init__()

        # The first LSTM at the very bottom
        self.base_lstm = torch.nn.LSTM(input_size, encoder_hidden_size//2,
                                       num_layers = 3,
                                       bidirectional = True,
                                       dropout = 0.15,
                                       batch_first = True)#TODO: Fill this up

        self.pBLSTM1 = pBLSTM(encoder_hidden_size*2, encoder_hidden_size)
        self.pBLSTM2 = pBLSTM(encoder_hidden_size*4, encoder_hidden_size)
        self.LD = LockedDropout()
        '''
        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be? 
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # ...
            # ...
        )'''
         
    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader
        
        # TODO: Pack Padded Sequence
        pack_x = pack_padded_sequence(x, x_lens.cpu().numpy(),
                                      batch_first = True,
                                      enforce_sorted = False)
        # TODO: Pass it through the first LSTM layer (no truncation)
        out, _ = self.base_lstm(pack_x)
        
        # TODO: Pad Packed Sequence
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        out, out_lens = self.pBLSTM1(out)
        
        out = self.LD(out)
        
        pack_out = pack_padded_sequence(out, out_lens.cpu().numpy(),
                                        batch_first = True,
                                        enforce_sorted = False)
        
        encoder_outputs, encoder_lens = self.pBLSTM2(pack_out)

        encoder_outputs = self.LD(encoder_outputs) 
        
        
        

        # Remember the number of output(s) each function returns

        return encoder_outputs, encoder_lens
    

class Attention(torch.nn.Module):
    '''
    Attention is calculated using the key, value (from encoder hidden states) and query from decoder.
    Here are different ways to compute attention and context:

    After obtaining the raw weights, compute and return attention weights and context as follows.:

    masked_raw_weights  = mask(raw_weights) # mask out padded elements with big negative number (e.g. -1e9 or -inf in FP16)
    attention           = softmax(masked_raw_weights)
    context             = bmm(attention, value)
    
    At the end, you can pass context through a linear layer too.

    '''
    
    def __init__(self, encoder_hidden_size, decoder_output_size, projection_size, DEVICE):
        super(Attention, self).__init__()

        self.key_projection     = nn.Linear(encoder_hidden_size*2, projection_size) # TODO: Define an nn.Linear layer which projects the encoder_hidden_state to keys
        self.value_projection   = nn.Linear(encoder_hidden_size*2, projection_size) # TODO: Define an nn.Linear layer which projects the encoder_hidden_state to value
        self.query_projection   = nn.Linear(decoder_output_size, projection_size) # TODO: Define an nn.Linear layer which projects the decoder_output_state to query
        
        # Optional : Define an nn.Linear layer which projects the context vector
        self.context_fc         = nn.Linear(projection_size, projection_size)

        self.softmax            =  nn.Softmax(dim = 1) # TODO: Define a softmax layer. Think about the dimension which you need to apply 
        # Tip: What is the shape of energy? And what are those?
        self.DEVICE = DEVICE
    # As you know, in the attention mechanism, the key, value and mask are calculated only once.
    # This function is used to calculate them and set them to self
    
    def make_mask(self, lens):
        mask = torch.arange(max(lens))
        mask = torch.tile(mask, (len(lens), 1)) # 0,1,2,3,4,5, ... max_len
        t = torch.tile( lens.reshape((len(lens), 1)) , (1, mask.shape[1])) # 44,44,44 ..
        return mask >= t # 주의하기.
        
    
    def set_key_value_mask(self, encoder_outputs, encoder_lens):
    
        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        self.key      = self.key_projection(encoder_outputs)   # TODO: Project encoder_outputs using key_projection to get keys
        self.value    = self.value_projection(encoder_outputs) # TODO: Project encoder_outputs using value_projection to get values

        # encoder_max_seq_len is of shape (batch_size, ) which consists of the lengths encoder output sequences in that batch
        # The raw_weights are of shape (batch_size, timesteps)
        self.raw_weights = torch.randn(batch_size, encoder_max_seq_len)
        
        # TODO: To remove the influence of padding in the raw_weights, we want to create a boolean mask of shape (batch_size, timesteps)
        # The mask is False for all indicies before padding begins, True for all indices after.
        self.padding_mask     =  self.make_mask(encoder_lens).to(self.DEVICE) # TODO: You want to use a comparison between encoder_max_seq_len and encoder_lens to create this mask. 
        # (Hint: Broadcasting gives you a one liner)
        
    def forward(self, decoder_output_embedding):
        # key   : (batch_size, timesteps, projection_size)
        # value : (batch_size, timesteps, projection_size)
        # query : (batch_size, projection_size)

        self.query              = self.query_projection(decoder_output_embedding) # TODO: Project the query using query_projection

        # Hint: Take a look at torch.bmm for the products below 
        self.length             = 1/math.sqrt(self.query.shape[-1])
        self.raw_weights        = self.length * torch.bmm(self.query.unsqueeze(1), self.key.transpose(1,2)).squeeze(1) # TODO: Calculate raw_weights which is the product of query and key, and is of shape (batch_size, timesteps)
        self.masked_raw_weights      = self.raw_weights.masked_fill(self.padding_mask, -np.inf) # TODO: Mask the raw_weights with self.padding_mask. 
        # Take a look at pytorch's masked_fill_ function (You want the fill value to be a big negative number for the softmax to make it close to 0)

        attention_weights  = self.softmax(self.masked_raw_weights) # TODO: Calculate the attention weights, which is the softmax of raw_weights
        context            = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1) # TODO: Calculate the context - it is a product between attention_weights and value

        # Hint: You might need to use squeeze/unsqueeze to make sure that your operations work with bmm
        context = self.context_fc(context)
        
        return context, attention_weights # Return the context, attention_weights


class Speller(torch.nn.Module):

    def __init__(self, embed_size, decoder_hidden_size, decoder_output_size, vocab_size, attention_module= None, DEVICE = None):
        super().__init__()

        self.vocab_size         = vocab_size

        self.embedding          = nn.Embedding(self.vocab_size, embed_size) # TODO: Initialize the Embedding Layer (Use the nn.Embedding Layer from torch), make sure you set the correct padding_idx  

        self.lstm_cells         = torch.nn.Sequential(
                                # Create Two LSTM Cells as per LAS Architecture
                                # What should the input_size of the first LSTM Cell? 
                                # Hint: It takes in a combination of the character embedding and context from attention
                                    nn.LSTMCell(embed_size + attention_module.value_projection.weight.shape[0], decoder_hidden_size),
                                    nn.LSTMCell(decoder_hidden_size, decoder_output_size)
                                )
    
                                # We are using LSTMCells because process individual time steps inputs and not the whole sequence.
                                # Think why we need this in terms of the query

        self.char_prob          = nn.Linear(decoder_output_size + attention_module.value_projection.weight.shape[0],
                                            vocab_size) # TODO: Initialize the classification layer to generate your probability distribution over all characters

        self.char_prob.weight   = self.embedding.weight # Weight tying

        self.attention          = attention_module
        
        self.DEVICE = DEVICE
        
        self.training = False


    
    def forward(self, encoder_outputs, encoder_lens, y = None, tf_rate = 1): 

        '''
        Args: 
            embedding: Attention embeddings 
            hidden_list: List of Hidden States for the LSTM Cells
        ''' 

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape
        
        if y is not None:
            self.training = True

        if self.training:
            timesteps     = y.shape[1] # The number of timesteps is the sequence of length of your transcript during training
            label_embed   = self.embedding(y) # Embeddings of the transcript, when we want to use teacher forcing
        else:
            timesteps     = 600 # 600 is a design choice that we recommend, however you are free to experiment.
        

        # INITS
        predictions     = []

        # Initialize the first character input to your decoder, SOS
        char            = torch.full((batch_size,), fill_value=SOS_TOKEN, dtype= torch.long).to(self.DEVICE) 

        # Initialize a list to keep track of LSTM Cell Hidden and Cell Memory States, to None
        hidden_states   = [None]*len(self.lstm_cells) 

        attention_plot          = []
        context                 = None # TODO: Initialize context (You have a few choices, refer to the writeup )
        attention_weights       = torch.zeros(batch_size, encoder_max_seq_len) # Attention Weights are zero if not using Attend Module

        # Set Attention Key, Value, Padding Mask just once
        if self.attention != None:
            self.attention.set_key_value_mask(encoder_outputs, encoder_lens)
            context = torch.mean(self.attention.value, dim = 1)


        for t in range(timesteps):
            
            char_embed = self.embedding(char) #TODO: Generate the embedding for the character at timestep t

            if self.training and t > 0:
                # TODO: We want to decide which embedding to use as input for the decoder during training
                # We can use the embedding of the transcript character or the embedding of decoded/predicted character, from the previous timestep 
                # Using the embedding of the transcript character is teacher forcing, it is very important for faster convergence
                # Use a comparison between a random probability and your teacher forcing rate, to decide which embedding to use
                if torch.rand(1).item() < tf_rate:
                    char_embed = label_embed[:, t, :] # TODO
      
            decoder_input_embedding = torch.cat([context, char_embed], dim = 1)# TODO: What do we want to concatenate as input to the decoder? (Use torch.cat)
            
            # Loop over your lstm cells
            # Each lstm cell takes in an embedding 
            for i in range(len(self.lstm_cells)):
                # An LSTM Cell returns (h,c) -> h = hidden state, c = cell memory state
                # Using 2 LSTM Cells is akin to a 2 layer LSTM looped through t timesteps 
                # The second LSTM Cell takes in the output hidden state of the first LSTM Cell (from the current timestep) as Input, along with the hidden and cell states of the cell from the previous timestep
                hidden_states[i] = self.lstm_cells[i](decoder_input_embedding, hidden_states[i]) 
                decoder_input_embedding = hidden_states[i][0]

            # The output embedding from the decoder is the hidden state of the last LSTM Cell
            decoder_output_embedding = hidden_states[-1][0]

            # We compute attention from the output of the last LSTM Cell
            if self.attention != None:
                context, attention_weights = self.attention(decoder_output_embedding) # The returned query is the projected query

            attention_plot.append(attention_weights[0].detach().cpu())

            output_embedding     = torch.cat([context, decoder_output_embedding], dim = 1) # TODO: Concatenate the projected query with context for the output embedding
            # Hint: How can you get the projected query from attention
            # If you are not using attention, what will you use instead of query?

            char_prob            = self.char_prob(output_embedding)
            
            # Append the character probability distribution to the list of predictions 
            predictions.append(char_prob)

            char = torch.argmax(char_prob, dim = 1)# TODO: Get the predicted character for the next timestep from the probability distribution 
            # (Hint: Use Greedy Decoding for starters)

        attention_plot  = torch.stack(attention_plot, dim = 0) # TODO: Stack list of attetion_plots 
        predictions     = torch.stack(predictions, dim = 2) # TODO: Stack list of predictions 

        return predictions, attention_plot


class LAS(torch.nn.Module):
    def __init__(self, input_size, encoder_hidden_size, 
                 vocab_size, embed_size,
                 decoder_hidden_size, decoder_output_size,
                 projection_size, DEVICE):
        
        super(LAS, self).__init__()

        self.encoder        = Listener(input_size, encoder_hidden_size) # TODO: Initialize Encoder
        attention_module    = Attention(encoder_hidden_size, decoder_output_size, projection_size, DEVICE)# TODO: Initialize Attention
        self.decoder        = Speller(embed_size,
                                      decoder_hidden_size,
                                      decoder_output_size,
                                      vocab_size,
                                      attention_module,
                                      DEVICE) #TODO: Initialize Decoder, make sure you pass the attention module 
    
        self.initalize_weights()

    def initalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -1, 1)
        print("[Done] Weight Initalization! ")
    
    
    def forward(self, x, x_lens, y = None, tf_rate = 1):

        encoder_outputs, encoder_lens = self.encoder(x, x_lens) # from Listener
        predictions, attention_plot = self.decoder(encoder_outputs, encoder_lens, y, tf_rate)
        
        return predictions, attention_plot