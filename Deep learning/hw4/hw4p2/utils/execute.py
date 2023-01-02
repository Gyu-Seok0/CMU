import torch
import numpy as np
from tqdm import tqdm
import Levenshtein



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

def create_loss_mask(lens, DEVICE):
    mask = torch.arange(max(lens))
    mask = torch.tile(mask, (len(lens), 1))
    t = torch.tile( lens.reshape((len(lens), 1)) , (1, mask.shape[1]))
    mask = mask >= t # 주의하기.
    return mask.to(DEVICE)


# We have given you this utility function which takes a sequence of indices and converts them to a list of characters
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i) == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens

# To make your life more easier, we have given the Levenshtein distantce / Edit distance calculation code
def calc_edit_distance(predictions, y, ly, vocab= VOCAB, print_example= False):

    dist                = 0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size): 

        y_sliced    = indices_to_chars(y[batch_idx,0:ly[batch_idx]], vocab)
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)
        
        dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above abd uncomment below for toy dataset 
        # dist      += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example: 
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("Ground Truth : ", y_string)
        print("Prediction   : ", pred_string)
        
    dist/=batch_size
    return dist

def train(model, dataloader, criterion, optimizer, teacher_forcing_rate, DEVICE, scaler):

    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    running_perplexity  = 0.0
    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.cuda.amp.autocast():

            predictions, attention_plot = model(x, lx, y= y, tf_rate= teacher_forcing_rate)
            greedy_predictions   = torch.argmax(predictions, dim = 1) # TODO: How do you get the most likely character from each distribution in the batch?

            # Calculate Levenshtein Distance
            running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) # You can use print_example = True for one specific index i in your batches if you want


            # Predictions are of Shape (batch_size, timesteps, vocab_size). 
            # Transcripts are of shape (batch_size, timesteps) Which means that you have batch_size amount of batches with timestep number of tokens.
            # So in total, you have batch_size*timesteps amount of characters.
            # Similarly, in predictions, you have batch_size*timesteps amount of probability distributions.
            # How do you need to modify transcipts and predictions so that you can calculate the CrossEntropyLoss? Hint: Use Reshape/View and read the docs
            loss        =  criterion(predictions, y) # TODO: Cross Entropy Loss

            mask        = create_loss_mask(ly, DEVICE)# TODO: Create a boolean mask using the lengths of your transcript that remove the influence of padding indices (in transcripts) in the loss 
            masked_loss = loss.masked_fill(mask, 0) # Product between the mask and the loss, divided by the mask's sum. Hint: You may want to reshape the mask too 
            masked_loss = masked_loss.sum() / (mask.shape[0] * mask.shape[1] - mask.sum())
            perplexity  = torch.exp(masked_loss) # Perplexity is defined the exponential of the loss

            running_loss        += masked_loss.item()
            running_perplexity  += perplexity.item()
        
        # Backward on the masked loss
        scaler.scale(masked_loss).backward()

        # Optional: Use torch.nn.utils.clip_grad_norm to clip gradients to prevent them from exploding, if necessary
        # If using with mixed precision, unscale the Optimizer First before doing gradient clipping
        
        scaler.step(optimizer)
        scaler.update()
        

        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            dist="{:.04f}".format(running_lev_dist/(i+1)),
            perplexity="{:.04f}".format(running_perplexity/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_rate))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    running_lev_dist /= len(dataloader)

    batch_bar.close()

    return running_loss, running_lev_dist, running_perplexity, attention_plot

def validate(model, dataloader, DEVICE):

    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx, ly

        with torch.inference_mode():
            predictions, attentions = model(x, lx, y = None)

        # Greedy Decoding
        greedy_predictions   = torch.argmax(predictions, dim = 1) # TODO: How do you get the most likely character from each distribution in the batch?

        # Calculate Levenshtein Distance
        running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) # You can use print_example = True for one specific index i in your batches if you want

        batch_bar.set_postfix(
            dist="{:.04f}".format(running_lev_dist/(i+1)))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist #, running_loss, running_perplexity, 

def predict(test_loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lx in test_loader:
            x = x.to(device)
            predictions, _ = model(x, lx)
            greedy_predictions   = torch.argmax(predictions, dim = 1)
            for batch_idx in range(greedy_predictions.shape[0]): 
                pred_sliced = indices_to_chars(greedy_predictions[batch_idx], VOCAB)

                # Strings - When you are using characters from the AudioDataset
                pred_string = ''.join(pred_sliced)
                preds.append(pred_string)
            
            

    return preds