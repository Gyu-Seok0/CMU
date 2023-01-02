#TODO: Make predictions

# Follow the steps below:
# 1. Create a new object for CTCBeamDecoder with larger (why?) number of beams
# 2. Get prediction string by decoding the results of the beam decoder

import torch

def make_output(h, lh, decoder, LABELS):
  
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens = lh) #TODO: What parameters would the decode function take in?
    batch_size = beam_results.shape[0] #What is the batch size

    dist = 0
    preds = []
    for i in range(batch_size): # Loop through each element in the batch

        h_sliced = beam_results[i][0][:out_seq_len[i][0]] #TODO: Obtain the beam results
        h_string = "".join([LABELS[idx] for idx in h_sliced])#TODO: Convert the beam results to phonemes
        preds.append(h_string)
    
    return preds

#TODO:
# Write a function (predict) to generate predictions and submit the file to Kaggle
def predict(test_loader, model, decoder, LABELS, device):
    
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lx in test_loader:
            x = x.to(device)
            out, out_lengths = model(x, lx)
            preds += make_output(out, out_lengths, decoder, LABELS)

    return preds
