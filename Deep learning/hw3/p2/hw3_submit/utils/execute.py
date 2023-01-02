# Use debug = True to see debug outputs

from Levenshtein import distance as lev
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts

def calculate_levenshtein(h, y, lh, ly, decoder, labels, debug = False):
    if debug:
        print(f"\n----- IN LEVENSHTEIN -----\n")
        print("h", h.shape)
        print("y", y.shape)
        print("lh", lh.shape)
        print('ly', ly.shape)
        # Add any other debug statements as you may need
        # you may want to use debug in several places in this function
        
        
    # TODO: look at docs for CTC.decoder and find out what is returned here
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens = lh)

    batch_size = beam_results.shape[0] # TODO
    distance = 0 # Initialize the distance to be 0 initially

    # TODO: Loop through each element in the batch
    targets = []
    for i in range(batch_size):
        target = beam_results[i][0][:out_seq_len[i][0]]
        t_string = "".join([labels[idx] for idx in target])
        y_string = "".join([labels[idx] for idx in y[i][:ly[i]]])
        distance += lev(t_string, y_string)
    

    distance /= batch_size # TODO: Uncomment this, but think about why we are doing this

    return distance


def evaluate(data_loader, model, decoder, LABELS, device, criterion):
    
    dist = 0
    total_loss = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
    # TODO Fill this function out, if you're using it.
    model.eval()
    for x, y, lx, ly in data_loader:
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            out, out_lengths = model(x, lx)
            
        loss = criterion(out.permute(1,0,2), y, out_lengths, ly)
        total_loss += loss.item()
        
        dist += calculate_levenshtein(out, y, out_lengths, ly,
                                     decoder, LABELS)
        
    total_loss /= len(data_loader)
    dist /= len(data_loader)
    return total_loss, dist

def train_step(train_loader, model, optimizer, criterion, scheduler, device, epoch, scaler = None):
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    train_loss = 0
    
    model.train()
    for i, data in enumerate(train_loader):

        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)

        # TODO: Fill this with the help of your sanity check
        out, out_lengths = model(x, lx)

        loss = criterion(out.permute(1,0,2), y, out_lengths, ly)

        # HINT: Are you using mixed precision? 

        batch_bar.set_postfix(
            loss = f"{train_loss/ (i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )

        train_loss += loss.item()
        batch_bar.update()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if type(scheduler) == CosineAnnealingWarmRestarts:
            scheduler.step(epoch + i / len(train_loader))
    
    batch_bar.close()
    train_loss /= len(train_loader) # TODO

    return train_loss # And anything else you may wish to get out of this function