import os
import pandas as pd
import numpy as np
import Levenshtein
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchsummaryX import summary
from torch.optim.lr_scheduler import StepLR 

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import gc
import wandb
from glob import glob


from utils.dataset import *
from utils.normalize import * 
from utils.model import *
from utils.execute import *

def save_predict(predictions, result_csv):
    sample_path = "/home/gyuseok/CMU/IDL/HW4/hw4p2/content/data/hw4p2/test-clean/transcript/random_submission.csv"
    df = pd.read_csv(sample_path)
    df.label = predictions
    df.to_csv(result_csv, index = False)


def main(args):

    wandb.login(key="e0408f5d7b96be3d00be30b39eda0f1e259672ed")
    run = wandb.init(
        name = args.project, ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "HW4P2", ### Project should be created in your wandb account 
        config = args, ### Wandb Config for your run
        entity = "gyuseoklee"
    )

    # Device
    DEVICE = f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu'
    print("Device: ", DEVICE)

    # Dataset
    path = "/home/gyuseok/CMU/IDL/HW4/hw4p2/content/data/hw4p2"
    train_path = os.path.join(path, 'train-clean-100')
    val_path = os.path.join(path, 'dev-clean')
    test_path = os.path.join(path, 'test-clean')

    train_data = AudioDataset(train_path, train = True)
    val_data = AudioDataset(val_path, train = False)
    test_data = AudioDatasetTest(test_path)

    # cepstral mean normalize
    noramlize_cmvn(train_data, val_data, test_data)

    # get me RAMMM!!!! 
    torch.cuda.empty_cache()
    gc.collect()
    
    # DataLoader
    train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                          shuffle = True, drop_last = False, collate_fn = train_data.collate_fn,
                           pin_memory = True) #TODO

    val_loader = DataLoader(val_data, batch_size = args.batch_size, 
                            shuffle = False, drop_last = False, collate_fn = val_data.collate_fn,
                            pin_memory = True)

    test_loader = DataLoader(test_data, batch_size = args.batch_size, 
                            shuffle = False, drop_last = False, collate_fn = test_data.collate_fn,
                            pin_memory = True)
    
    # Check
    print("Batch size: ", args.batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    # model
    input_size = 15 #15인데 cnn 추가함.
    encoder_hidden_size = 512 # encoder에서는 256으로 setting
    vocab_size = 30
    embed_size = 256
    decoder_hidden_size = 512
    decoder_output_size = 128
    projection_size = 128

    model = LAS(
                input_size, 
                encoder_hidden_size, 
                vocab_size, 
                embed_size,
                decoder_hidden_size,
                decoder_output_size,
                projection_size,
                DEVICE
                )
    model = model.to(DEVICE)
    print(model)

    ### Save your model architecture as a string with str(model) 
    model_arch = str(model)
    model_arch_path = f"{args.model_arch}/{args.project}.txt"
    arch_file = open(model_arch_path, "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()

    ### log it in your wandb run with wandb.save()
    wandb.save(model_arch_path)

    # optimizer & Loss
    optimizer   = torch.optim.Adam(model.parameters(), lr = args.learning_rate, amsgrad= True, weight_decay= 5e-6)
    criterion   = torch.nn.CrossEntropyLoss(reduction='none') # Why are we using reduction = 'none' ? 
    scaler      = torch.cuda.amp.GradScaler()
    scheduler   = StepLR(optimizer, step_size = 30, gamma = 0.1)#TODO

    # Optional: Create a custom class for a Teacher Force Schedule -> 나중에.

    # train & val
    best_lev_dist = float("inf")
    tf_rate = args.tf_rate
    best_running_loss = np.inf

    for epoch in range(args.epochs):
        
        print("\nEpoch: {}/{}".format(epoch+1, args.epochs))

        # Call train and validate 
        running_loss, running_lev_dist, running_perplexity, attention_plot = train(model, train_loader, criterion, optimizer, tf_rate, DEVICE, scaler)
        val_dist = validate(model, val_loader, DEVICE)
        
        # Print your metrics
        curr_lr = float(optimizer.param_groups[0]['lr'])
        result = {"train_loss":running_loss,
                  "train_dist" : running_lev_dist,
                  'train_perplexity': running_perplexity,
                  "val_dist": val_dist, 
                  "learning_Rate": curr_lr}

        print(result)
        
        # Plot Attention
        att_path = os.path.join("./attention_plot", args.project) + ".npy"
        np.save(att_path, attention_plot.cpu().numpy())
        #plot_attention(attention_plot)

        # Log metrics to Wandb
        wandb.log(result) 
        
        # Optional: Scheduler Step / Teacher Force Schedule Step
        if running_loss < best_running_loss:
            best_running_loss = running_loss
            tf_rate *= 0.95
                
        if val_dist <= best_lev_dist:
            best_lev_dist = val_dist

            # Save your model checkpoint here
            print("[START] Saving model")
            model_path = f"{args.model_save}/{args.project}.pth"
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_dist': val_dist, 
                        'epoch': epoch}, model_path)
            print("[Done] Saving model")
            
            # predict
            print("[START] Testing")
            predictions = predict(test_loader, model, DEVICE)
            result_csv = os.path.join(args.result_save, args.project) + ".csv"
            save_predict(predictions, result_csv)
            print("[Done] Testing")

        scheduler.step()

    # finish
    run.finish()

    # submit
    command = f"kaggle competitions submit -c 11-785-f22-hw4p2 -f {result_csv} -m Message"
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type = str, default = None)
    parser.add_argument("--model_save", type = str, default = "/home/gyuseok/CMU/IDL/HW4/hw4p2/weight", help = "path of saving the model parameter")
    parser.add_argument("--model_arch", type = str, default = "/home/gyuseok/CMU/IDL/HW4/hw4p2/arch", help = "path of saving the model architecture")
    parser.add_argument("--result_save", type = str, default = "/home/gyuseok/CMU/IDL/HW4/hw4p2/result", help = "path of prediction result")

    parser.add_argument("--pretrain", type = str, default = None, help = "path of the pretrained model's parameter")     
    parser.add_argument("--epochs", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--tf_rate", type = float, default = 0.5)
 

    parser.add_argument("--gpu_number", type = int, default = 0)
    #parser.add_argument("--optimizer", type = str, default = "AdamW", help = "AdamW or SGD")
    #parser.add_argument("--scheduler", type = str, default = "reduce", help = "step or cosine or reduce")

    args = parser.parse_args()

    print("[START] AI Modeling")
    print(args)

    main(args)

    print("[Finish] AI Modeling")
    print(args)