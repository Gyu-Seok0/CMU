import argparse
import wandb
import torch
import random
import gc
import zipfile
import os

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import datetime

import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio #.transforms as tat
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts,ReduceLROnPlateau

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder
import warnings

# utils
from utils.dataset import *
from utils.normalize import *
from utils.execute import *
from utils.model import *
from utils.prediction import *

def main(args):
    # wandb
    wandb.login(key="e0408f5d7b96be3d00be30b39eda0f1e259672ed")
    run = wandb.init(
        name = args.project, ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw3p2-ablations", ### Project should be created in your wandb account 
        config = args, ### Wandb Config for your run
        entity = "gyuseoklee"
    )

    # Labels
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

    # device
    warnings.filterwarnings('ignore')
    device = f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Data
    pwd = os.getcwd()
    train_data = AudioDataset(os.path.join(pwd, "hw3p2", "train-clean-360"), train = True) #train-clean-100
    val_data = AudioDataset(os.path.join(pwd,"hw3p2","dev-clean"), train = False)
    test_data = AudioDatasetTest(os.path.join(pwd,"hw3p2","test-clean"))

    # normalize
    noramlize_cmvn(train_data, val_data, test_data)

    # DataLoader
    gc.collect()
    train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                              shuffle = True, drop_last = False, collate_fn = train_data.collate_fn,
                              pin_memory = True) #TODO

    val_loader = DataLoader(val_data, batch_size = args.batch_size, 
                            shuffle = False, drop_last = False, collate_fn = val_data.collate_fn,
                            pin_memory = True)

    test_loader = DataLoader(test_data, batch_size = args.batch_size, 
                            shuffle = False, drop_last = False, collate_fn = test_data.collate_fn,
                            pin_memory = True)
    
    print("Batch size: ", args.batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    # model
    torch.cuda.empty_cache()
    #model = Network().to(device)
    model = New_Network().to(device)
    if args.pretrain is not None:
        dic = torch.load(args.pretrain)
        model.load_state_dict(dic["model_state_dict"])
        print("[Done] Load the Pretrained Model! ")
    else:
        print("[Fail] No pretrained Model ")

    # test
    # x,y,lx,ly = next(iter(train_loader))
    # out, out_length = model(x.to(device), lx.to(device))
    # print("here", out.shape, out_length.shape)

    # loss & optimizer & scheduler
    criterion = nn.CTCLoss()

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate) # What goes in here?
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 1e-4)
    print(f"[Optimizer] {args.optimizer}")

    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size = 30, gamma = 0.1) #TODO
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2, eta_min = 0.0001)
    elif args.scheduler == "reduce":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.05, patience = 5, min_lr = 0.0001)
    print(f"[scheduler] {args.scheduler}")

    # cuda
    torch.cuda.empty_cache()
    gc.collect()

    # train loop
    best_val_dist = float("inf") # if you're restarting from some checkpoint, use what you saw there.
    decoder = CTCBeamDecoder(
                            labels = LABELS,
                            beam_width = 2,
                            num_processes = 40,
                            log_probs_input = True
                            )

    for epoch in range(args.epochs):
        print(f"epoch: {epoch + 1}/{args.epochs}")

        # one training step
        train_loss = train_step(train_loader, model, optimizer, criterion, scheduler, device, epoch)

        # one validation step (if you want)
        val_loss, val_dist = evaluate(val_loader, model, decoder, LABELS, device, criterion)

        # Where you have your scheduler.step depends on the scheduler you use.
        if args.scheduler == "step":
            scheduler.step()
        
        if args.scheduler == "reduce":
            scheduler.step(val_loss)
        
        # Use the below code to save models
        if val_dist < best_val_dist:
            print("Saving model")
            model_path = f"{args.model_save}/{args.project}.pth"

            torch.save({'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'val_dist' : val_dist, 
                        'epoch' : epoch}, model_path)

            best_val_dist = val_dist
            wandb.save(model_path)
        
        # You may want to log some hyperparameters and results on wandb
        curr_lr = float(optimizer.param_groups[0]['lr'])
        result = {"train_loss":train_loss, 'validation_loss': val_loss,
                   "val_dist": val_dist, "learning_Rate": curr_lr}
        print(result)
        wandb.log(result)

    run.finish()

    # load the best mnodel
    dic = torch.load(model_path)
    model.load_state_dict(dic["model_state_dict"])

    # Decoder
    decoder_test =  CTCBeamDecoder(
                            labels = LABELS,
                            beam_width = 20,
                            num_processes = 40,
                            log_probs_input = True
                            )
    
    # Predict & Save
    predictions = predict(test_loader, model, decoder_test, LABELS, device)
    path = os.path.join(os.getcwd(),'hw3p2/test-clean/transcript/random_submission.csv')
    df = pd.read_csv(path)
    df.label = predictions
    df.to_csv(f'{args.project}.csv', index = False)

    # sumbit
    # command = f"kaggle competitions submit -c 11-785-f22-hw3p2 -f {args.project}.csv -m Message"
    # os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type = str, default = None)
    parser.add_argument("--model_save", type = str, default = "/home/gyuseok/CMU/IDL/HW3/weight", help = "path of saving the model parameter")
    parser.add_argument("--pretrain", type = str, default = "/home/gyuseok/CMU/IDL/HW3/weight/New_last.pth", help = "path of the pretrained model's parameter")
     
    parser.add_argument("--epochs", type = int, default = 50)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--learning_rate", type = float, default = 0.002)

    parser.add_argument("--gpu_number", type = int, default = 0)
    parser.add_argument("--optimizer", type = str, default = "AdamW", help = "AdamW or SGD")
    parser.add_argument("--scheduler", type = str, default = "reduce", help = "step or cosine or reduce")

    args = parser.parse_args()

    print(args)
    main(args)

    print("[Finish]")
    print(args)