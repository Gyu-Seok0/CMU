
import argparse
import os
import pickle
from random import random, shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import datetime
import wandb
import sklearn.metrics
from sklearn.neural_network import MLPClassifier

from CustomSet import *
from model import *
from Modules import *

import time

def forward_backward(args, model, device, train_loader, val_loader, test_loader, video_ids):
    # optimizer & criterion & schduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                            lr = args.lr,
                            weight_decay = args.weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 2, eta_min = 0.0001)
    #scheduler = None

    # forward & backward
    best_acc = 0
    start = time.time()
    early_stop = 0
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch+1, args.epochs))

        train_loss = train_epoch(args, model, optimizer, criterion, train_loader, device, scheduler, epoch)
        end = time.time()

        accuracy, df = evaluate(args, model, val_loader, device)

        print("\tTrain Loss: {:.4f}".format(train_loss))
        print("\tTraining Time {:.4f}".format(end - start))

        print("\tValidation Accuracy: {:.2f}%".format(accuracy))

        early_stop += 1
        if accuracy > best_acc:
            best_acc = accuracy
            early_stop = 0
            if best_acc > 96.0:
                #1 
                make_CM_PRF(args, accuracy, df)

                #2 prediction file.
                pred_classes = test(args, model, test_loader, device)
                with open(f"{args.fusion}_{best_acc:.4f}.csv", "w") as f:
                    f.writelines("Id,Category\n")
                    for i, pred_class in enumerate(pred_classes):
                        f.writelines("%s,%d\n" % (video_ids[i], pred_class))
        if early_stop > args.impatience:
            print("[Done] Early Stopping")
            print(f"Best Acc : {best_acc:.4f}")
            break



def early_fusion(args, train_loader, val_loader, test_loader, video_ids):
    #cuda
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    #model
    model = MLP(input_size = args.sfeat_dim + args.dfeat_dim,
                output_size = 15).to(device)

    # forward & backward
    forward_backward(args, model, device, train_loader, val_loader, test_loader, video_ids)

   
def late_fusion(args, train_loader, val_loader, test_loader, video_ids):    
    #cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    #models 
    smodel = Simple_MLP(input_size = args.sfeat_dim,
                        output_size = 32).to(device)
    dmodel = Simple_MLP(input_size = args.dfeat_dim,
                        output_size = 32).to(device)
    models = [smodel, dmodel]
                
    model = LF_Model(input_size = 64, output_size = 15,
                     models = models).to(device)
            
    forward_backward(args, model, device, train_loader, val_loader, test_loader, video_ids)

   
def double_fusion(args, train_loader, val_loader, test_loader, video_ids):    
    #cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    #models 
    smodel = Simple_MLP(input_size = args.sfeat_dim,
                        output_size = 32).to(device)
    dmodel = Simple_MLP(input_size = args.dfeat_dim,
                        output_size = 32).to(device)
    cmodel = Simple_MLP(input_size = args.dfeat_dim + args.sfeat_dim,
                        output_size = 32).to(device)
    models = [smodel, dmodel, cmodel]
                
    model = DF_Model(input_size = 96, output_size = 15,
                     models = models).to(device)
            
    forward_backward(args, model, device, train_loader, val_loader, test_loader, video_ids)