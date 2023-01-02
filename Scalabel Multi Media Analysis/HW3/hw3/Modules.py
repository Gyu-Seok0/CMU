import argparse
import os
import pickle
from random import random, shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from torch.utils.data import DataLoader
from CustomSet import *
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import datetime
import wandb
import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_mean_var(train_dataset):
    X2, X = 0,0
    for idx in range(len(train_dataset)):
        target = train_dataset[idx][0]
        
        X2 += target**2
        X += target

    mean = X / len(train_dataset)
    var = X2/len(train_dataset) - mean**2
    return mean, var



# Train MLP classifier with labels
def load_train_val_data(df, sound_path):
    feat_list, label_list = [],[]
    for i in range(len(df)):
        video_id, label = df.iloc[i]
        feat_filepath = os.path.join(sound_path, video_id + ".csv")
        if os.path.exists(feat_filepath):
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
        else:
            feat_list.append(np.zeros(1024))
        label_list.append(int(label))

    return np.array(feat_list), np.array(label_list)

def load_test_data(df, sound_path):
    feat_list = []
    video_ids = []
    not_found_count = 0
    for i in range(len(df)):
        video_id = df.iloc[i,0]
        video_ids.append(video_id)
        
        feat_filepath = os.path.join(sound_path, video_id + ".csv")
        if not os.path.exists(feat_filepath):
            feat_list.append(np.zeros(1024))
            not_found_count += 1
        else:
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

    if not_found_count > 0:
        print(f'Could not find the features for {not_found_count} samples.')
    
    return np.array(feat_list), video_ids 





def train_epoch(args, model, optimizer, criterion, dataloader, device, scheduler, epoch):
    model.train()
    train_loss = 0.0 #Monitoring Loss
    iters = len(dataloader)
    loader_iters = None

    if args.fusion != "early":
        # dataloader:list
        loader_iters = [iter(each_dataloder) for each_dataloder in dataloader]
        iters = len(loader_iters[0]) 
    else:
        loader = iter(dataloader)

    for i in range(iters):
        
        if args.fusion != "early":
            mfccs,phonemes = [], None
            for loader in loader_iters:
                x_train, y_train = next(loader)
                mfccs.append(x_train.to(device))
                if phonemes is None:
                    phonemes = y_train.to(device)
        
        else:
            mfccs, phonemes = next(loader)
            
            ### Move Data to Device (Ideally GPU)
            mfccs = mfccs.to(device)
            phonemes = phonemes.to(device)
        
        ### Forward Propagation
        logits = model(mfccs)

        ### Loss Calculation
        loss = criterion(logits, phonemes)
        
        # ### Initialize Gradients
        optimizer.zero_grad()

        # ### Backward Propagation
        loss.backward()

        # ### Gradient Descent
        optimizer.step()  
        train_loss += loss.item()

        ###schdueler
        if scheduler is not None:
            scheduler.step(epoch + i / iters)
        
    train_loss /= len(dataloader)
    return train_loss

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def evaluate(args, model, dataloader, device):

    model.eval() # set model in evaluation mode
    enable_dropout(model)

    phone_true_list = []
    phone_pred_list = []

    iters = len(dataloader)
    loader_iters = None
    if args.fusion != "early":
        loader_iters = [iter(each_dataloder) for each_dataloder in dataloader]
        iters = len(loader_iters[0]) 
    else:
        loader = iter(dataloader)

    # evaluate
    for i in range(iters):
        if args.fusion != "early":
            frames,phonemes = [], None
            for loader in loader_iters:
                x_train, y_train = next(loader)
                frames.append(x_train.to(device))
                if phonemes is None:
                    phonemes = y_train.to(device)
        
        else:
            frames, phonemes = next(loader)
            
            ### Move Data to Device (Ideally GPU)
            frames = frames.to(device)
            phonemes = phonemes.to(device)

        with torch.inference_mode(): # makes sure that there are no gradients computed as we are not training the model now
            ### Forward Propagation
            logits = model(frames)

        ### Get Predictions
        predicted_phonemes = torch.argmax(logits, dim=1)
        
        ### Store Pred and True Labels
        phone_pred_list.extend(predicted_phonemes.cpu().tolist())
        phone_true_list.extend(phonemes.cpu().tolist())
        
        # Do you think we need loss.backward() and optimizer.step() here?
    
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    ### Calculate Accuracy
    accuracy = sklearn.metrics.accuracy_score(phone_pred_list, phone_true_list) 
    df = pd.DataFrame(confusion_matrix(phone_true_list, phone_pred_list))
    return accuracy*100, df

def test(args, model, dataloader, device):
    
    ### What you call for model to perform inference?
    model.eval()
    enable_dropout(model)
    
    ### List to store predicted phonemes of test data
    test_predictions = []

    iters = len(dataloader)
    loader_iters = None

    if args.fusion != "early":
        loader_iters = [iter(each_dataloder) for each_dataloder in dataloader]
        iters = len(loader_iters[0])
    else:
        loader = iter(dataloader)

    with torch.no_grad():
        for i in range(iters):
            if args.fusion != "early":
                frames = []
                for loader in loader_iters:
                    x_test = next(loader)
                    if type(x_test) == list:
                        x_test = x_test[0]
                    frames.append(x_test.to(device))  
            else:
                frames = next(loader)
                frames = frames.float().to(device)
        
            output = model(frames)

            ### Get most likely predicted phoneme with argmax
            predicted_phonemes = np.argmax(output.cpu().numpy(), axis = 1)
            ### How do you store predicted_phonemes with test_predictions? Hint, look at eval 
            test_predictions.append(predicted_phonemes)
            
    test_predictions = np.concatenate(test_predictions, axis = 0)
    return test_predictions


def make_CM_PRF(args, top1_accuracy, df, length = 750):
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(15):
        TP = df.iloc[i,i]
        FP = sum(df.iloc[:,i]) - TP
        TN = sum(df.iloc[i,:]) - TP
        FN = length - (TP + FP + TN)
        
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
    
    # Acc & Precision & Recall & F1-Score
    PRF_df = pd.DataFrame([TP_list, FP_list, TN_list, FN_list]).T
    PRF_df.columns = ["TP","FP","TN","FN"]
    PRF_df["Acurracy"] = PRF_df["TP"] / length
    PRF_df["Precision"] = PRF_df["TP"] / (PRF_df["TP"] + PRF_df["FP"] + 1e-10)
    PRF_df["Recall"] = PRF_df["TP"] / (PRF_df["TP"] + PRF_df["FN"] + 1e-10)
    PRF_df["F1-Score"] = 2*PRF_df["Precision"]*PRF_df["Recall"] / (PRF_df["Precision"] + PRF_df["Recall"] + 1e-10)

    #print("Top-1 Acc", top1_accuracy)

    #Save
    PRF_df.to_csv(f"PRF_df_{args.fusion}.csv")
    df.to_csv(f"Confusion_df_{args.fusion}.csv")


