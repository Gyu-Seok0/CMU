#!/bin/python

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
from fusion import *



def main(args):
    # 0)  split the data
    train_val_df = pd.read_csv(args.train_list_videos)
    train_df, val_df = train_test_split(train_val_df, test_size = 0.1, random_state = 256)
    test_df = pd.read_csv(args.test_list_videos)
    
    # 1) load the sound data
    sx_train, sy_train = load_train_val_data(train_df, args.sound_feat_dir)
    sx_val, sy_val = load_train_val_data(val_df, args.sound_feat_dir) 
    sx_test, video_ids = load_test_data(test_df, args.sound_feat_dir)

    print(f"sx_train: {sx_train.shape}, sy_train: {sy_train.shape}")
    print(f"sx_val: {sx_val.shape}, sy_train: {sy_val.shape}")
    print(f"sx_test: {sx_test.shape}")

    # normalize
    scaler = StandardScaler()
    scaler.fit(sx_train)
    sx_train = scaler.transform(sx_train)
    sx_val = scaler.transform(sx_val)
    sx_test = scaler.transform(sx_test)
    print("[Done Normalize for Speech dataset]\n")

    # Custom Dataset
    strain_set = CustomTrain(sx_train, sy_train)
    sval_set = CustomTrain(sx_val, sy_val)
    stest_set = CustomTest(sx_test)


    # 2) load the 3D data -> normalize
    dtrain_set = FeatureDataset(train_df, args.D_feat_dir)
    dt_mean, dt_var = calculate_mean_var(dtrain_set)
    dtrain_set.mean = dt_mean
    dtrain_set.var = dt_var

    dval_set = FeatureDataset(val_df, args.D_feat_dir, mean = dt_mean, var = dt_var)
    dtest_set = FeatureDataset(test_df, args.D_feat_dir, mean = dt_mean, var = dt_var)

    print(f"dtrain_set: {dtrain_set.__len__()}")
    print(f"dval_set: {dval_set.__len__()}")
    print(f"dtest_set: {dtest_set.__len__()}")
    print("[Done Normalize for 3D dataset]")


    # concat까지 끝냄. fusion모델에 대해 수행하면 될듯?
    print("[Fusion] ",args.fusion)
    if args.fusion == "early":
        train_set = ConcatData([strain_set, dtrain_set])
        val_set = ConcatData([sval_set, dval_set])
        test_set = ConcatData([stest_set, dtest_set], test_dataset = True)

        # DataLoader
        train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

        early_fusion(args, train_loader, val_loader, test_loader, video_ids)

    if args.fusion == "late":
        strain_loader = DataLoader(strain_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        sval_loader = DataLoader(sval_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        stest_loader = DataLoader(stest_set, batch_size = args.batch_size, shuffle = False, drop_last = False) 
        
        dtrain_loader = DataLoader(dtrain_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        dval_loader = DataLoader(dval_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        dtest_loader = DataLoader(dtest_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

        train_loader = [strain_loader, dtrain_loader]
        val_loader = [sval_loader, dval_loader]
        test_loader = [stest_loader, dtest_loader]

        late_fusion(args, train_loader, val_loader, test_loader, video_ids)

    if args.fusion == "double":
        ctrain_set = ConcatData([strain_set, dtrain_set])
        cval_set = ConcatData([sval_set, dval_set])
        ctest_set = ConcatData([stest_set, dtest_set], test_dataset = True)

        # DataLoader
        ctrain_loader = DataLoader(ctrain_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        cval_loader = DataLoader(cval_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        ctest_loader = DataLoader(ctest_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

        strain_loader = DataLoader(strain_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        sval_loader = DataLoader(sval_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        stest_loader = DataLoader(stest_set, batch_size = args.batch_size, shuffle = False, drop_last = False) 
        
        dtrain_loader = DataLoader(dtrain_set, batch_size = args.batch_size, shuffle = True, drop_last= True)
        dval_loader = DataLoader(dval_set, batch_size = args.batch_size, shuffle = False, drop_last = False)
        dtest_loader = DataLoader(dtest_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

        train_loader = [strain_loader, dtrain_loader,ctrain_loader]
        val_loader = [sval_loader, dval_loader,cval_loader]
        test_loader = [stest_loader, dtest_loader,ctest_loader]
        
        double_fusion(args, train_loader, val_loader, test_loader, video_ids)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sound_feat_dir", type = str, default = "./dataset/pf2")
    parser.add_argument("--D_feat_dir", type = str, default = "./dataset/cnn3d")

    parser.add_argument("--train_list_videos", type = str, default = "./dataset/train_val.csv")
    parser.add_argument("--test_list_videos", type = str, default = "./dataset/test_for_students.csv")
    parser.add_argument("--sfeat_dim", type = int, default = 1024)
    parser.add_argument("--dfeat_dim", type = int, default = 512)
    parser.add_argument("--batch_size", type = int, default = 750)
    parser.add_argument("--lr", type = float, default = 0.01, help = "learning rate")
    parser.add_argument("--weight_decay", type = float, default = 5e-4, help = "regularization term")
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--impatience", type = int, default = 15, help = "early stopping")

    parser.add_argument("--fusion", type = str, default = "double", help = "early, late, double")

    args = parser.parse_args()
    main(args)