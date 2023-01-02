import torch
import torch.nn as nn

from torchsummary import summary
import torchvision #This library is used for image-based operations (Augmentations)
import torchvision.transforms as transforms 
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb
from torch import Tensor
from typing import List

def train(model, dataloader, optimizer, criterion, scheduler, config, device, epoch, margin = None):
    
    model.train()

    # Progress Bar 
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
    
    num_correct = 0
    total_loss = 0
    
    iters = len(dataloader)
    for i, (images, labels) in enumerate(dataloader):
        
        optimizer.zero_grad() # Zero gradients

        images, labels = images.to(device), labels.to(device)
        
        #with torch.cuda.amp.autocast(): # This implements mixed precision. Thats it! 
        if config["model"] == ["ArcFace", "New_Arc_Network"]:
            outputs = model(images, labels, mode = "Train")
        else:
            outputs = model(images)

        if margin is not None:
            outputs = margin(outputs, labels)

        
        loss = criterion(outputs, labels)

        if config["criterion"] == "Sphereface":
            outputs = outputs[0]
        

        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
#         scaler.scale(loss).backward() # This is a replacement for loss.backward()
#         scaler.step(optimizer) # This is a replacement for optimizer.step()
#         scaler.update()

        loss.backward()
        optimizer.step()
        
        # TODO? Depending on your choice of scheduler,
        if scheduler is not None: 
            scheduler.step(epoch + i / iters)

        # You may want to call some schdulers inside the train function. What are these?
      
        batch_bar.update() # Update tqdm bar

    batch_bar.close() # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss

def validate(model, dataloader, criterion, config, device, margin = None):
  
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        
        # Move images to device
        images, labels = images.to(device), labels.to(device)
        
        # Get model outputs
        with torch.inference_mode():
            if config["model"] == ["ArcFace", "New_Arc_Network"]:
                outputs = model(images, mode = "Test")
            else:
                outputs = model(images)

            if margin is not None:
                outputs = margin(outputs)
            
            # loss
            loss = criterion(outputs, labels)

            if config["criterion"] == "Sphereface":
                outputs = outputs[0]

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()
        
    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss

def test(model, dataloader, device, config, margin = None):

  model.eval()
  batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
  test_results = []
  
  for i, (images) in enumerate(dataloader):
      # TODO: Finish predicting on the test set.
      images = images.to(device)

      with torch.inference_mode():
        if config["model"] == ["ArcFace", "New_Arc_Network"]:
            outputs = model(images, mode = "Test")
        else:
            outputs = model(images)

        if margin is not None:
            outputs = margin(outputs)
       


        if config["criterion"] == "Sphereface":
            outputs = outputs[0]

      outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
      test_results.extend(outputs)
      
      batch_bar.update()
      
  batch_bar.close()
  return test_results

def eval_verification(unknown_images, known_images, model, similarity, batch_size, mode='val', known_paths = None, device = None, config = None): 
    unknown_feats, known_feats = [], []

    batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    model.eval()

    # We load the images as batches for memory optimization and avoiding CUDA OOM errors
    for i in range(0, unknown_images.shape[0], batch_size):
        unknown_batch = unknown_images[i:i+batch_size] # Slice a given portion upto batch_size
        
        with torch.no_grad():
            if config["model"] == ["ArcFace", "New_Arc_Network"]:
                unknown_feat = model(unknown_batch.float().to(device),
                                     return_feats = config["return_feats"],
                                     mode = "Test") #Get features from model
            elif config["model"] == "ResNet":
                unknown_feat = model(unknown_batch.float())
            else:
                unknown_feat = model(unknown_batch.float().to(device),
                                     return_feats = config["return_feats"]) #Get features from model         

        unknown_feats.append(unknown_feat)
        batch_bar.update()
    
    batch_bar.close()
    
    batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
    
    for i in range(0, known_images.shape[0], batch_size):
        known_batch = known_images[i:i+batch_size] 
        with torch.no_grad():
            if config["model"] == ["ArcFace", "New_Arc_Network"]:
                known_feat = model(known_batch.float().to(device),
                                   return_feats = config["return_feats"],
                                   mode = "Test")
            elif config["model"] == "ResNet":
                known_feat = model(known_batch.float().to(device))
            else:
                known_feat = model(known_batch.float().to(device),
                                   return_feats = config["return_feats"])

     
          
        known_feats.append(known_feat)
        batch_bar.update()

    batch_bar.close()

    # Concatenate all the batches
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)

    #print("unknown_feats",unknown_feats.shape)
    #print("known_feats", known_feats.shape)

    similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
    # Print the inner list comprehension in a separate cell - what is really happening?

    predictions = similarity_values.argmax(0).cpu().numpy() #Why are we doing an argmax here?

    # Map argmax indices to identity strings
    pred_id_strings = [known_paths[i] for i in predictions]
    
    if mode == 'val':
      true_ids = pd.read_csv('./content/data/verification/dev_identities.csv')['label'].tolist()
      accuracy = accuracy_score(pred_id_strings, true_ids)
      print("Verification Accuracy = {}".format(accuracy))
    
    return pred_id_strings