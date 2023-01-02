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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Model.CNN import *
from Model.net_sphere import *
from Model.resnet import *
from Model.arc_margin import * 
from Model.senet.se_resnet import *

from Module.execute import *
import argparse

from preprocess import *
import torchvision.models as models
from torch.optim.lr_scheduler import *
def main(args):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    config = {
        'batch_size': args.batch_size, # Increase this if your GPU can handle it
        'lr': args.lr,
        'epochs': args.epochs, # 10 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
        'model' : args.model, # Include other parameters as needed.
        'return_feats' : args.return_feats,
        "criterion" : args.criterion,
    }
    
    DATA_DIR = './content/data/11-785-f22-hw2p2-classification/'# TODO: Path where you have downloaded the data
    train_dataset, val_dataset, test_dataset = make_dataset(DATA_DIR)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], 
                                               shuffle = True, num_workers = 4, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['batch_size'], 
                                             shuffle = False, drop_last = False, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'],
                                              shuffle = False, drop_last = False, num_workers = 2)

    print("Number of classes: ", len(train_dataset.classes))
    print("No. of train images: ", train_dataset.__len__())
    print("No. of val images: ", val_dataset.__len__())
    print("No. of test images: ", test_dataset.__len__())
    print("Shape of image: ", train_dataset[0][0].shape)
    print("Batch size: ", config['batch_size'])
    print("Train batches: ", train_loader.__len__())
    print("Val batches: ", val_loader.__len__())
    print("Test batches: ", test_loader.__len__())
    
    #model = models.resnet34(weights="ResNet34_Weights.DEFAULT").to(device)
    model = se_resnet50(num_classes = 7000).to(device)
    model.fc = nn.Identity()

    metric_fc = ArcMarginProduct(2048, 7000, s=30, m=0.35).to(device)
    
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=[0, 1])

    if args.pretrain is not None:
        dic = torch.load(args.pretrain)
        model.load_state_dict(dic["model_state_dict"])
        print("[Done] Loading the pretrained model")
    else:
        print("[Fail] Load the Pretrain Model")

    
    x_train = torch.randn(2,3,224,224).to(device)
    y_train = torch.tensor([0,1]).to(device)

    feats = model(x_train)
    print("feats", feats.shape) # 2 x 512
    output = metric_fc(feats, y_train)
    print("[Exist Label] output", output.shape) # 2 x 7000
    output = metric_fc(feats)
    print("[Not Exist Label] output", output.shape) # 2 x 7000
    
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr = config["lr"], weight_decay = 1e-4, momentum = 0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    gc.collect()
    torch.cuda.empty_cache()


    
    if args.train_mode:
        # wandb
        wandb.login(key="e0408f5d7b96be3d00be30b39eda0f1e259672ed") #API Key is in your wandb account, under settings (wandb.ai/settings)
        wandb.init()
        run = wandb.init(
            name = args.project, ## Wandb creates random run names if you skip this field
            reinit = True, ### Allows reinitalizing runs when you re-run this cell
            # run_id = ### Insert specific run id here if you want to resume a previous run
            # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
            project = "hw2p2-ablations", ### Project should be created in your wandb account 
            config = config ### Wandb Config for your run
        )
        
        print("[Train Mode]")
        best_valacc = 0.0

        for epoch in range(config['epochs']):

            curr_lr = float(optimizer.param_groups[0]['lr'])
            
            train_acc, train_loss = train(model, train_loader, optimizer, criterion, None, config, device, epoch, metric_fc)
            
            print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1,
                config['epochs'],
                train_acc,
                train_loss,
                curr_lr))
            
            val_acc, val_loss = validate(model, val_loader, criterion, config, device, metric_fc)
            
            print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

            wandb.log({"train_loss":train_loss, 'train_Acc': train_acc, 'validation_Acc':val_acc, 
                    'validation_loss': val_loss, "learning_Rate": curr_lr})
            
            # If you are using a scheduler in your train function within your iteration loop, you may want to log
            # your learning rate differently 

            # #Save model in drive location if val_acc is better than best recorded val_acc
            if val_acc >= best_valacc:
            #path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
                print("Saving model")
                torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        #'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': val_acc, 
                        'epoch': epoch}, f'./{args.project}.pth')
                best_valacc = val_acc
                wandb.save(f'{args.project}.pth')
                if args.test_mode:
                    print("[Test Mode]")
                    
                    if args.recognition:
                        print("[Test1: Recognition]")
                        # classification
                        test_results = test(model, test_loader, device, config, metric_fc)
                        with open(f"classification_{args.project}.csv", "w+") as f:
                            f.write("id,label\n")
                            for i in range(len(test_dataset)):
                                f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", test_results[i]))

                    if args.verification:
                        print("[Test2: Verification]")

                        # verification
                        known_regex = "./content/data/verification/known/*/*"
                        known_paths = [i.split('/')[-2] for i in sorted(glob.glob(known_regex))]
                        
                        # This obtains the list of known identities from the known folder

                        val_unknown_regex = "./content/data/verification/unknown_dev/*" #Change the directory accordingly for the test set
                        test_unknown_regex = "./content/data/verification/unknown_test/*" 

                        # We load the images from known and unknown folders
                        val_unknown_images = [Image.open(p) for p in tqdm(sorted(glob.glob(val_unknown_regex)))]
                        test_unknown_images = [Image.open(p) for p in tqdm(sorted(glob.glob(test_unknown_regex)))] 
                        known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

                        # Why do you need only ToTensor() here?
                        ver_transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor()])

                        val_unknown_images = torch.stack([ver_transforms(x) for x in val_unknown_images])
                        test_unknown_images = torch.stack([ver_transforms(x) for x in test_unknown_images])
                        known_images  = torch.stack([ver_transforms(y) for y in known_images ])
                        #Print your shapes here to understand what we have done

                        # You can use other similarity metrics like Euclidean Distance if you wish
                        similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6) 
                        
                        val_pred_id_strings = eval_verification(val_unknown_images,known_images,
                                                            model, similarity_metric,
                                                            config['batch_size'], mode='val', 
                                                            known_paths = known_paths,
                                                            device = device,
                                                            config = config)

                        test_pred_id_strings = eval_verification(test_unknown_images,known_images,
                                                            model, similarity_metric,
                                                            config['batch_size'], mode='test', 
                                                            known_paths = known_paths,
                                                            device = device,
                                                            config = config)
                        
                        with open(f"verification_{args.project}.csv", "w+") as f:
                            f.write("id,label\n")
                            for i in range(len(test_pred_id_strings)):
                                f.write("{},{}\n".format(i, test_pred_id_strings[i]))
            # You may find it interesting to exlplore Wandb Artifcats to version your models
            scheduler.step()
        run.finish()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type = str, default = None, help = "pretrain model's path")
    parser.add_argument("--project", type = str, default = None, help = "name of project")
    
    parser.add_argument("--lr", type = float, default=0.1, help = "learning_rate")
    parser.add_argument("--epochs",type = int, default = 40)
    parser.add_argument("--batch_size", type = int, default =256)
    
    parser.add_argument("--train_mode", type = bool, default = True)
    parser.add_argument("--test_mode", type = bool, default = True)
    parser.add_argument("--recognition", type = bool, default = True)
    parser.add_argument("--verification", type = bool, default= True)
    parser.add_argument("--return_feats", type = bool, default= True)

    parser.add_argument("--model", type = str, default = "ResNet", help = "ArcFace or SphereNet or Simple, Conv_Sphere")
    parser.add_argument("--criterion", type = str, default = "CrossEntropy", help = "Sphereface or CrossEntropy")

    args = parser.parse_args()

    print(args)
    main(args)
    print(args)