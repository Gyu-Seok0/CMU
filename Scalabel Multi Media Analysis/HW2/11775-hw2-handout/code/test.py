import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import FeatureDataModule, MlpClassifier

from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score
import time

def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(accelerator='gpu', devices=1,
                        default_root_dir=osp.abspath(
                            osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)
    return args


def main(args):
    
    data_module = FeatureDataModule(args)
    val_df = data_module.val_df
    labels = val_df["Category"].tolist()
   
    # load
    model = MlpClassifier(args)
    logger = TensorBoardLogger(args.default_root_dir, args.name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback])
    
    start = time.time()
    trainer.fit(model, data_module)
    end = time.time()
    predictions = trainer.predict(datamodule=data_module, ckpt_path='best')

    pred = torch.cat(predictions, axis = 0)
    print("pred", pred.shape)

    # Calculate
    df = pd.DataFrame(confusion_matrix(labels, pred))
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(15):
        TP = df.iloc[i,i]
        FP = sum(df.iloc[:,i]) - TP
        TN = sum(df.iloc[i,:]) - TP
        FN = len(labels) - (TP + FP + TN)
        
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
    
    # Acc & Precision & Recall & F1-Score
    PRF_df = pd.DataFrame([TP_list, FP_list, TN_list, FN_list]).T
    PRF_df.columns = ["TP","FP","TN","FN"]
    PRF_df["Acurracy"] = PRF_df["TP"] / len(pred)
    PRF_df["Precision"] = PRF_df["TP"] / (PRF_df["TP"] + PRF_df["FP"] + 1e-10)
    PRF_df["Recall"] = PRF_df["TP"] / (PRF_df["TP"] + PRF_df["FN"] + 1e-10)
    PRF_df["F1-Score"] = 2*PRF_df["Precision"]*PRF_df["Recall"] / (PRF_df["Precision"] + PRF_df["Recall"] + 1e-10)

    top1 = accuracy_score(labels, pred)
    print("Top-1 Acc", top1)
    print("Training Time", end - start)

    #Save
    PRF_df.to_csv(f"PRF_df_{args.name}.csv")
    df.to_csv(f"Confusion_df_{args.name}.csv")

if __name__ == '__main__':
    main(parse_args())
