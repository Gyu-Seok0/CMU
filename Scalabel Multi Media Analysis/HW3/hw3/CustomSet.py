import torch
from torch.utils.data import Dataset
import numpy as np
import os.path as osp
import pickle

class CustomTrain(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        
        x_data = self.x[idx]
        x_data = torch.FloatTensor(x_data)

        y_data = self.y[idx]
        y_data = torch.tensor(y_data)

        return x_data, y_data


class CustomTest(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        x_data = self.x[idx]
        x_data = torch.FloatTensor(x_data)
        return x_data


class FeatureDataset(Dataset):

    def __init__(self, df, feature_dir, mean = None, var = None):
        self.df = df
        self.feature_dir = feature_dir
        self.mean = mean
        self.var = var

    def __len__(self):
        return len(self.df)

    def aggregate_frame_features(self, frame_features: np.ndarray) -> np.ndarray:
        
        ans = np.mean(frame_features, axis = 0)
        
        if ans.ndim == 1:
            return ans
        else:
            return ans.squeeze()

    def load_features(self, feature_path):
        features = []
        with open(feature_path, 'rb') as f:
            while True:
                try:
                    _, frame_feature = pickle.load(f)
                    features.append(frame_feature)
                except EOFError:
                    break
        return features

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        vid = item['Id']
        label = item.get('Category', None)
        feature_path = osp.join(self.feature_dir, f'{vid}.pkl')
        frame_features = np.stack(self.load_features(feature_path))

        feature = self.aggregate_frame_features(frame_features)
        feature = torch.as_tensor(feature, dtype=torch.float)
        if self.mean is not None:
            feature = (feature - self.mean) / self.var
        return feature, label

class ConcatData(Dataset):
    def __init__(self, datasets:list, test_dataset = False):
        self.datasets = datasets
        self.test_dataset = test_dataset
        
    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self,idx):
        x_data = []
        for dataset in self.datasets:
            target = dataset[idx]
            if type(target) == tuple:
                x_data.append(target[0])
            else:
                x_data.append(target)

        x_data = torch.concat(x_data)
        x_data = torch.FloatTensor(x_data)
        if self.test_dataset:
            return x_data
        
        y_data = self.datasets[0][idx][1]
        y_data = torch.tensor(y_data)

        return x_data, y_data