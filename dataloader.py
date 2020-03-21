from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import utils
import numpy as np
import torch
import pandas as pd

class Wearables_Dataset(Dataset):

    def __init__(self, dataset_number, dataset_name, dataset_path, train_dataset):
        # PAM dataset
        # x_train size = (14438,27,512,1)
        # y_train size = (14438,12)
        # x_val size = (2380,27,512,1)
        # y_val size = (2380,12)
        # Opportunity Dataset
        # x_train size = (8198,45,150,1)
        # y_train size = (8198,4)
        # x_val size = (1256,45,150,1)
        # y_val size = (1256,4)
        self.num_classes, self.sensors, self.locations, self.label_names, self.f_hz, \
            self.dimensions, self.path = utils.get_details(dataset_name,dataset_path)
        x_train, y_train, x_val, y_val = utils.load_dataset(dataset_name, self.path, self.num_classes)
        x_train, x_val = utils.data_reshaping(x_train, x_val)
        mean_x = np.mean(x_train,axis=0)
        # train dataset
        if (train_dataset):
            # normalize input data
            self.num_samples = x_train.shape[0]
            x_norm = x_train/mean_x
            if (dataset_number == 1):
                self.x = x_norm[:,0:9,:,:]
                self.y = y_train
            elif (dataset_number == 2):
                self.x = x_norm[:,9:18,:,:]
                self.y = y_train
            elif (dataset_number == 3):
                self.x = x_norm[:,18:27,:,:]
                self.y = y_train
            else:
                self.x = x_norm
                self.y = y_train

        # val dataset
        else:
            self.num_samples = x_val.shape[0]
            x_norm = x_val/mean_x
            if (dataset_number == 1):
                self.x = x_norm[:,0:9,:,:]
                self.y = y_val
            elif (dataset_number == 2):
                self.x = x_norm[:,9:18,:,:]
                self.y = y_val
            elif (dataset_number == 3):
                self.x = x_norm[:,18:27,:,:]
                self.y = y_val
            else:
                self.x = x_norm
                self.y = y_val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

