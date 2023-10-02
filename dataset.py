import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

class TrainDataset(Dataset):
    def __init__(self, path_list, bbox_list, label_list):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        i = np.random.randint(4)
        
        if i == 0:
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (384, 384))
            
            image = np.fliplr(image)
            image = image / 255.
            
            image = torch.as_tensor(image.copy(), dtype=torch.float32)
            
            label = self.label_list[index].type(torch.float32)
            
            return image.permute(2, 0, 1), label
        
        elif i == 1:
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (384, 384))
            
            image = cv2.GaussianBlur(image, (7, 7), 0)
            image = image / 255.
            
            image = torch.as_tensor(image, dtype=torch.float32)
            label = self.label_list[index].type(torch.float32)
            
            return image.permute(2, 0, 1), label
        
        elif i == 2:
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (384, 384))
            
            h, w, c= image.shape
            gauss = np.random.randn(h, w, c)
            sigma = 25.0
            noise = gauss * sigma
            image = image + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = image / 255.
            
            image = torch.as_tensor(image, dtype=torch.float32)
            label = self.label_list[index].type(torch.float32)
            
            return image.permute(2, 0, 1), label
        
        else:
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (384, 384))
            image = image / 255.
            image = torch.as_tensor(image, dtype=torch.float32)
            label = self.label_list[index].type(torch.float32)
            return image.permute(2, 0, 1), label
        
    def __len__(self):
        return len(self.path_list)

class ValidDataset(Dataset):
    def __init__(self, path_list, bbox_list, label_list):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        image1 = cv2.imread(self.path_list[index])
        image1 = image1[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
        image1 = cv2.resize(image1, (384, 384))
        image1 = image1 / 255.
        
        image1 = torch.as_tensor(image1, dtype=torch.float32)
        label1 = self.label_list[index].type(torch.float32)
            
        return image1.permute(2, 0, 1), label1
        
    def __len__(self):
        return len(self.path_list)

class EtriDataset(Dataset):
    def __init__(self, path_list, bbox_list, label_list, transforms=None):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.path_list[index]
        image = Image.open(img_path)
        if len(image.split()) > 3:
            r,g,b,_ = image.split()
            image = Image.merge('RGB', (r,g,b))
        image = image.crop((self.bbox_list[index][0], self.bbox_list[index][1], self.bbox_list[index][2], self.bbox_list[index][3]))
        if self.transforms is not None:
            image = self.transforms(image)
        
        label = self.label_list[index].type(torch.float32)
            
        return image, label
        
    def __len__(self):
        return len(self.path_list)
    
def train_(batch_size):
    df = pd.read_csv('/home/dcvlab/dcv/Coco/sub-task2/Dataset/info_etri20_color_train.csv')
    path = '/home/dcvlab/dcv/Coco/sub-task2/Dataset/train/'
    x = [path + i for i in df['image_name']]
    df['image_name'] = x
    
    path_train = df['image_name'].values
    bbox_train = df.iloc[:, 1:5].values
    label_train = F.one_hot(torch.as_tensor(df['Color'].values))
    
    color = [i for i in df['Color']]
    color.sort()
    weight = compute_class_weight(class_weight='balanced', classes=np.unique(color), y=color)
    
    train_dataset = TrainDataset(path_train, bbox_train, label_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    return weight, train_loader


def valid_(batch_size):
    df = pd.read_csv('/home/dcvlab/dcv/Coco/sub-task2/Dataset/info_etri20_color_validation.csv')
    path = '/home/dcvlab/dcv/Coco/sub-task2/Dataset/Valid/'
    x = [path + i for i in df['image_name']]
    df['image_name'] = x
    
    path_valid = df['image_name'].values
    bbox_valid = df.iloc[:, 1:5].values
    label_valid = F.one_hot(torch.as_tensor(df['Color'].values))
    
    valid_dataset = ValidDataset(path_valid, bbox_valid, label_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    
    return valid_loader