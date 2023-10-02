'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
from dataset import ValidDataset
from coatnet import coatnet_1

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = ['/home/dcvlab/dcv/Genie/sub-task2/models/epoch71', '/home/dcvlab/dcv/Genie/sub-task2/models/epoch48']
df = pd.read_csv('/home/dcvlab/dcv/Coco/sub-task2/Dataset/info_etri20_color_validation.csv')
path = '/home/dcvlab/dcv/Coco/sub-task2/Dataset/Valid/'
x = [path + i for i in df['image_name']]
df['image_name'] = x
    
path_valid = df['image_name'].values
bbox_valid = df.iloc[:, 1:5].values
label_valid = F.one_hot(torch.as_tensor(df['Color'].values))
    
valid_dataset = ValidDataset(path_valid, bbox_valid, label_valid)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=10)

def main():
    net1 = coatnet_1().to(DEVICE)
    trained_weights1 = torch.load('./models/epoch71', map_location=DEVICE)
    net1.load_state_dict(trained_weights1)
    net1.eval()
        
    net2 = coatnet_1().to(DEVICE)
    trained_weights2 = torch.load('./models/epoch48', map_location=DEVICE)
    net2.load_state_dict(trained_weights2)
    net2.eval()

    gt_list = np.array([])
    pred_list = np.array([])
    val_acc = []
    for images, labels in tqdm(iter(valid_loader)):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs1 = net1(images)
        outputs2 = net2(images)
        outputs = torch.stack([outputs1, outputs2])
        outputs, _ = torch.max(outputs, dim=0)

        gt = np.array(labels.argmax(1).detach().cpu())
        gt_list = np.concatenate([gt_list, gt], axis=0)

        _, indx = outputs.max(1)
        pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)
                
        y_true, y_pred = gt_list.astype(np.int8), pred_list.astype(np.int8)
        cnf_matrix = confusion_matrix(y_true, y_pred) + 1e-9
                
        TP = np.diag(cnf_matrix)
    
        cs_accuracy = (TP / cnf_matrix.sum(axis=1)).mean()
        val_acc.append(cs_accuracy)
        
    _val_acc = np.mean(val_acc)

    print("------------------------------------------------------")
    print(
        "ACSA=%.5f" % (_val_acc))
    print("------------------------------------------------------")


if __name__ == '__main__':
    main()

