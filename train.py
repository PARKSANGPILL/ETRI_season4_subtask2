from utils import *
from dataset import *
from coatnet import coatnet_1

import os
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

CFG = {
    'EPOCHS':200,
    'MIN_LR':1e-6,
    'MAX_LR':1e-4,
    'BATCH_SIZE':16,
    'STEP':16,
    'SEED':2023
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])

def main():
    if os.path.exists('models') is False:
        os.makedirs('models')

    net = coatnet_1().to(device)

    weight, train_dataloader = train_(CFG['BATCH_SIZE'])
    weight = torch.FloatTensor(weight).to(device)
    valid_dataloader = valid_(CFG['BATCH_SIZE'])
    
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=0)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=2, gamma=0.5, eta_max=CFG['MAX_LR'])
    
    best_val_acc = 0
    
    for epoch in range(1, CFG['EPOCHS']+1):
        net.train()
        train_loss = []
        batch = 0
        for images, labels in tqdm(iter(train_dataloader)):
            batch += 1
            images, labels = images.to(device), labels.to(device)
        
            outputs = net(images)
            loss1 = DiceLoss()(outputs, labels).to(device)
            loss2 = FocalLoss(weight, label_smoothing=0.1)(outputs, labels).to(device)
            loss = loss1 + loss2
            loss.backward()

            train_loss.append(loss.item())
            if (batch % CFG['STEP'] == 0) or (batch == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()  
                        
        net.eval()
        val_loss = []
        val_acc = []
        gt_list = np.array([])
        pred_list = np.array([])
        with torch.no_grad():
            for images, labels in tqdm(iter(valid_dataloader)):

                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                loss1 = DiceLoss()(outputs, labels).to(device)
                loss2 = FocalLoss(label_smoothing=0.1)(outputs, labels).to(device)
            
                loss = loss1 + loss2

                val_loss.append(loss.item())

                gt = np.array(labels.argmax(1).detach().cpu())
                gt_list = np.concatenate([gt_list, gt], axis=0)

                _, indx = outputs.max(1)
                pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)
                
                y_true, y_pred = gt_list.astype(np.int8), pred_list.astype(np.int8)
                cnf_matrix = confusion_matrix(y_true, y_pred) + 1e-9
                
                TP = np.diag(cnf_matrix)
    
                cs_accuracy = (TP / cnf_matrix.sum(axis=1)).mean()
                val_acc.append(cs_accuracy)
        
        _train_loss = np.mean(train_loss)
        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)
        
        print(f'Epoch [{epoch}]  Learning Rate [{scheduler.get_lr()[0]}]')
        print(f'Train Loss : [{_train_loss:.4f}]  Val Loss : [{_val_loss:.4f}]  Val ACC : [{_val_acc:.4f}]  Best Val ACC : [{best_val_acc:.4f}]')
        
        if scheduler is not None:
            scheduler.step()
        
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            torch.save(net.state_dict(), f"./models/epoch{epoch:02d}")
            patience_check = 0
            print('***** Best Model *****')


if __name__ == '__main__':
    main()
