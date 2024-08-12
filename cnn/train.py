# Script to prepare data and train a CNN4ANC model.
# Simple R Ratio test after training.

from metrics import R_ratio
from model import CNN4ANC
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import normalize_by_max, split_complex, combine_complex

starttime = time.time()

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = CNN4ANC().to(device)

# Set model save path
model_path = './models'
os.makedirs(model_path,exist_ok=True)
model_path += '/model-state-dict.pth'

# Load train data
train_data_path = '../data/august_data/train_data/005/'
EMIs = np.load(train_data_path+'train_EMI.npy')
RX = np.load(train_data_path+'train_Rx.npy')

# Preprocess EMI coil signals
EMIs_normalized = []
nb_channels = EMIs.shape[-1]
for ch in range(nb_channels):
    EMI_ch, _ = normalize_by_max(EMIs[:, :, ch])
    EMI_ch = split_complex(EMI_ch)
    EMIs_normalized.append(EMI_ch)
EMIs_normalized = np.stack((EMIs_normalized), axis=3)
train_EMIs = np.transpose(EMIs_normalized, (1, 0, 2, 3))

# Preprocess RX signals
train_RX, _ = normalize_by_max(RX)
train_RX = split_complex(train_RX)
train_RX = np.transpose(train_RX,(1,0,2))

train_emi, val_emi, train_rx, val_rx = train_test_split(train_EMIs,train_RX,shuffle=True,test_size=0.2)

train_emi = torch.tensor(train_emi,dtype=torch.float)
train_rx = torch.tensor(train_rx,dtype=torch.float)
val_emi = torch.tensor(val_emi,dtype=torch.float)
val_rx = torch.tensor(val_rx,dtype=torch.float)


# Set training parameters
batch_size = 64
epochs = 500
lr = 0.0005
num_workers = 8
optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer) # Update learning rate after a 10 epoch long plateau (default factor=0.1)
finisher = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0,patience=15) # End training after a 15 epoch long plateau
loss_fn = nn.MSELoss()


# Load datasets
train_ds = TensorDataset(train_emi,train_rx)
val_ds = TensorDataset(val_emi,val_rx)

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def train(dataloader,model,loss_fn, optimizer):
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    for i, (emi,rx) in enumerate(train_dl):
        inputs = emi.to(device)
        labels = torch.unsqueeze(rx,3).to(device)
        
        outs = model(inputs)
        #print(outs.shape,labels.shape)
        loss = loss_fn(outs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return round(train_loss/num_batches,10)

def val(dataloader,model,loss_fn):
    model.eval()
    eval_loss = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (emi,rx) in enumerate(dataloader):
            inputs = emi.to(device)
            labels = torch.unsqueeze(rx,3).to(device)
            outs = model(inputs)
            loss = loss_fn(outs,labels)
            
            eval_loss += loss.item()
    return round(eval_loss/num_batches,10)


train_loss = []
val_loss = []

best_model_i = 0
# Training Loop
for e in range(epochs):
    tloss = train(train_dl,model,loss_fn, optimizer)
    train_loss.append(tloss)

    vloss = val(val_dl,model,loss_fn)
    val_loss.append(vloss)
    finisher.step(vloss)
    if finisher.get_last_lr()==[0.0]:
        break
    scheduler.step(vloss)

    print(f'Epoch: {e}, Train Loss: {tloss}, Val Loss: {vloss}, lr: {scheduler.get_last_lr()}')
    
    if vloss <= np.min(np.array(val_loss)):
        torch.save(model.state_dict(),model_path)

endtime = time.time()

print(f'Training Complete! Training time elapsed {endtime-starttime}s\n')


## Prepare Testing Data
test_data_path = '../data/test_data/'

EMIs = np.load(test_data_path+'test_EMI.npy')
RX = np.load(test_data_path+'test_Rx.npy')

model = CNN4ANC().to(device)
model.load_state_dict(torch.load(model_path))

# Preprocess test data
EMIs_normalized = []
nb_channels = EMIs.shape[-1]
for ch in range(nb_channels):
    EMI_ch, _ = normalize_by_max(EMIs[:, :, ch])
    EMI_ch = split_complex(EMI_ch)
    EMIs_normalized.append(EMI_ch)
EMIs_normalized = np.stack((EMIs_normalized), axis=3)
test_EMIs = np.transpose(EMIs_normalized, (1, 0, 2, 3))
test_EMIs = torch.tensor(test_EMIs,dtype=torch.float).to(device)

RX_normalized, scale = normalize_by_max(RX)

# apply CNN noise cancellation
cnn_noise = model(test_EMIs)
cnn_noise = np.squeeze(cnn_noise,axis=3).cpu().detach().numpy()
cnn_noise = np.transpose(cnn_noise,(1,0,2))
cnn_noise = combine_complex(cnn_noise)
cnn_corrected = (RX_normalized - cnn_noise) * scale
cnn_R = R_ratio(cnn_corrected,RX)

print(f'CNN R Ratio: {cnn_R}\n')


'''
# prep data for testing EDITER
anc_EMIs = np.expand_dims(EMIs,1)
anc_RX = np.expand_dims(RX,1)

# apply EDITER noise cancellation
anc_corrected = ttc.anc.apply_editer(anc_RX,anc_EMIs) 
anc_corrected = np.squeeze(anc_corrected,axis=1)
anc_R = R_ratio(anc_corrected,RX)

print(f'EDITER R Ratio: {anc_R}\n')
'''
