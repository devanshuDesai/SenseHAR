from model import *
from dataloader import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import warnings as wn
wn.filterwarnings('ignore')

#load either PAMAP2 or Opportunity Datasets
batch_size_train = 500 # PAM
batch_size_val = 300 # PAM
#batch_size_train = 10000 # OPP
#batch_size_val = 1 # OPP
# 1 = PAM, 0 = OPP
PAM_dataset = 1
if (PAM_dataset):
    # PAM Dataset
    train_dataset = Wearables_Dataset(0,dataset_name='PAM2',dataset_path='data/PAM2',train_dataset=True)
    val_dataset = Wearables_Dataset(0,dataset_name='PAM2',dataset_path='data/PAM2',train_dataset=False)
else:
    # Opportunity Dataset
    train_dataset = Wearables_Dataset(dataset_name='OPP',dataset_path='data/OPP',train_dataset=True)
    val_dataset = Wearables_Dataset(dataset_name='OPP',dataset_path='data/OPP',train_dataset=False)
# Get dataloaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          num_workers=4,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size_val,
                          num_workers=4,
                          shuffle=False)

writer = SummaryWriter()

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.xavier_uniform_(m.bias.data)

def plot(train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, dataset):
    # train/val acc plots
    x = np.arange(len(train_loss))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x,train_acc)
    ax1.plot(x,val_acc)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs. Validation Accuracy')
    ax1.legend(['Training Acc','Val Acc'])
    fig1.savefig('train_val_accuracy_' + dataset + '.png')
    # train/val loss plots
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x,train_loss)
    ax2.plot(x,val_loss)
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_title('Training vs. Validation Loss')
    ax2.legend(['Training Loss','Val Loss'])
    fig2.savefig('train_val_loss_' + dataset + '.png')
    # train/val f1 plots
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(x,train_f1)
    ax3.plot(x,val_f1)
    ax3.set_xlabel('Number of Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Training vs. Validation F1 Score')
    ax3.legend(['Train F1 Score','Val F1 Score'])
    fig3.savefig('train_val_f1_' + dataset + '.png')

def train():
    train_epoch_loss = []
    train_epoch_acc = []
    train_epoch_f1 = []
    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_f1 = []
    best_model = 'best_model_train'

    best_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        train_loss_per_iter = []
        train_acc_per_iter = []
        train_f1_per_iter = []
        ts = time.time()
        for iter, (X, Y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.cuda()
                labels = Y.long().cuda()
            else:
                inputs, labels = X, Y.long()
            clear(X, Y)
            inputs = torch.split(inputs, 9, 1)
            outputs = psm(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            clear(outputs)
            loss.backward()
            optimizer.step()
            clear(loss)

            # save loss per iteration
            train_loss_per_iter.append(loss.item())
            t_acc = compute_acc(outputs,labels)
            train_acc_per_iter.append(t_acc)
            micro_f1, macro_f1, weighted = calculate_f1(outputs, labels)
            train_f1_per_iter.append(weighted)
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Accuracy/train', t_acc, epoch)

        (print("Finish epoch {}, time elapsed {}, train acc {}, train weighted f1 {}".format(epoch, 
               time.time() - ts, np.mean(train_acc_per_iter), np.mean(train_f1_per_iter))))

        # calculate validation loss and accuracy
        val_loss, val_acc, val_f1 = val(epoch)
        print("Val loss {}, Val Acc {}, Val F1 {}".format(val_loss, val_acc, val_f1))
        # Early Stopping
        if loss < best_loss:
            best_loss = loss
            # TODO: Consider switching to state dict instead
            torch.save(psm, best_model)
        train_epoch_loss.append(np.mean(train_loss_per_iter))
        train_epoch_acc.append(np.mean(train_acc_per_iter))
        train_epoch_f1.append(np.mean(train_f1_per_iter))
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)
        val_epoch_f1.append(val_f1)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
    # plot val/training plot curves
    plot(train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc, train_epoch_f1, val_epoch_f1, 'shared')


def val(epoch):
    batch_loss = []
    batch_acc = []
    batch_f1 = []
    for iter, (X, Y) in tqdm(enumerate(val_loader), total=len(val_loader)):
        '''
        y -> Labels (Used for pix acc and IOU)
        tar -> One-hot encoded labels (used for loss)
        '''
        if use_gpu:
            inputs = X.cuda()
            labels = Y.long().cuda()
        else:
            inputs, labels = X, Y.long()
        clear(X, Y)
        inputs = torch.split(inputs, 9, 1)
        outputs = psm(inputs)
        # save val loss/accuracy
        loss = criterion(outputs, torch.max(labels, 1)[1])
        batch_loss.append(loss.item())
        batch_acc.append(compute_acc(outputs,labels))
        micro_f1, macro_f1, weighted = calculate_f1(outputs, labels)
        batch_f1.append(weighted)
        clear(outputs, loss)

        # if iter % 20 == 0:
        #     print("iter: {}".format(iter))

    return np.mean(batch_loss), np.mean(batch_acc), np.mean(batch_f1)


if __name__ == "__main__":
    # Define model parameters
    epochs    = 3
    criterion = nn.CrossEntropyLoss()
    sensors_per_device = 3
    fr = 100
    # Initialize model sensor model (senseHAR paper Figure 3/4)
    # Initialize encoder model A,B,C
    psm = PSM(12, sensors_per_device, fr, p=0.15)
    psm.apply(init_weights)
    params = psm.parameters()
    optimizer = optim.Adam(params, lr=1e-2)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        psm = psm.cuda()

    #print("Init val loss: {}, Init val acc: {}, Init val iou: {}".format(val_loss, val_acc, val_iou))
    train()

