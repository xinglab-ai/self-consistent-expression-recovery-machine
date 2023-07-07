#   SERM 
#
#   Produces a recovered gene expression matrix from expressions with dropout.
#
#   Written by Md Tauhidul Islam and Hongyi Ren, Radiation Oncology,
#   Stanford University

import os

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SparseAutoencoder(nn.Module):
    # The Sparse Autoencoder
    # 
    # Attribute:
    #  feature_dim: input data dimensions
    #  reduced_dim: neurons in the hidden layer
    def __init__(self, feature_dim, reduced_dim):
        super(SparseAutoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=feature_dim, out_features=reduced_dim)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=reduced_dim, out_features=feature_dim)
 
    def forward(self, x):
        # encoding
        x = torch.sigmoid(self.enc1(x))
 
        # decoding
        x = torch.clamp(self.dec1(x), min=0., max=1.)
        return x

class geneDataset(Dataset):
    # Dataset for gene expression data
    # 
    # Attribute:
    #  data: rows denote cells and columns denote the genes. 
    def __init__(self,data):
        self.data=data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
def _get_device():
    # help function to detect whether the computer has a GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def _kl_divergence(rho, rho_hat, device):
    # KL divergence function
    #
    # rho: the desired value
    # rho_hat: the activation value of hidden layer of Sparse Autoencoder
    # device: the device that the model is running on
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    
    return torch.sum(rho * torch.log(rho/rho_hat) + \
                     (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

def _sparse_regularizer(x, rho, model_children, device, hidden_layer=0):
    # sparse regularizer (KL Divergence loss)
    #
    # x: the input data
    # rho: the desired value
    # model_children: the layers of the Sparse Autoencoder
    # device: the device that the model is running on
    # hidden_layer: which layer is the hidden layer. 
    #  The KL Divergence regularizer will be applied on this layer.
    values = x
    regularizer = 0
    values = model_children[0](values)
    regularizer += _kl_divergence(rho, values, device)
    return regularizer

def _l2_regularizer(model_pars):
    # l2 regularizer
    #
    # model_pars: all the trainable parameters of the Sparse Autoencoder
    regularizer = 0
    for param in model_pars:
        regularizer += torch.norm(param)
    return regularizer

def fit(model, dataloader, epoch, criterion, optimizer,
        model_children, model_pars, l2, sparse, rho, device, trainset):
    # fit one epoch
    #
    # model: the Sparse Autoencoder model
    # epoch: the current epoch number
    # criterion: the loss function (by default MSE)
    # optimizer: the optimizer (by default Adam)
    # model_children: list of the layers
    # model_pars: all the trainable parameters
    # l2: the weight for l2 regularizer of all parameters
    # sparse: the weight for sparse regularizer (KL Divergence loss)
    # rho: the desired value for KL Divergence
    # device: the device that the model is running on
    # trainset: the training dataset
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader),
                        total=int(len(trainset)/dataloader.batch_size), disable=True):
        counter += 1
        img = data
        img = img.to(device).float()
        img = img.view(img.size(0), -1)
        optimizer.zero_grad()
        outputs = model(img)
        mse_loss = criterion(outputs, img)
        sparsity = _sparse_regularizer(img, rho, model_children, device)
        l2_reg = _l2_regularizer(model_pars)
        # add the sparsity penalty
        loss = mse_loss + l2 * l2_reg + sparse * sparsity
        #loss = mse_loss + l2 * l2_reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / counter
    return epoch_loss

def predict(model, dataloader, device):
    # predict the transformed data to calculate ideal distribution with trained model
    prediction_list = []
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, img in enumerate(dataloader):
            img = img.to(device).float()
            img = img.view(img.size(0), -1)
            pred = model(img)
            prediction_list.append(pred.cpu().numpy())
    return np.concatenate(prediction_list)

def trainAutoencoder(data, reduced_Dim, maxEPOCH, batchSize=64,
                     l2=0.05, sparse_rho=0.9, sparse=1):
    # train the Sparse Autoencoder model
    #
    # data: training data, rows denote cells and columns denote the genes.
    # reduced_Dim: neurons in the hidden layer
    # maxEPOCH: max training EPOCH number
    # batchSize: batchSize for training
    # l2: the weight for l2 regularizer of all parameters
    # sparse_rho: the desired value for KL Divergence
    # sparse: the weight for sparse regularizer (KL Divergence loss)
    device = _get_device()
    
    trainset = geneDataset(data)
    trainloader = DataLoader(trainset,
                             batch_size=batchSize,
                             shuffle=True)
    
    feature_dim = data.shape[1] # in matlab, each column is a sample
    model = SparseAutoencoder(feature_dim, reduced_Dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model_children = list(model.children())
    model_pars = model.parameters()
    
    train_loss = []
    for epoch in range(maxEPOCH):
        train_epoch_loss = fit(model, trainloader, epoch,
                               criterion, optimizer,
                               model_children, model_pars,
                               l2, sparse, sparse_rho, device, trainset)
        train_loss.append(train_epoch_loss)
    
    
    return predict(model, trainloader,device).T