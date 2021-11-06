import torch
from unet.unet_model import UNet
from functions.ModelTrainer import train_model
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os

from functions.SentinelLoader import SentinelPatchLoader
from functions.DataTransformer import GetDataTransformer

# options
batchSize = 64
sentinelPatchPath = "E:/Dropbox/Education/PhD/Projects/MucilageDetection/data_analysis/functions/patches/"

# define train images
TrainingImages = ["S2A_MSIL2A_20210402T085551_N0300_R007_T35TPF_20210402T133128",
                  "S2A_MSIL2A_20210509T084601_N0300_R107_T35TPF_20210509T115513",
                  "S2A_MSIL2A_20210512T085601_N0300_R007_T35TPF_20210512T133202"]
TrainingDataLoader = SentinelPatchLoader(sentinelPatchPath, TrainingImages, 200, True, GetDataTransformer())

# define validation images
ValidationImages = ["S2B_MSIL2A_20210414T084559_N0300_R107_T35TPF_20210414T112733"]
ValidationDataLoader = SentinelPatchLoader(sentinelPatchPath, ValidationImages, 200, True, GetDataTransformer())

# create custom data loader
DataLoader = {
    'train': DataLoader(TrainingDataLoader, batch_size=batchSize, shuffle=True, num_workers=0),
    'val': DataLoader(ValidationDataLoader, batch_size=batchSize, shuffle=True, num_workers=0)
}

# create pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a ResNetUNet model and apply it to model
model = UNet(n_channels=9, n_classes=1)
model = model.to(device)

# define optimizer and scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# use pre trained weights
if os.path.isfile('bestModel'):
    print('using the pre-trained best model weights...')
    weights = torch.load('bestModel')
    model.load_state_dict(weights)

# check for pre-trained model
if os.path.isfile('models/trainedModel'):
    print('loading the previous model...')
    
    # load the model
    checkpoint = torch.load('trainedModel')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
# start training
print('training model...')
train_model(model, optimizer, exp_lr_scheduler, DataLoader, device, num_epochs=50)
