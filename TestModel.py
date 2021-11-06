import torch
from unet.unet_model import UNet
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as func
import scipy.io as sio
import os

from functions.SentinelLoader import SentinelTestDataset
from functions.DataTransformer import GetDataTransformer

# options
batchSize = 64
modelResolutionSize = 192
sentinelTestPath = "E:/Dropbox/Dataset/satellite/sentinel2/35TPE_MATDATA/"
outputDirectory = "outputs"

# create a ResNetUNet model and apply it to model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=9, n_classes=1)
model = model.to(device)
print('loading pretrained model...')
model.load_state_dict(torch.load('models/bestModel'))

# create directory
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

# define train images
TestingImages = {"S2B_MSIL2A_20210514T084559_N0300_R107_T35TPE_20210514T113538_20m"}
CropZone = (0,0, 4223, 1727)

for TestImage in TestingImages:
    
    # create loader
    TestingDataLoader = SentinelTestDataset(sentinelTestPath, TestImage, CropZone, modelResolutionSize, GetDataTransformer())
    
    # load the patches with batchSize
    TestDataLoader = {
        'test': DataLoader(TestingDataLoader, batch_size=batchSize, shuffle=False, num_workers=0)
    }
    
    # create output patches
    patchCountX = (CropZone[2] - CropZone[0]) // modelResolutionSize
    patchCountY = (CropZone[3] - CropZone[1]) // modelResolutionSize
    result = np.zeros((patchCountY,patchCountX,1,modelResolutionSize,modelResolutionSize), dtype=np.float32)
    
    # find the results
    for inputs, rows, cols in TestDataLoader['test']:
        inputs = inputs.to(device)
    
        # get the result
        with torch.set_grad_enabled(False):
            outputs = func.sigmoid(model(inputs)).contiguous()
    
        # convert tensor to np array
        outputNP = outputs.cpu().detach().numpy()
        rowsNP = rows.cpu().detach().numpy()
        colsNP = cols.cpu().detach().numpy()
        
        # add the result into the big array
        for i in range(outputNP.shape[0]):
            result[rowsNP[i],colsNP[i],:,:,:] = outputNP[i,:,:,:]
        
    # convert patches to image
    OutputFileName = TestImage + '_MUCILAGE.mat'
    sio.savemat(os.path.join(outputDirectory, OutputFileName), {'mucilage': result})