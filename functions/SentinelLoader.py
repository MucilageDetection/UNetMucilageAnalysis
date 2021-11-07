from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
from patchify import patchify

class SentinelPatchLoader(Dataset):
    def __init__(self, filepath, filenames, batchCount, loadAtOnce = False, transform=None):
        # create input image names and mask names
        # S2A_MSIL2A_20210402T085551_N0300_R007_T35TPF_20210402T133128_1
        self.inputs = []
        self.matfiles = []
        self.loadAtOnce = loadAtOnce
        
        # add each file to the list
        for file in filenames:
            for b in range(batchCount):
                self.inputs.append(os.path.join(filepath, file + '_' + str(b+1) + '.mat'))
        
        # if load at once selected, load at once
        if loadAtOnce:
            for file in self.inputs:
                self.matfiles.append(sio.loadmat(file))
        
        # think about the transform later
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.loadAtOnce:
            data = self.matfiles[idx]
        else:
            data = sio.loadmat(self.inputs[idx])
        
        # load the data file
        image = data['DataI'].astype(np.float32)
        label = data['LabelI'].astype(np.float32)
        
        # if transform defined, apply it
        if self.transform:
            image = self.transform(image)

        return [image, label]

class SentinelTestDataset(Dataset):
    def __init__(self, filename, cropZone, windowWidth, transform=None):
        # create input image names and mask names
        # S2A_MSIL2A_20210402T085551_N0300_R007_T35TPF_20210402T133128
        self.input = filename
        BandData = sio.loadmat(self.input)['BandData']
        BandData = BandData[cropZone[1]:cropZone[3], cropZone[0]:cropZone[2], :]
        
        # now find all the windows
        self.patchSize = (windowWidth, windowWidth, BandData.shape[2])
        self.patches = patchify(BandData, self.patchSize, windowWidth)
        
        # think about the transform later
        self.transform = transform
    
    def __len__(self):
        return self.patches.shape[0] * self.patches.shape[1]
    
    def __getitem__(self, idx):
        # get the subscript
        rows = int(idx / self.patches.shape[1])
        cols = int(idx % self.patches.shape[1])
        # print(f'{rows,cols}')
        # get the patch
        image = self.patches[rows, cols, 0, :, :, :].astype(np.float32)
        
        # if transform defined, apply it
        if self.transform:
            image = self.transform(image)
        
        return image, rows, cols