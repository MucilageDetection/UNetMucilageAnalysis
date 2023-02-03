from torch.utils.data import Dataset
import os
import numpy as np
from MatFileImporter import MatFileImport
from ImagePatchHandler import ImagePatchHandler

class SentinelPatchLoader(Dataset):
    def __init__(self, filepath, filenames, batchCount, dataAugmentationTypes, loadAtOnce = False, transform=None):
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
                self.matfiles.append(MatFileImport(file))
        
        # think about the transform later
        self.transform = transform
        self.dataAugmentationTypes = dataAugmentationTypes

    def __len__(self):
        return len(self.inputs) * len(self.dataAugmentationTypes)

    def __getitem__(self, idx):

        # rectify the indeces
        imageIndex =  idx % len(self.inputs)

        if self.loadAtOnce:
            data = self.matfiles[imageIndex]
        else:
            data = MatFileImport(self.inputs[imageIndex])
        
        # load the data file
        image = data['DataI'].astype(np.float32)
        label = data['LabelI'].astype(np.float32)
        
        # apply augmentations
        augmentationType =  self.dataAugmentationTypes[idx // len(self.inputs)]
        if augmentationType == 'original':
            pass
        elif augmentationType == 'vflip':
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        elif augmentationType == 'hflip':
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        else:
            print(f'unknown augmentation type {augmentationType}')
        
        # if transform defined, apply it
        if self.transform:
            image = self.transform(image)

        return [image, label]

class SentinelTestDataset(Dataset):
    def __init__(self, filename, cropZone, patchHandler, transform=None):
        # create input image names and mask names
        # S2A_MSIL2A_20210402T085551_N0300_R007_T35TPF_20210402T133128
        self.input = filename
        BandData = MatFileImport(self.input)['BandData']
        BandData = BandData[cropZone[1]:cropZone[3], cropZone[0]:cropZone[2], :]

        # now find all the windows
        self.patchHandler = patchHandler
        self.bandData = self.patchHandler.GetPaddedImage(BandData)

        # think about the transform later
        self.transform = transform
    
    def __len__(self):
        return self.patchHandler.GetNumberOfPatches()
    
    def __getitem__(self, idx):
        # get the patch
        image = self.patchHandler.GetPatchImage(self.bandData, idx).astype(np.float32)
        
        # if transform defined, apply it
        if self.transform:
            image = self.transform(image)
        
        return image, idx