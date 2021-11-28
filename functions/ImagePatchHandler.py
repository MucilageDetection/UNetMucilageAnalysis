import numpy as np

class ImagePatchHandler():
    def __init__(self, imageSize, windowSize, overlapSize):
        
        # first make sure image size is multiple of overlapSize
        self.overlapPadding = (overlapSize[0] - imageSize[0] % overlapSize[0], overlapSize[1] - imageSize[1] % overlapSize[1])
        self.overlapPaddedImageSize = (imageSize[0] + self.overlapPadding[0], imageSize[1] + self.overlapPadding[1])
        self.blockSize = (self.overlapPaddedImageSize[0] // overlapSize[0], self.overlapPaddedImageSize[1] // overlapSize[1])
        
        # compute the input and outputs
        self.numBlocks   = self.blockSize[0] * self.blockSize[1]
        self.inputRange  = np.empty((self.numBlocks, 4), dtype=np.int32)
        self.outputRange = np.empty((self.numBlocks, 4), dtype=np.int32)

        # set overlap window
        self.overlapSize = overlapSize

        # compute half window size
        self.windowSize = windowSize
        self.halfWindowSizeW = (self.windowSize[0] // 2, windowSize[0] - self.windowSize[0] // 2 - 1)
        self.halfWindowSizeH = (self.windowSize[1] // 2, windowSize[1] - self.windowSize[1] // 2 - 1)

        # find the active patch inside the window
        self.patchPaddingSize = ((self.windowSize[0] - self.overlapSize[0]) // 2, (self.windowSize[1] - self.overlapSize[1]) // 2)

        # now construct patches
        idx = 0
        for h in range(self.blockSize[1]):
            for w in range(self.blockSize[0]):
                self.outputRange[idx][0] = w * self.overlapSize[0]
                self.outputRange[idx][1] = h * self.overlapSize[1]
                self.outputRange[idx][2] = self.outputRange[idx][0] + self.overlapSize[0]
                self.outputRange[idx][3] = self.outputRange[idx][1] + self.overlapSize[1]

                self.inputRange[idx][0] = self.outputRange[idx][0]
                self.inputRange[idx][1] = self.outputRange[idx][1]
                self.inputRange[idx][2] = self.inputRange[idx][0] + windowSize[0]
                self.inputRange[idx][3] = self.inputRange[idx][1] + windowSize[1]
                
                # go to next index
                idx = idx + 1
    
    # return the number of patches
    def GetNumberOfPatches(self):
        return self.numBlocks
    
    # get the data
    def GetPaddedImage(self, data):

        # pad data to useful size
        rowsPadding = (self.patchPaddingSize[0], self.patchPaddingSize[1] + self.overlapPadding[1])
        colsPadding = (self.patchPaddingSize[0], self.patchPaddingSize[1] + self.overlapPadding[0])
        channelPadding = (int(0), int(0))
        
        return np.pad(data, (rowsPadding, colsPadding, channelPadding), 'reflect')

    # returns the cropped image
    def GetPatchImage(self, fullData, idx):
        return fullData[self.inputRange[idx][1]:self.inputRange[idx][3], self.inputRange[idx][0]:self.inputRange[idx][2], :].copy()

    # fill output
    def SetPatchImage(self, fullData, patchData, idx):
        patchData = np.squeeze(patchData)
        
        # crop the patch data
        patchData = patchData[self.patchPaddingSize[1] : (self.patchPaddingSize[1] + self.overlapSize[0]), self.patchPaddingSize[0] : (self.patchPaddingSize[0] + self.overlapSize[1])]

        # make sure the output indices are inside the image
        redundantW = max(0, self.outputRange[idx][2] - fullData.shape[1])
        redundantH = max(0, self.outputRange[idx][3] - fullData.shape[0])

        # do the final cropping
        patchData = patchData[0:(self.overlapSize[1]-redundantH), 0:(self.overlapSize[0]-redundantW)]

        # set the data
        fullData[self.outputRange[idx][1]:(self.outputRange[idx][3] - redundantH), self.outputRange[idx][0]:(self.outputRange[idx][2] - redundantW)] = patchData

        # return the data
        return fullData

