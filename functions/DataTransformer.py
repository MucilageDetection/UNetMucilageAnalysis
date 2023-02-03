from torchvision import transforms

# use the following for all dataset
def GetDataTransformer(resolution):

    # sentinel stats to make range [-1,1]
    if resolution == 10:
        bandMeans = [0.0381, 0.0340, 0.0237, 0.0207, 0.0239, 0.0219, 0.0219, 0.0207, 0.0177, 0.0159]
        bandStds  = [0.0152, 0.0145, 0.0119, 0.0116, 0.0124, 0.0119, 0.0121, 0.0121, 0.0116, 0.0107]
    else:
        bandMeans = [0.0381, 0.0340, 0.0237, 0.0239, 0.0219, 0.0219, 0.0207, 0.0177, 0.0159]
        bandStds  = [0.0152, 0.0145, 0.0119, 0.0124, 0.0119, 0.0121, 0.0121, 0.0116, 0.0107]

    # do transform
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(bandMeans, bandStds)
    ])
    
    return trans