# try to load the mat file
def MatFileImport(filepath):
    try:
        # load v7.2 or earlier
        import scipy.io as sio
        fmat = sio.loadmat(filepath)
    except:
        # v7.3 or greater
        import mat73
        fmat = mat73.loadmat(filepath, 'r')

    # return the file
    return fmat
