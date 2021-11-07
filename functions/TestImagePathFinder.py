# download test files from dropbox
import os
import dropbox
import secrets

# checks that the given image is in the path, if not adds it to 'tempImage' path
def GetTestImagePath(folderPath, filePath, fileName):
    targetImageName = os.path.join(folderPath, filePath, fileName)
    
    if not os.path.exists(targetImageName):
        dbx = dropbox.Dropbox(secrets.dropbox_access_token)
        files = dbx.files_list_folder(filePath, recursive=True).entries
        
        # create output directory
        outputPath = 'tempImage'
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        # check that the file exist
        for entry in files:
            if entry.name == fileName:
                # file found, download it
                targetImageName = outputPath + '/' + entry.name
                if not os.path.exists(targetImageName):
                    print(f'Downloading {fileName}...')
                    dbx.files_download_to_file(targetImageName, entry.path_lower)
                else:
                    print(f'Target file already downloaded!')
                
    # return the address of the image
    return targetImageName