import os

def listFiles(dirName):
    """
    list paths to all files in directory tree in parent directory
    Parameters:
        dirName: str of parent directory
    """
    listFile = os.listdir(dirName)
    allFiles = list()
    for entry in listFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + listFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles