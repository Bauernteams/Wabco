import os
import natsort

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def listdir(path, onlyDirs = False, ext = None):
    if onlyDirs:
        folders = []
        for folder in natsort.natsorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path,folder)):
                folders.append(folder)
        return folders
    elif ext is not None:
        files = os.listdir(path)
        return [f for f in files if os.path.splitext(f)[1] == ext]
    else:
        return natsort.natsorted(os.listdir(path))

def getHighestFilenumber(path):
    files = listdir(path, ext=".pkl")
    filenumbers = [int(f.split(".")[0]) for f in files]
    if len(filenumbers) == 0:
        return 0
    else:
        return max(filenumbers)

