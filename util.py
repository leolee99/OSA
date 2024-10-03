import os
import torch
import random
import numpy as np
import logging
import torch.backends.cudnn as cudnn

def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    return args.seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def cocoimage_rename():
    folder_path = "$PATH$"
    filelist = os.listdir(folder_path)  
    for files in filelist:   
        Olddir = os.path.join(folder_path, files)
        if os.path.isdir(Olddir):
                continue
        filename = os.path.splitext(files)[0]     
        filetype = os.path.splitext(files)[1]
        new_name = "COCO_2014_" + filename.split('_')[2] + filetype
        Newdir = os.path.join(folder_path, new_name)
        os.rename(Olddir, Newdir)   
    return True

if __name__ == '__main__':
    cocoimage_rename()