import cv2
import numpy as np
import random
import sys
import os
from tqdm import tqdm

src='dataset'
for root,directories,fs in os.walk(src):
    dirlen=len(directories)
    i=0
    for dirname in directories:
        print('>>>> Currently Processing Dir:',dirname)
        dirpath = os.path.join(root, dirname)

        for r1,d,files in os.walk(dirpath):
            flen=len(files)
            random.shuffle(files)
            for i in tqdm(range(0,int(flen*.6))):
                filename=files[i]
                filepath=os.path.join(r1,filename)
                image=cv2.imread(filepath)
                outfile = filename
                outdirpath=os.path.join('train',dirname)
                outfilepath = os.path.join(outdirpath, outfile)
                cv2.imwrite(outfilepath, image)
            for i in tqdm(range(int(flen*.6),int(flen*.8))):
                filename=files[i]
                filepath=os.path.join(r1,filename)
                image=cv2.imread(filepath)
                outfile = filename
                outdirpath=os.path.join('validate',dirname)
                outfilepath = os.path.join(outdirpath, outfile)
                cv2.imwrite(outfilepath, image)
            for i in tqdm(range(int(flen*.8),int(flen*1))):
                filename=files[i]
                filepath=os.path.join(r1,filename)
                image=cv2.imread(filepath)
                outfile = filename
                outdirpath=os.path.join('test',dirname)
                outfilepath = os.path.join(outdirpath, outfile)
                cv2.imwrite(outfilepath, image)



            #
        #
        i=i+1
    #
#
