import numpy as np
import cv2
from tqdm import tqdm
import sys
from os import listdir
import re

if __name__ == "__main__":
    print("Maps/"+sys.argv[1])
    a = cv2.imread("Maps/"+sys.argv[1], 0)
    print(a.shape)
    c = np.zeros(a.shape, dtype=np.uint8)
    c = ~c
    direc = "Extracted/Experiment/"
    for i in tqdm(range(10)):
        for j in listdir(direc + str(i)):
            p = cv2.imread(direc + str(i) + "/" + j, 0)
            x, y = re.split('\)|,|\(|\.', j)[1:3]
            x, y = int(x), int(y)
            c[x:x+p.shape[0], y:y+p.shape[1]] = p
    cv2.imwrite("Extracted/new"+sys.argv[1], c)
