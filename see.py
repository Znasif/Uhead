from Process import Process
import numpy as np
import cv2
from tqdm import tqdm
import sys

if __name__ == "__main__":
    print("Maps/"+sys.argv[1])
    a = cv2.imread("Maps/"+sys.argv[1], 0)
    # a = cv2.GaussianBlur(a, (5, 5), 0)
    # a = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=1)
    b = Process.get_contour(a, 3)
    s = np.zeros_like(a, dtype=np.uint8)
    s[s==0] = 255
    si = s.copy()
    mn = len(b)
    for i in tqdm(range(mn)):
        r = []
        for k in b[i]:
            m, n = k[0]
            r.append((n, m))
        p, c, mnq = Process.region_growing(a.copy(), r)
        # print(c.shape)
        # ik = input()
        # cv2.drawContours(c, [j], 0, 255, 3)
        sp = c.shape[0]*c.shape[1]
        if sp > 50 and sp <2000:
            s = s & p
            if c.shape[1] >= 40 and c.shape[1] < 70:
                mns = c.shape[1]//2
                cv2.imwrite("Extracted/Experiment/"+str(mnq)+".jpg", c[::, :mns])
                cv2.imwrite("Extracted/Experiment/"+str((mnq[0], mnq[1]+mns))+".jpg", c[::, mns:])
            elif c.shape[1] >= 70:
                mns = c.shape[1]//3
                cv2.imwrite("Extracted/Experiment/"+str(mnq)+".jpg", c[::, :mns])
                cv2.imwrite("Extracted/Experiment/"+str((mnq[0], mnq[1]+mns))+".jpg", c[::, mns:mns*2])
                cv2.imwrite("Extracted/Experiment/"+str((mnq[0], mnq[1]+2*mns))+".jpg", c[::, mns*2:])
            else:
                cv2.imwrite("Extracted/Experiment/"+str(mnq)+".jpg", c)
    cv2.imwrite("Extracted/Experiment/all"+".jpg", s)
