import numpy as np
import scipy.ndimage as ndi
import cv2

im = np.array([[10, 50, 10, 50, 10],
               [10, 55, 10, 55, 10],
               [10, 65, 10, 65, 10],
               [10, 50, 10, 50, 10]])
tr = np.array([[True, True, True, True, True],
               [True, True, True, True, True],
               [True, True, True, True, True],
               [True, True, True, True, True],])
k = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])


def create_idx(img):
    dim = max(img.shape[0], img.shape[1])
    a = np.repeat(np.arange(dim), dim).reshape((dim, dim))
    return np.dstack((a, a.T))


def getn(img, idx, coord, d=1):
    i, j = coord
    return idx[max(i - d, 0):min(i + d + 1, img.shape[0]),
           max(j - d, 0):min(j + d + 1, img.shape[1])]


def gets(coord, d, x, y):
    out = []
    i, j = coord
    for m in range(-d, d + 1):
        for n in range(-d, d + 1):
            if k[m+1, n+1]:
                out.append((min(max(i + m, 0), x - 1), min(max(j + n, 0), y - 1)))
    return out

'''
idn = create_idx(im)
tr = np.ones_like(im, dtype=bool)
while True:
    p, q = input().split()
    p, q = int(p), int(q)
    se = getn(im, idn, (p, q))
    for r in se[np.where(k)]:
        r = tuple(r)
        if tr[r]:
            print(im[r], end=" ")
            tr[r] = False
'''
tr = np.ones_like(im, dtype=bool)
while True:
    p, q = input().split()
    p, q = int(p), int(q)
    se = gets((p, q), 1, im.shape[0], im.shape[1])
    for i in se:
        if tr[i]:
            tr[i] = False
            print(im[i], end=" ")
