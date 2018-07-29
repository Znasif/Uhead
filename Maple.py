import cv2
import json
import random

f = open("contours.json", "r")
s = f.read()
annotations = json.loads(s)

dir = "Extracted/ALL/"
n = 20
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(10)]
fls = 10

for i in range(n):
    a = cv2.imread(dir + "Gen/" + str(i) + ".tif")
    for j in range(fls):
        for k in range(len(annotations[str(i)][j]['x'])):
            x, x_ = annotations[str(i)][j]['x'][k]
            y, y_ = annotations[str(i)][j]['y'][k]
            cv2.rectangle(a, (y, x), (y_, x_), color[j], 3)
    cv2.imwrite(dir + "Gen/" + str(i) + "a.tif", a)
