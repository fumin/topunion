import os
import cv2

imgDir = "images-ori"
outDir = "images"

os.makedirs(outDir, exist_ok=True)

imgs = os.listdir(imgDir)
for img in imgs:
    fpath = os.path.join(imgDir, img)
    im = cv2.imread(fpath)

    scale = 0.25
    dim =  (int(im.shape[1] * scale), int(im.shape[0] * scale))
    resized = cv2.resize(im, dim)

    outPath = os.path.join(outDir, img)
    cv2.imwrite(outPath, resized)
