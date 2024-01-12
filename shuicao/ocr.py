import urllib.request
import os

import cv2
import paddleocr


def downloadImg(urlStr):
    dst = "input.jpg"
    urllib.request.urlretrieve(urlStr, dst)
    img = cv2.imread(dst)
    os.remove(dst)
    return img


def processImg(dst, img):
    result = ocr.ocr(img)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # 显示结果
    result = result[0]
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = paddleocr.draw_ocr(image, boxes, txts, scores, font_path="msjh.ttc")
    cv2.imwrite(dst, cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR))


# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)  # need to run only once to download and load model into memory

imgURLs = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/11.jpg",
    "https://images.prismic.io/xometry-marketing/548de24c-23f3-400d-9f2f-906e741288fc_laser-engraving-machine.jpg?auto=compress%2Cformat&fit=max&w=1000&h=562&fm=webp",
    "https://www.keyence.com/Images/ss_cij-application-selecting_017_1849954.jpg",
    "https://d2j6dbq0eux0bg.cloudfront.net/images/64746796/3460263946.jpg",
    "https://bsg-i.nbxc.com/product/37/2c/c3/3ad8b44aaace998891b1f55d18.png",
]
for i, imgURL in enumerate(imgURLs):
    img = downloadImg(imgURL)
    processImg(("img/%d.jpg"%(i)), img)
