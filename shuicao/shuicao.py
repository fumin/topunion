# python shuicao.py -src=/dev/video2
import argparse
import logging
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime
from PIL import Image as PILImage
from PIL import ImageFont as PILImageFont
import torch
import torchvision
import ultralytics
import deepsparse


class Prediction:
    def __init__(self):
        self.labels = np.zeros([0], dtype=int)
        self.scores = np.zeros([0], dtype=float)
        self.boxes = np.zeros([0, 4], dtype=float)
        self.src = None

    def draw(self, categories):
        drawn = self.src
        if len(self.labels) > 0:
            font = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
            if sys.platform == "win32":
                font = "cour.ttf"

            labels_text = []
            for i, idx in enumerate(self.labels):
                label = "%s %.0f" % (categories[idx], self.scores[i]*100)
                labels_text.append(label)

            drawn = torchvision.utils.draw_bounding_boxes(
                    self.src,
                    boxes=torch.from_numpy(self.boxes), labels=labels_text,
                    colors="red", width=4, font=font, font_size=30)

        chw = drawn
        hwc = chw.permute(1, 2, 0)
        bgr = hwc[:, :, [2, 1, 0]]
        return bgr.numpy()


class Handler:
    def __init__(self, size):
        self.model = SSDLite(size)
        self.model = Yolo(size)
        self.model = HaarCascade(size)
        self.model = NeuralMaginDeepSparse(size)

        self.cnt = 0
        self.spent_seconds = 0

    def do(self, image):
        start_time = time.time()

        self._do(image)

        self.cnt += 1
        self.spent_seconds += time.time() - start_time
        if self.cnt > 100:
            fps = float(self.cnt) / self.spent_seconds
            self.cnt = 0
            self.spent_seconds = 0

            logging.info("fps %.2f", fps)

    def _do(self, image):
        pred = self.model.predict(image)
        pred_img = pred.draw(self.model.categories)
        cv2.imshow("img", pred_img)


class NeuralMagicDeepSparse:
    def __init__(self, size):
        model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"
        self.yolo_pipeline = deepsparse.Pipeline.create(
                task="yolo",
                model_path=model_stub)
        self.categories = ultralytics.YOLO().names

    def predict(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = self.yolo_pipeline(images=[rgb], iou_thres=0.6, conf_thres=0.001)

        pred = Prediction()
        pred.labels = output.labels
        pred.scores = output.scores
        pred.boxes = output.boxes
        pred.src = self.to_torch(image)
        return pred

    def to_torch(self, rgb):
        rgb = torch.from_numpy(rgb)
        chw = rgb.permute(2, 0, 1)
        return chw


class HaarCascade:
    def __init__(self, size):
        self.face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.categories = ["face"]

    def predict(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
                gray_image, 1.3, 5, minSize=(128, 128))

        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x+w, y+h])
        pred = Prediction()
        pred.labels = np.zeros([len(boxes)], dtype=int)
        pred.scores = np.ones([len(boxes)], dtype=float)
        pred.boxes = np.array(boxes, dtype=float)
        pred.src = self.to_torch(image)
        return pred

    def to_torch(self, image):
        bgr = torch.from_numpy(image)
        rgb = bgr[:, :, [2, 1, 0]]
        chw = rgb.permute(2, 0, 1)
        return chw


class Yolo:
    def __init__(self, size):
        yolo = ultralytics.YOLO()
        self.yolo = yolo
        self.categories = yolo.names

    def predict(self, image):
        with torch.no_grad():
            raw, preprocessed = self.preprocess(image)
            output = self.yolo.predict(preprocessed, verbose=False)
            pred = self.postprocess(raw, output)
        return pred

    def preprocess(self, image):
        bgr = torch.from_numpy(image)
        rgb = bgr[:, :, [2, 1, 0]]
        chw = rgb.permute(2, 0, 1)
        batch = chw.unsqueeze(0)

        imgf = batch.float() / 255
        return batch, imgf

    def postprocess(self, src, output):
        result = output[0]
        pred = Prediction()
        pred.labels = result.boxes.cls.numpy()
        pred.scores = result.boxes.conf.numpy()
        pred.boxes = result.boxes.xyxy.numpy()
        pred.src = src[0]
        return pred


class SSDLite:
    def __init__(self, size):
        self.weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.categories = self.weights.meta["categories"]

        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                weights=self.weights, box_score_thresh=0.9)
        model.eval()
        model = torch.jit.script(model)
        model = torch.jit.freeze(model)
        self.model = model

        self.transforms = self.weights.transforms()

    def predict(self, image):
        with torch.no_grad():
            raw, preprocessed = self.preprocess(image)
            output = self.model([preprocessed[0]])
            pred = self.postprocess(raw, output)
        return pred

    def preprocess(self, image):
        bgr = torch.from_numpy(image)
        rgb = bgr[:, :, [2, 1, 0]]
        chw = rgb.permute(2, 0, 1)
        batch = chw.unsqueeze(0)

        imgf = self.transforms(batch)
        return batch, imgf

    def postprocess(self, raw, output):
        # Filter confident labels.
        out = output[1][0]
        labels = []
        scores = []
        boxes = []
        for i, label in enumerate(out["labels"].numpy()):
            s = out["scores"][i]
            if s < 0.9:
                continue

            labels.append(label)
            scores.append(s)
            boxes.append(out["boxes"][i].numpy())

        pred = Prediction()
        pred.src = raw[0]
        pred.labels = np.array(labels, dtype=int)
        pred.scores = np.array(scores, dtype=float)
        if len(labels) > 0:
            pred.boxes = np.stack(boxes)
        return pred
    

class VideoCapture:
    def __init__(self, src):
        self.src = src

        self.cap = None
        self.img = None

        self.thread = threading.Thread(target=self._run, daemon=True)

    def open(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok = cap.isOpened()
        if not ok:
            return False

        self.cap = cap
        self.thread.start()
        start_time = time.time()
        while time.time() - start_time < 3:
            if self.img is not None:
                break
            time.sleep(0.1)
        if self.img is None:
            return False

        return True

    def read(self):
        return True, self.img

    def _run(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                continue

            self.img = img


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-src")
    args = parser.parse_args()

    # cv2.setNumThreads(2)
    # torch.set_num_threads(4)

    src = args.src
    # src = "rtsp://admin:0000@192.168.1.121:8080/h264_ulaw.sdp"
    cap = VideoCapture(src)
    if not cap.open():
        raise Exception("not opened")

    _, img = cap.read()
    size = img.shape[:2]
    handler = Handler(size)

    while True:
        ok, img = cap.read()
        if not ok:
            continue
        handler.do(img)

        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == ord('q'):
            break


if __name__ == "__main__":
    main()
