# python shuicao.py -src=/dev/video2
import argparse
import logging
import math
import os
import queue
import socket
import sys
import threading
import time

import cv2
import numpy as np
import paddleocr
import torch
import torchvision
import ultralytics


def intWord(x):
    high = x >> 8
    low = x & ((1 << 8)-1)
    return [high, low]


def wordInt(w):
    return (w[0] << 8) + w[1]


class ModbusClient:
    def __init__(self, host, port, moduleID):
        self.moduleID = moduleID
        self.txID = -1
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))

    def close(self):
        self.s.close()

    def readMultiple(self, registerStart, registerNum):
        register = intWord(registerStart)
        numReg = intWord(registerNum)

        dataSent = register + numReg
        self.send(0x03, dataSent)
        resp = self.recv()

        numBytes = resp[1]
        data = []
        i = 2
        while i+1 < len(resp):
            w = resp[i:i+2]
            data.append(wordInt(w))
            i += 2

        return data


    def writeMultiple(self, registerStart, registerData):
        """
        HeadHoldingRegister WritePoints NumberOfBytes DeviceData1 ... DeviceDataN
        2 byte              2 byte      1 byte        2 byte          2 byte
                                              |
                                              |---->  |<---------n * 2--------->|
        """
        register = intWord(registerStart)
        numReg = intWord(len(registerData))

        values = []
        for d in registerData:
            values += intWord(d)

        numBytes = intWord(len(values))[1:]
        data = register + numReg + numBytes + values

        self.send(0x10, data)
        resp = self.recv()

    def send(self, functionCode, data):
        """
        TransactionID ProtocolID MessageLength ModuleID FunctionCode Data
        2 byte        2 byte     2 byte        1 byte   1 byte       0-252 byte
                                       |
                                       |--->   |<---------------------->|
        """
        self.txID += 1

        msg = [self.moduleID, functionCode] + data
        msgLen = intWord(len(msg))

        txID = intWord(self.txID)
        protocolID = intWord(0)
        frame = txID + protocolID + msgLen + msg

        # logging.info("%s", " ".join("{:02x}".format(x) for x in frame))
        self.s.sendall(bytes(frame))

    def recv(self):
        resp = self.s.recv(1024)

        if len(resp) < 7:
            return []
        msgLen = wordInt(resp[4:6])
        moduleID = resp[6]
        data = resp[7:]
        return data


class ObjectDetectionPred:
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


def dist2D(x, y):
    return math.sqrt(math.pow(x[0]-y[0], 2)+math.pow(x[1]-y[1], 2))


class Target:
    def __init__(self, stateIdx, name, center):
        self.stateIdx = stateIdx
        self.name = name
        self.center = center

    def __str__(self):
        return "%s %s" % (self.name, self.center)


def setTargets(names, boxes, targets):
    for i in range(len(names)):
        name = names[i]
        box = boxes[i]

        tID = -1
        for i, t in enumerate(targets):
            if t.name == name:
                tID = i
                break
        if tID == -1:
            continue

        targets[tID].center = ((box[0][0]+box[2][0])/2, (box[0][1]+ box[2][1])/2)


class Match:
    def __init__(self, idx, target):
        self.idx = idx
        self.target = target


def findTarget(names, boxes, targets):
    matches = []
    for i in range(len(names)):
        name = names[i]
        box = boxes[i]

        tID = findTargetOne(name, box, targets)
        if tID != -1:
            matches.append(Match(i, targets[tID]))

    return matches


def findTargetOne(name, box, targets):
    center = ((box[0][0]+box[2][0])/2, (box[0][1]+ box[2][1])/2)
    size = (dist2D(box[0], box[1]), dist2D(box[0], box[3]))

    for i, t in enumerate(targets):
        sameName = (t.name == name)
        dist = dist2D(t.center, center)
        if sameName and False:
            logging.info("%s target %s center %s dist %s", name, t.center, center, dist)
        if sameName and (dist < 2):
            return i

    return -1


class Tank:
    def __init__(self):
        self.ok = 0

    def reset(self):
        self.ok = 0


class State:
    def __init__(self):
        self.counter = 0
        self.setTarget = False
        self.tanks = []
        for i in range(4):
            self.tanks.append(Tank())

        self.b = [0 for x in range(10)]

    def marshal(self):
        for i in range(len(self.b)):
            self.b[i] = 0

        for i, t in enumerate(self.tanks):
            self.b[i] = t.ok
        self.b[9] = self.counter

        return self.b

    def reset(self):
        for t in self.tanks:
            t.reset()


class Handler:
    def __init__(self, modbusC, ocr):
        self.modbusC = modbusC

        self.state = State()
        self.targets = [
                Target(0, "01", (411.0, 202.0)),
                Target(1, "02", (410.5, 189.0)),
                Target(2, "06", (412.5, 192.5)),
                Target(3, "04", (410.0, 190.0)),
                ]
        self.ocr = ocr

        self._profile_cnt = 0
        self._profile_seconds = 0

    def do(self, image):
        start_time = time.time()

        self._do(image)

        self._profile_cnt += 1
        self._profile_seconds += time.time() - start_time
        if self._profile_cnt > 100:
            fps = float(self._profile_cnt) / max(self._profile_seconds, 1e-3)
            self._profile_cnt = 0
            self._profile_seconds = 0

            logging.info("fps %.2f", fps)

    def _do(self, img):
        self.sync()
        predImg = self._ocr(img)
        self.report()
        cv2.imshow("img", predImg)

    def _ocr(self, img):
        vizEffectHeight = 25
        result = self.ocr.ocr(img[:img.shape[0]-vizEffectHeight], cls=False)
        result = result[0]
        txts, scores, boxes = [], [], []
        if result:
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            # The shape of boxes is: [text_count, 4 (rectangle_points), 2 (x, y)]
            boxes = np.stack([line[0] for line in result])

        if self.state.setTarget:
            setTargets(txts, boxes, self.targets)
        # logging.info("targets %s", [str(x) for x in self.targets])
        matches = findTarget(txts, boxes, self.targets)
        for m in matches:
            self.draw_match(img, m, txts)
        predImg = paddleocr.draw_ocr(img, boxes, txts, scores, font_path="msjh.ttc")

        self.state.reset()
        self.state.counter += 1
        for m in matches:
            sid = m.target.stateIdx
            self.state.tanks[sid].ok = 1

        return predImg

    def report(self):
        if self.state.setTarget:
            self.modbusC.writeMultiple(1020, [0])

        self.modbusC.writeMultiple(1010, self.state.marshal())

    def draw_match(self, img, match, txts):
        thickness = 100
        cv2.rectangle(img, [0, 0], (img.shape[1], img.shape[0]), [0, 0, 255], thickness)
        fontSize = 5
        loc = np.array([thickness/2, thickness/2+fontSize*10+5], dtype=int)
        cv2.putText(img, match.target.name, loc, cv2.FONT_HERSHEY_PLAIN, fontSize, (0, 0, 255), thickness=5)

    def sync(self):
        reg1020 = self.modbusC.readMultiple(1020, 1)
        if len(reg1020) != 1:
            return
        if reg1020[0] == 1:
            self.state.setTarget = True
        else:
            self.state.setTarget = False


class HaarCascade:
    def __init__(self):
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
        pred = ObjectDetectionPred()
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
    def __init__(self):
        yolo = ultralytics.YOLO("yolov8m.pt")
        self.yolo = yolo
        self.categories = yolo.names

    def predict(self, image):
        with torch.no_grad():
            raw, preprocessed = self.preprocess(image)
            output = self.yolo.predict(preprocessed, conf=0.9, verbose=False)
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
        pred = ObjectDetectionPred()
        pred.labels = result.boxes.cls.numpy()
        pred.scores = result.boxes.conf.numpy()
        pred.boxes = result.boxes.xyxy.numpy()
        pred.src = src[0]
        return pred


class SSDLite:
    def __init__(self):
        # self.weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        #         weights=self.weights)
        self.weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=self.weights)

        self.categories = self.weights.meta["categories"]

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

        pred = ObjectDetectionPred()
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
        self.img_ok = False
        self.img = None

        self.kill = False
        self.thread = None

    def open(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok = cap.isOpened()
        if not ok:
            return "not opened"
        self.cap = cap
        self.thread = threading.Thread(target=self._run, daemon=True)

        self.img_ok = False
        self.img = None
        self.thread.start()

        start_time = time.time()
        while time.time() - start_time < 3:
            if self.img is not None:
                break
            time.sleep(0.1)
        if self.img is None:
            return "no img"

        return ""

    def close(self):
        self.kill = True
        self.thread.join()
        self.kill = False
        self.cap.release()

    def read(self):
        return self.img_ok, self.img

    def _run(self):
        while True:
            if self.kill:
                break
            self.img_ok, self.img = self.cap.read()


class Img:
    def __init__(self, img):
        self._img = img

    def open(self):
        return ""

    def read(self):
        return True, self._img


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

    modbusHost = "192.168.1.111"
    modbusPort = 502
    modbusModuleID = 0x0F
    modbusC = ModbusClient(modbusHost, modbusPort, modbusModuleID)

    ocr = paddleocr.PaddleOCR(
        ocr_version="PP-OCRv4",
        use_angle_cls=True, lang="en",
        show_log=False)

    src = "rtsp://169.254.36.6:554/mpeg4cif?username=admin&password=123456"
    # src = args.src
    cap = VideoCapture(src)
    # cap = Img(cv2.imread("C:\\Users\\a3367\\Downloads\\3ad8b44aaace998891b1f55d18.png"))
    err = cap.open()
    if err:
        raise Exception(err)

    handler = Handler(modbusC, ocr)

    while True:
        ok, img = cap.read()
        if not ok:
            cap.close()
            err = cap.open()
            if err:
                logging.info("%s", err)
            continue
        handler.do(img)

        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == ord("s"):
            cv2.imwrite("data/{}.jpg".format(int(time.time()*1000)), img)
        if keyboard == ord("q"):
            break


if __name__ == "__main__":
    main()
