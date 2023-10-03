# Example usage:
# print "0.mp4\n1.mp4" | python count.py -o_track=track -i_width=1280 -i_height=720 -yolo_weights=yolo_best.pt -yolo_size=640

import argparse
import datetime
import inspect
import logging
import os
import sys
import time

import numpy as np
import torch
import torchvision
import cv2
import av

from detectron2.structures import Boxes, Instances

import ultralytics

# BYTETracker needs this numpy fix...
np.float = float
from yolox.tracker.byte_tracker import BYTETracker


class Handler:
    def __init__(self, masker, predictor, tracker):
        self.masker = masker
        self.predictor = predictor
        self.tracker = tracker

    def h(self, frame, ts):
        img = torch.from_numpy(frame).cuda()
        cropped, masked1, maskedViz = self.masker.run(img, ts)
        masked = [masked1]

        outputs_batch = self.predictor.predict(self.predictor, masked, ts)
        ts.append({"name": "predict", "t": time.perf_counter()})

        trackPredImg = track(self.tracker, outputs_batch[0], maskedViz.cpu())
        ts.append({"name": "track", "t": time.perf_counter()})

        res = argparse.Namespace()
        res.track = trackPredImg
        return res


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot(inImg, tlwhs, obj_ids, scores, msg):
    im = np.copy(inImg)

    text_scale = 1
    text_thickness = 1
    line_thickness = 3

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{} {}'.format(int(obj_id), int(scores[i]*100))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0),
                    thickness=text_thickness)

    text_scale *= 3
    cv2.putText(im, msg,
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0), thickness=2)

    return im


def track(tracker, outputs, im):
    instances = outputs["instances"]

    trackBoxes = []
    trackIDs = []
    trackScores = []
    numObjs = instances.pred_classes.shape[0]
    if numObjs > 0:
        trackInput = np.zeros([numObjs, 5], dtype=np.float32)
        trackInput[:, 4] = instances.scores.cpu()
        trackInput[:, :4] = instances.pred_boxes.tensor.cpu()
        w, h = im.shape[1], im.shape[0]
        trackOutput = tracker.t.update(trackInput, [h, w], [h, w])
        for t in trackOutput:
            trackBoxes.append(t.tlwh)
            trackIDs.append(t.track_id)
            trackScores.append(t.score)

    for tid in trackIDs:
        tracker.ids[tid] = True

    trackPredImg = plot(im, trackBoxes, trackIDs, trackScores, f"egg: {len(tracker.ids)}")
    return trackPredImg


class Masker:
    def __init__(self, config, dtype):
        self.x1 = -1
        if config["crop"]["x"] < 0:
            return

        imgH, imgW = config["height"], config["width"]
        self.x1 = config["crop"]["x"]
        self.y1 = config["crop"]["y"]
        self.x2 = self.x1 + config["crop"]["w"]
        if self.x2 > imgW:
            self.x2 = imgW
        self.y2 = self.y1 + config["crop"]["w"]
        if self.y2 > imgH:
            self.y2 = imgH

        croppedH, croppedW = self.y2-self.y1, self.x2 - self.x1
        numChannels = config["numChannels"]
        mask = np.zeros([croppedH, croppedW, numChannels], dtype=dtype)
        slope = config["mask"]["slope"] / croppedW
        for x in range(croppedW):
            yS = config["mask"]["y"] + int(x*slope)
            yE = yS + config["mask"]["h"]
            mask[yS:yE, x] = 1
        self.mask = torch.from_numpy(mask).cuda()

        self.vizMask = torch.clip(self.mask.float() + 0.25, 0, 1)

    def run(self, img, ts):
        if self.x1 < 0:
            return img, img, img

        cropped = img[self.y1:self.y2, self.x1:self.x2]
        ts.append({"name": "crop", "t": time.perf_counter()})
        masked = cropped * self.mask
        maskedViz = (cropped.float() * self.vizMask).to(torch.uint8)
        ts.append({"name": "mask", "t": time.perf_counter()})
        return cropped, masked, maskedViz


def newYolov8(weights, yoloSize, shapes):
    model = ultralytics.YOLO(weights)

    myPredictor = argparse.Namespace()
    myPredictor.model = model

    functionals = []
    for shape in shapes:
        h, w = shape
        if w > h:
            resizedH, resizedW = int(yoloSize*h/w), yoloSize
            if (resizedH % 2) != 0:
                resizedH -= 1
            resize = torchvision.transforms.Resize([resizedH, resizedW], antialias=True)
            padWidth = int((yoloSize-resizedW)/2)
            padHeight = int((yoloSize-resizedH)/2)
            pad = torchvision.transforms.Pad([padWidth, padHeight])
        else:
            resizedH, resizedW = yoloSize, int(yoloSize*w/h)
            if (resizedW % 2) != 0:
                resizedW -= 1
            resize = torchvision.transforms.Resize([resizedH, resizedW], antialias=True)
            padWidth = int((yoloSize-resizedW)/2)
            padHeight = int((yoloSize-resizedH)/2)
            pad = torchvision.transforms.Pad([padWidth, padHeight])
        fn = argparse.Namespace()
        fn.h, fn.w = h, w
        fn.resize = resize
        fn.padHeight = padHeight
        fn.padWidth = padWidth
        fn.pad = pad
        functionals.append(fn)
    def predictFn(self, imgs, ts):
        chws = []
        for i, img in enumerate(imgs):
            hwc = img
            hwc = hwc.float()
            chw = torch.permute(hwc, [2, 0, 1])
            chw = chw / 255
            chw = torch.clamp(chw, min=0, max=0.9999)
            fn = functionals[i]
            chw = fn.resize(chw)
            chw = fn.pad(chw)
            chws.append(chw)
        bchw = torch.stack(chws, dim=0)
        ts.append({"name": "preprocess", "t": time.perf_counter()})
        results = self.model(bchw, half=True, verbose=False)
        ts.append({"name": "inference", "t": time.perf_counter()})

        batch = []
        for i, result in enumerate(results):
            fn = functionals[i]

            xyxy = result.boxes.xyxy.cpu().numpy()
            if fn.w > fn.h:
                xyxy[:, 1] -= fn.padHeight
                xyxy[:, 3] -= fn.padHeight
                xyxy *= (fn.w / yoloSize)
            else:
                xyxy[:, 0] -= fn.padWidth
                xyxy[:, 2] -= fn.padWidth
                xyxy *= (fn.h / yoloSize)

            instances = Instances([fn.h, fn.w])
            instances.pred_classes = result.boxes.cls
            instances.pred_boxes = Boxes(xyxy)
            instances.scores = result.boxes.conf
            o = {"instances": instances}
            batch.append(o)
        ts.append({"name": "boxes", "t": time.perf_counter()})
        return batch
    myPredictor.predict = predictFn
    return myPredictor


def newTracker():
    trackerArg = argparse.Namespace()

    # Only detections over track_thresh will be considered.
    trackerArg.track_thresh = 0.75

    trackerArg.track_buffer = 30

    # 1 - match_thresh is the minimum IOU.
    trackerArg.match_thresh = 0.99

    trackerArg.mot20 = False
    tracker = BYTETracker(trackerArg)

    t = argparse.Namespace()
    t.t = tracker
    t.ids = {}
    return t


def parseFNameTime(fname):
    if len(fname) < 19:
        return None, f"{len(fname)} {inspect.getframeinfo(inspect.currentframe())}"

    year = int(fname[:4])
    month = int(fname[4:6])
    day = int(fname[6:8])
    hour = int(fname[9:11])
    minute = int(fname[11:13])
    second = int(fname[13:15])
    milliSec = int(fname[16:19])

    t = datetime.datetime.utcnow().replace(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=milliSec*1000, tzinfo=datetime.timezone.utc)
    return t


def process(trackRoot, ipath, handler):
    rMux = av.open(ipath)
    rStream = rMux.streams.video[0]

    base = os.path.basename(ipath)
    start = parseFNameTime(base)
    trackDir = os.path.join(trackRoot, start.strftime("%Y%m%d"))
    os.makedirs(trackDir, exist_ok=True)
    trackFPath = os.path.join(trackDir, base)
    trackMux = av.open(trackFPath, mode="w")
    trackStream = trackMux.add_stream("h264", rate=rStream.average_rate)
    trackStream.width = rStream.width
    trackStream.height = rStream.height

    for frame in rMux.decode(rStream):
        ts = [{"t": time.perf_counter()}]

        img = frame.to_rgb().to_ndarray()
        out = handler.h(img, ts)

        trackF = av.VideoFrame.from_ndarray(out.track, format="rgb24")
        for packet in trackStream.encode(trackF):
            trackMux.mux(packet)

    for packet in trackStream.encode():
        trackMux.mux(packet)
    trackMux.close()

    rMux.close()


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-o_track")
    parser.add_argument("-i_width", type=int)
    parser.add_argument("-i_height", type=int)
    parser.add_argument("-yolo_weights")
    parser.add_argument("-yolo_size", type=int)
    args = parser.parse_args()

    err = mainWithErr(args)
    if err:
        logging.fatal("%s", err)


def mainWithErr(args):
    numChannels = 3
    masker = Masker({"width": args.i_width, "height": args.i_height, "numChannels": numChannels, "crop": {"x": -1}}, dtype=np.uint8)
    img = torch.from_numpy(np.zeros([args.i_height, args.i_width, numChannels], dtype=np.uint8)).cuda()
    _, masked, _ = masker.run(img, [])

    detector = newYolov8(args.yolo_weights, args.yolo_size, [[masked.shape[0], masked.shape[1]]])
    tracker = newTracker()
    handler = Handler(masker, detector, tracker)

    for line in sys.stdin:
        line = line.rstrip()
        process(args.o_track, line, handler)

    return None


if __name__ == "__main__":
        main()
