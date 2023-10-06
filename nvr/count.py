# Example usage:
# printf '{"type": "process", "dstTrack": "track.mp4", "src": "sample/short.mp4"}\n{"type": "quit"}' | python count.py -c='{"input": "sample/short.mp4", "device": "cpu", "mask": {"enable": false}, "yolo": {"weights": "yolo_best.pt", "size": 640}, "track": {"prevCounted": 10000}, "warmup": ["sample/short.mp4"]}'

import argparse
import datetime
import inspect
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torchvision
import cv2
import av

HAS_AI = False
if HAS_AI:
    from detectron2.structures import Boxes, Instances
    
    import ultralytics
    
    # BYTETracker needs this numpy fix...
    np.float = float
    from yolox.tracker.byte_tracker import BYTETracker


class Handler:
    def __init__(self, cfg, height, width):
        dtype = np.uint8
        masker = Masker(cfg["mask"], height, width, dtype, cfg["device"])
        numChannels = 3
        img = torch.from_numpy(np.zeros([height, width, numChannels], dtype=dtype)).to(cfg["device"])
        _, masked, _ = masker.run(img, [])

        detector = None
        tracker = None
        if HAS_AI:
            detector = newYolov8(cfg["yolo"]["weights"], cfg["yolo"]["size"], [[masked.shape[0], masked.shape[1]]])
            tracker = newTracker(cfg["track"])
 

        self.config = cfg
        self.masker = masker
        self.predictor = predictor
        self.tracker = tracker

    def h(self, img, ts):
        img = torch.from_numpy(img).to(self.config["device"])
        cropped, masked1, maskedViz = self.masker.run(img, ts)
        masked = [masked1]

        numCounted = 0
        trackPredImg = masked[0].cpu().numpy()
        if HAS_AI:
            outputs_batch = self.predictor.predict(self.predictor, masked, ts)
            ts.append({"name": "predict", "t": time.perf_counter()})

            numCounted, trackPredImg = track(self.tracker, outputs_batch[0], maskedViz.cpu())
            ts.append({"name": "track", "t": time.perf_counter()})

        res = argparse.Namespace()
        res.numCounted = numCounted
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

    numCounted = len(tracker.ids) - tracker.warmup
    numCounted += tracker.prevCounted
    trackPredImg = plot(im, trackBoxes, trackIDs, trackScores, f"egg: {numCounted}")
    return numCounted, trackPredImg


class Masker:
    def __init__(self, config, imgH, imgW, dtype, device):
        self.config = config
        if not config["enable"]:
            return

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
        self.mask = torch.from_numpy(mask).to(device)

        self.vizMask = torch.clip(self.mask.float() + 0.25, 0, 1)

    def run(self, img, ts):
        if not self.config["enable"]:
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


def newTracker(config):
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
    t.warmup = 0
    t.prevCounted = config["prevCounted"]
    return t


class Differ:
    def __init__(self, dstIndexPath, srcIndexPath):
        self.dstIndexPath = dstIndexPath
        self.srcIndexPath = srcIndexPath

        self.dstIndex = None

    def newDstIndex(self):
        dstIndex, err = parseIndex(self.dstIndexPath)
        if err:
            return err
        srcIndex, err = parseIndex(self.srcIndexPath)
        if err:
            return err

        # Find the index where dst is different from src.
        diffIdx = -1
        for i, dstLine in enumerate(dstIndex):
            if i >= len(srcIndex):
                diffIdx = i
                break
            srcLine = srcIndex[i]

            if dstLine.b != srcLine.b:
                diffIdx = i
                break
        if diffIdx == -1:
            diffIdx = len(dstIndex)

        self.dstIndex = dstIndex[:diffIdx]
        return None

    def getWarmup(self):
        # Loop backwards and find the first previous segment.
        urlIdx = -1
        i = len(self.dstIndex)
        while True:
            i -= 1
            if i < 0:
                break
            dstLine = self.dstIndex[i]

            if dstLine.tag == "":
                urlIdx = i
                break
            # Discontinuity means images from previous segments are separated from the current one.
            # In this case, there's no warmup to be done.
            if dstLine.tag == "EXT-X-DISCONTINUITY":
                break

        if urlIdx < 0:
            return "", None
        urlStr = self.dstIndex[urlIdx].b
        return urlStr, None

    def refresh(self):
        srcIndex, err = parseIndex(self.srcIndexPath)
        if err:
            return None, err

        # Check again that dst is the same as src, just to be on the safe side.
        for i, dstLine in enumerate(self.dstIndex):
            if i >= len(srcIndex):
                return None, f"dst longer than src {i} {len(srcIndex)} {self.dstIndex} {srcIndex} {inspect.getframeinfo(inspect.currentframe())}"
            srcLine = srcIndex[i]

            if dstLine.b != srcLine.b:
                return None, f"dst not equal to src {i} {dstLine} {srcLine} {self.dstIndex} {srcIndex} {inspect.getframeinfo(inspect.currentframe())}"

        # Find the next unprocessed segment.
        nextSegmentIdx = -1
        i = len(self.dstIndex) - 1
        while True:
            i += 1
            if i >= len(srcIndex):
                break
            srcLine = srcIndex[i]

            newLines.append(srcLine)
            if srcLine.tag == "":
                nextSegmentIdx = i
                break
        if nextSegmentIdx == -1:
            return [], None

        newLines = srcIndex[len(self.dstIndex) : (nextSegmentIdx+1)]
        return newLines, None

    def save(self):


def getImgSize(indexPath):
    # Find an example video file.
    dir = os.path.dirname(indexPath)
    entries = os.listdir(dir)
    ipath = ""
    for entry in entries:
        if entry.endswith(".ts"):
            ipath = os.path.join(dir, entry)
            break
    if ipath == "":
        return -1, -1, f"no videos in {dir} {inspect.getframeinfo(inspect.currentframe())}"

    mux = av.open(ipath)
    stream = mux.streams.video[0]
    h, w = stream.height, stream.width
    mux.close()
    return h, w, None


def handleVideo(dstTrack, src, handler):
    rMux = av.open(src)
    rStream = rMux.streams.video[0]

    if dstTrack != "":
        trackMux = av.open(dstTrack, mode="w")
        trackStream = trackMux.add_stream("h264", rate=rStream.average_rate)
        trackStream.width = rStream.width
        trackStream.height = rStream.height

    for frame in rMux.decode(rStream):
        ts = [{"t": time.perf_counter()}]

        img = frame.to_rgb().to_ndarray()
        out = handler.h(img, ts)

        if dstTrack != "":
            trackF = av.VideoFrame.from_ndarray(out.track, format="rgb24")
            for packet in trackStream.encode(trackF):
                trackMux.mux(packet)

    if dstTrack != "":
        for packet in trackStream.encode():
            trackMux.mux(packet)
        trackMux.close()

    rMux.close()


def process(differ, newLines, handler):
    if len(newLines) == 0:
        return None
    sgm = newLines[len(newLines)-1].b
    srcPath = os.path.join(src, sgm)
    dstPath = os.path.join(dst, sgm)
    handleVideo(dstPath, srcPath, handler)

    for l in newLines:
        differ.dstIndex.append(l)
    err = differ.save()
    if err:
        return err

    return None


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-c")
    args = parser.parse_args()

    err = mainWithErr(args)
    if err:
        logging.fatal("%s", err)


def mainWithErr(args):
    logging.info("%s", args.c)
    cfg = json.loads(args.c)

    dst = cfg["dst"]
    src = cfg["src"]
    differ = Differ(dst, src)
    err = differ.newDiffIdx()
    if err:
        return err
    height, width, err = getImgSize(src)
    if err:
        return err

    handler = Handler(cfg, height, width)

    # Warmup the tracker, so that it does not count objects in the first frame as new objects.
    warmup, err = differ.getWarmup()
    if err:
        return err
    if warmup != "":
        handleVideo("", warmup, handler)
    if HAS_AI:
        handler.tracker.warmup = len(tracker.ids)

    while True:
        newLines, err = differ.refresh()
        if err:
            return err
        err = process(differ, newLines, handler)
        if err:
            return err

    return None


if __name__ == "__main__":
        main()
