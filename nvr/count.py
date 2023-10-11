# Example usage:
# python count.py -c='{"TrackIndex": "track/index.m3u8", "Src": "server/record/20060102/20060102_150405_000000/rtsp0/index.m3u8", "Device": "cpu", "Mask": {"Enable": false}, "Yolo": {"Weights": "yolo_best.pt", "Size": 640}, "Track": {"PrevCount": 10000}}'

import argparse
import datetime
import inspect
import json
import logging
import os
import select
import sys
import threading
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


def readline(secs):
    rlist, _, _ = select.select([sys.stdin], [], [], secs)
    if rlist:
        line = sys.stdin.readline()
    else:
        line = ""
    return line


class Handler:
    def __init__(self, cfg, height, width):
        dtype = np.uint8
        masker = Masker(cfg["Mask"], height, width, dtype, cfg["Device"])
        numChannels = 3
        img = torch.from_numpy(np.zeros([height, width, numChannels], dtype=dtype)).to(cfg["Device"])
        _, masked, _ = masker.run(img, [])

        detector = None
        tracker = None
        if HAS_AI:
            detector = newYolov8(cfg["Yolo"]["Weights"], cfg["Yolo"]["Size"], [[masked.shape[0], masked.shape[1]]])
            tracker = newTracker(cfg["Track"])
 

        self.config = cfg
        self.masker = masker
        self.detector = detector
        self.tracker = tracker

    def h(self, img, ts):
        img = torch.from_numpy(img).to(self.config["Device"])
        cropped, masked1, maskedViz = self.masker.run(img, ts)
        masked = [masked1]

        numCounted = 0
        trackPredImg = masked[0].cpu().numpy()
        if HAS_AI:
            outputs_batch = self.detector.predict(self.detector, masked, ts)
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
    numCounted += tracker.prevCount
    trackPredImg = plot(im, trackBoxes, trackIDs, trackScores, f"egg: {numCounted}")
    return numCounted, trackPredImg


class Masker:
    def __init__(self, config, imgH, imgW, dtype, device):
        self.config = config
        if not config["Enable"]:
            return

        self.x1 = config["Crop"]["X"]
        self.y1 = config["Crop"]["Y"]
        self.x2 = self.x1 + config["Crop"]["W"]
        if self.x2 > imgW:
            self.x2 = imgW
        self.y2 = self.y1 + config["Crop"]["W"]
        if self.y2 > imgH:
            self.y2 = imgH

        croppedH, croppedW = self.y2-self.y1, self.x2 - self.x1
        numChannels = 3
        mask = np.zeros([croppedH, croppedW, numChannels], dtype=dtype)
        slope = config["Mask"]["Slope"] / croppedW
        for x in range(croppedW):
            yS = config["Mask"]["Y"] + int(x*slope)
            yE = yS + config["Mask"]["H"]
            mask[yS:yE, x] = 1
        self.mask = torch.from_numpy(mask).to(device)

        self.vizMask = torch.clip(self.mask.float() + 0.25, 0, 1)

    def run(self, img, ts):
        if not self.config["Enable"]:
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
    t.prevCount = config["PrevCount"]
    return t


class M3U8Line:
    def __init__(self):
        self.b = ""
        self.tag = ""
    def __repr__(self):
        return f"M3U8Line{{tag: {self.tag}, b: {self.b}}}"


def newM3U8Line(line):
    m = M3U8Line()
    m.b = line
    if not line.startswith("#"):
        return m

    # Remove preceding #
    line = line[1:]
    colonSS = line.split(":")
    m.tag = colonSS[0]

    return m


def readIndex(fpath):
    with open(fpath) as f:
        b = f.read()
    lines = b.split("\n")

    # Remove empty line at the end
    if len(lines) >= 1 and lines[len(lines)-1] == "":
        lines = lines[:len(lines)-1]

    m3u8s = []
    for l in lines:
        m = newM3U8Line(l)
        m3u8s.append(m)

    return m3u8s, None


class Differ:
    def __init__(self, trackIndexPath, srcIndexPath):
        self.trackIndexPath = trackIndexPath
        self.srcIndexPath = srcIndexPath

        self.trackIndex = None

    def __repr__(self):
        return f"Differ{{{self.trackIndexPath} {self.srcIndexPath} {self.trackIndex}}}"

    def getWarmup(self):
        trackIndex, err = self._load()

        # Loop backwards and find the first previous segment.
        urlIdx = -1
        i = len(trackIndex)
        while True:
            i -= 1
            if i < 0:
                break
            trackLine = trackIndex[i]

            if trackLine.tag == "":
                urlIdx = i
                break
            # ENDLIST and DISCONTINUITY means images from previous segments are separated from the current one.
            # In these cases, there's no warmup to be done.
            if trackLine.tag == "EXT-X-ENDLIST":
                break
            if trackLine.tag == "EXT-X-DISCONTINUITY":
                break

        urlStr = ""
        if urlIdx >= 0:
            base = trackIndex[urlIdx].b
            srcDir = os.path.dirname(self.srcIndexPath)
            urlStr = os.path.join(srcDir, base)

        # Do not load ENDLIST, as it is sure to be different after the next refresh.
        if len(trackIndex) > 0:
            last = trackIndex[len(trackIndex)-1]
            if last.tag == "EXT-X-ENDLIST":
                trackIndex = trackIndex[ : len(trackIndex)-1]
        self.trackIndex = trackIndex

        return urlStr, None

    def refresh(self):
        srcIndex, err = readIndex(self.srcIndexPath)
        if err:
            return None, err

        # Check again that track is the same as src, just to be on the safe side.
        for i, trackLine in enumerate(self.trackIndex):
            if i >= len(srcIndex):
                return None, f"track longer than src {i} {len(srcIndex)} {self.trackIndex} {srcIndex} {inspect.getframeinfo(inspect.currentframe())}"
            srcLine = srcIndex[i]

            if trackLine.b != srcLine.b:
                return None, f"dst not equal to src {i} {trackLine} {srcLine} {self.trackIndex} {srcIndex} {inspect.getframeinfo(inspect.currentframe())}"

        # Find the next unprocessed segment.
        nextSegmentIdx = -1
        i = len(self.trackIndex) - 1
        while True:
            i += 1
            if i >= len(srcIndex):
                break
            srcLine = srcIndex[i]

            if srcLine.tag == "EXT-X-ENDLIST" or srcLine.tag == "EXT-X-DISCONTINUITY" or srcLine.tag == "":
                nextSegmentIdx = i
                break
        if nextSegmentIdx == -1:
            nextSegmentIdx = len(srcIndex)-1

        newLines = srcIndex[len(self.trackIndex) : (nextSegmentIdx+1)]
        return newLines, None

    def _load(self):
        trackIndex = []
        if os.path.isfile(self.trackIndexPath):
            trackIndex, err = readIndex(self.trackIndexPath)
            if err:
                return None, err
        srcIndex, err = readIndex(self.srcIndexPath)
        if err:
            return None, err

        # Find the index where track is different from src.
        diffIdx = -1
        for i, trackLine in enumerate(trackIndex):
            if i >= len(srcIndex):
                diffIdx = i
                break
            srcLine = srcIndex[i]

            if trackLine.b != srcLine.b:
                diffIdx = i
                break
        if diffIdx == -1:
            diffIdx = len(trackIndex)

        trackIndex = trackIndex[:diffIdx]
        return trackIndex, None

    def save(self):
        b = "\n".join([m.b for m in self.trackIndex])
        with open(self.trackIndexPath, "w") as f:
            f.write(b)


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


class Frame:
    def __init__(self, img, time_base, pts, dts):
        self.img = img
        self.time_base = time_base
        self.pts = pts
        self.dts = dts


def writeVideo(fpath, rate, time_base, pts, dts, height, width, frames):
    options = {
        # "movflags": "frag_keyframe",
        # "muxdelay": "10",
        # "muxpreload": "10",
        # "output_ts_offset": "10",
    }
    mux = av.open(fpath, mode="w", format="mp4", options=options)
    stream = mux.add_stream("h264", rate=rate)
    # stream = mux.add_stream("h264")
    stream.codec_context.flags |= "GLOBAL_HEADER"
    stream.time_base = time_base
    stream.codec_context.time_base = time_base
    stream.height = height
    stream.width = width

    prevPTS = -1
    for frm in frames:
        frame = av.VideoFrame.from_ndarray(frm.img, format="rgb24")
        frame.time_base = frm.time_base
        frame.pts = frm.pts
        # frame.pts = int(frm.pts * frm.time_base)
        if frame.pts <= prevPTS:
            frame.pts = prevPTS + 1
        prevPTS = frame.pts
        if os.path.basename(fpath) == "1697046462_4200000.ts":
            logging.info("frame %s %s %s", os.path.basename(fpath), frame.pts, frame.time_base)
        for packet in stream.encode(frame):
            if os.path.basename(fpath) == "1697046462_4200000.ts":
                stream = mux.streams.video[0]
                logging.info("packet %s %s %s %s %s %s %s %s", os.path.basename(fpath), packet.pts, packet.dts, packet.time_base, packet.duration, stream.codec_context.time_base, stream.time_base, av.time_base)
            # if packet.pts is not None:
            #     packet.pts += pts
            # if packet.dts is not None:
            #     packet.dts += dts
            mux.mux(packet)
            if os.path.basename(fpath) == "1697046462_4200000.ts":
                stream = mux.streams.video[0]
                logging.info("packet after %s %s %s %s %s %s %s %s", os.path.basename(fpath), packet.pts, packet.dts, packet.time_base, packet.duration, stream.codec_context.time_base, stream.time_base, av.time_base)

    for packet in stream.encode():
        # if packet.pts is not None:
        #     packet.pts += pts
        # if packet.dts is not None:
        #     packet.dts += dts
        # logging.info("%s %s %s %s %s", fpath, pts, dts, packet.pts, packet.dts)
        mux.mux(packet)
    mux.close()


def handleVideo(trackVidPath, srcVid, handler):
    rMux = av.open(srcVid)
    rMux.streams.video[0].thread_type = "AUTO"
    rStream = rMux.streams.video[0]

    rate = rStream.average_rate
    time_base = rStream.time_base
    pts = 0
    dts = 0
    height = rStream.height
    width = rStream.width
    imgs = []

    ptsSet = False
    for frame in rMux.decode(rStream):
        ts = [{"t": time.perf_counter()}]

        if not ptsSet:
            ptsSet = True
            pts = frame.pts
            dts = frame.dts
        img = frame.to_rgb().to_ndarray()
        out = handler.h(img, ts)

        imgs.append(Frame(out.track, frame.time_base, frame.pts, frame.dts))

    rMux.close()

    writeFn = lambda: None
    if trackVidPath != "":
        writeFn = lambda: writeVideo(trackVidPath, rate, time_base, pts, dts, height, width, imgs)
    return writeFn


def process(differ, handler):
    newLines, err = differ.refresh()
    if err:
        return None, -1, False, err
    logging.info("newLines %s", newLines)
    if len(newLines) == 0:
        return None, 5, False, None
    last = newLines[len(newLines)-1]

    isEOF = False
    if last.tag == "EXT-X-ENDLIST" or last.tag == "EXT-X-DISCONTINUITY":
        isEOF = True
    writeFn = lambda: None
    if last.tag == "":
        sgm = last.b
        srcDir = os.path.dirname(differ.srcIndexPath)
        trackDir = os.path.dirname(differ.trackIndexPath)
        srcVidPath = os.path.join(srcDir, sgm)
        trackVidPath = os.path.join(trackDir, sgm)
        writeFn = handleVideo(trackVidPath, srcVidPath, handler)

    for l in newLines:
        differ.trackIndex.append(l)
    err = differ.save()
    if err:
        return None, -1, False, err

    return writeFn, 0, isEOF, None


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

    differ = Differ(cfg["TrackIndex"], cfg["Src"])
    height, width, err = getImgSize(cfg["Src"])
    if err:
        return err

    handler = Handler(cfg, height, width)

    # Warmup the tracker, so that it does not count objects in the first frame as new objects.
    warmup, err = differ.getWarmup()
    if err:
        return err
    logging.info("warmup \"%s\"", warmup)
    if warmup != "":
        handleVideo("", warmup, handler)
    if HAS_AI:
        handler.tracker.warmup = len(tracker.ids)

    threads = []
    while True:
        writeFn, waitSecs, isEOF, err = process(differ, handler)
        thrd = threading.Thread(target=writeFn)
        thrd.start()
        threads.append(thrd)
        if err:
            return err
        if isEOF:
            break

        stdin = readline(waitSecs)
        if len(stdin) > 0:
            break

    for t in threads:
        t.join()

    return None


if __name__ == "__main__":
        main()
