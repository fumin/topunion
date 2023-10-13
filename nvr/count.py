# Example usage:
# python count.py -c='{"TrackIndex": "track/index.m3u8", "TrackDir": "track/track", "Src": "server/record/2006/20060102/20060102_150405_000000/rtsp0/index.m3u8", "Device": "cpu", "Mask": {"Enable": false}, "Yolo": {"Weights": "yolo_best.pt", "Size": 640}, "Track": {}}'

import argparse
import datetime
import inspect
import json
import logging
import os
import queue
import select
import sys
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt
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


class HandleResult:
    def __init__(self, count, track: npt.NDArray[np.uint8]):
        self.count = count
        self.track = track
    def json(self):
        d = {"Count": self.count}
        return json.dumps(d)


def readHandleResult(fpath):
    with open(fpath) as f:
        b = f.read()
    d = json.loads(b)
    return HandleResult(d["Count"], np.zeros([], dtype=np.uint8))


class Handler:
    def __init__(self, cfg, height, width, lastResultPath):
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

        if lastResultPath != "":
            lastRes = readHandleResult(lastResultPath)
            if HAS_AI:
                self.tracker.prevCount = lastRes.count

        self.config = cfg
        self.masker = masker
        self.detector = detector
        self.tracker = tracker

    def afterWarmup(self):
        if HAS_AI:
            self.tracker.warmup = len(self.tracker.ids)

    def h(self, img: npt.NDArray[np.uint8], ts: list[Any]) -> HandleResult:
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

        return HandleResult(numCounted, trackPredImg)


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
    return t


class StdinWindows:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def readline(self, secs):
        try:
            line = self.queue.get(timeout=secs)
        except queue.Empty:
            line = ""
        return line

    def _run(self):
        while True:
            line = sys.stdin.readline()
            self.queue.put(line)


class StdinUnix:
    def __init__(self):
       pass

    def readline(self, secs):
        rlist, _, _ = select.select([sys.stdin], [], [], secs)
        if rlist:
            line = sys.stdin.readline()
        else:
            line = ""
        return line


def newStdinReader():
    if sys.platform == "win32":
        return StdinWindows()
    return StdinUnix()


class M3U8Line:
    def __init__(self):
        self.b = ""
        self.tag = ""
    def __repr__(self):
        return f"M3U8Line{{tag: {self.tag}, b: {self.b}}}"


def newM3U8Line(line: str) -> M3U8Line:
    m = M3U8Line()
    m.b = line
    if not line.startswith("#"):
        return m

    # Remove preceding #
    line = line[1:]
    colonSS = line.split(":")
    m.tag = colonSS[0]

    return m


def readIndex(fpath: str) -> tuple[list[M3U8Line], Any]:
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
    def __init__(self, trackIndexPath: str, srcIndexPath: str):
        self.trackIndexPath = trackIndexPath
        self.srcIndexPath = srcIndexPath

        self.trackIndex: list[M3U8Line] = []

    def __repr__(self):
        return f"Differ{{{self.trackIndexPath} {self.srcIndexPath} {self.trackIndex}}}"

    def init(self) -> tuple[str, str, Any]:
        trackIndex, err = self._load()
        if err:
            return "", "", err

        lastRes = self._getLastResult(trackIndex)
        warmup = self._getWarmup(trackIndex)

        # Do not load ENDLIST, as it is sure to be different after the next refresh.
        if len(trackIndex) > 0:
            last = trackIndex[len(trackIndex)-1]
            if last.tag == "EXT-X-ENDLIST":
                trackIndex = trackIndex[ : len(trackIndex)-1]
        self.trackIndex = trackIndex

        return lastRes, warmup, None

    def refresh(self) -> tuple[list[M3U8Line], Any]:
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

    def _load(self) -> tuple[list[M3U8Line], Any]:
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

    def _getLastResult(self, trackIndex: list[M3U8Line]) -> str:
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

        lastRes = ""
        if urlIdx >= 0:
            base = trackIndex[urlIdx].b
            noext, _ = os.path.splitext(base)
            lastRes = noext
        return lastRes

    def _getWarmup(self, trackIndex: list[M3U8Line]) -> str:
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
            urlStr = os.path.join(os.path.dirname(self.srcIndexPath), base)
        return urlStr


def saveIndex(fpath: str, index: list[M3U8Line]):
    b = "\n".join([m.b for m in index])
    with open(fpath, "w") as f:
        f.write(b)


def getImgSize(indexPath: str) -> tuple[int, int, Any]:
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


class VideoInfo:
    def __init__(self):
        self.fpath: str = ""
        self.rate: Any = -1
        self.bit_rate: int = -1
        self.height: int = -1
        self.width: int = -1
        self.frames: list[Frame] = []

    def __repr__(self):
        return f"VideoInfo {self.fpath} {len(self.frames)}"


def writeVideo(info: VideoInfo):
    mux = av.open(info.fpath, mode="w", format="mpegts")
    stream = mux.add_stream("h264", rate=info.rate)
    stream.bit_rate = int(info.bit_rate)
    stream.height = info.height
    stream.width = info.width

    for frm in info.frames:
        frame = av.VideoFrame.from_ndarray(frm.img, format="rgb24")
        frame.time_base = frm.time_base
        frame.pts = frm.pts
        for packet in stream.encode(frame):
            mux.mux(packet)

    for packet in stream.encode():
        mux.mux(packet)
    mux.close()


def handleVideo(trackVidPath: str, srcVid: str, handler: Any) -> tuple[VideoInfo, list[HandleResult]]:
    rMux = av.open(srcVid)
    rMux.streams.video[0].thread_type = "AUTO"
    rStream = rMux.streams.video[0]

    info = VideoInfo()
    info.fpath = trackVidPath
    info.rate = rStream.average_rate
    info.height = rStream.height
    info.width = rStream.width

    bits = 0
    secs = 0

    handleRes = []
    for packet in rMux.demux(rStream):
        bits += packet.size * 8
        secs += packet.duration * packet.time_base
        for frame in packet.decode():
            ts = [{"t": time.perf_counter()}]

            img = frame.to_rgb().to_ndarray()
            out = handler.h(img, ts)

            info.frames.append(Frame(out.track, frame.time_base, frame.pts, frame.dts))
            handleRes.append(out)

    rMux.close()

    info.bit_rate = bits / secs
    return info, handleRes


class ProcessInfo:
    def __init__(self):
        self.indexPath: str = ""
        self.indexCurrent: list[M3U8List] = []
        self.indexNew: list[M3U8List] = []
        self.video: VideoInfo = None

        self.trackPath: str = ""
        self.handleResults: list[HandleResult] = []

    def __repr__(self):
        return f"ProcessInfo {self.indexPath} {len(self.indexCurrent)} {len(self.indexNew)} {self.video}"

    def isEOF(self):
        if len(self.indexNew) == 0:
            return False
        last = self.indexNew[len(self.indexNew)-1]
        if last.tag == "EXT-X-ENDLIST" or last.tag == "EXT-X-DISCONTINUITY":
            return True
        return False

    def sleep(self):
        if len(self.indexNew) == 0:
            return 5
        return 0


class ThreadQueue:
    def __init__(self, thread: threading.Thread, qu: queue.Queue):
        self.thread = thread
        self.queue = qu

        self.mutex = threading.Lock()
        self.gotOK = False
        self.ok = False

    def wait(self):
        with self.mutex:
            if self.gotOK:
                return self.ok

            self.thread.join()
            try:
                self.ok = self.queue.get(block=False)
            except queue.Empty as e:
                self.ok = False
            self.gotOK = True
            return self.ok

    def is_alive(self):
        return self.thread.is_alive()


def runBackground(qu: queue.Queue, threads: list[ThreadQueue], info: ProcessInfo):
    if info.video:
        writeVideo(info.video)
    if info.trackPath != "":
        lastRes = info.handleResults[len(info.handleResults)-1]
        with open(info.trackPath, "w") as f:
            f.write(lastRes.json())

    for t in threads:
        if not t.wait():
            return

    logging.info("newlines %s, video %s, threads %d", info.indexNew, info.video, len(threads))
    index = info.indexCurrent
    for l in info.indexNew:
        index.append(l)
    saveIndex(info.indexPath, index)

    qu.put(True)


def process(differ: Differ, trackDir: str, handler: Any) -> tuple[ProcessInfo, Any]:
    newLines, err = differ.refresh()
    if err:
        return None, err

    info = ProcessInfo()
    info.indexPath = differ.trackIndexPath
    info.indexCurrent = [l for l in differ.trackIndex]
    info.indexNew = newLines

    for l in newLines:
        differ.trackIndex.append(l)

    if len(newLines) > 0:
        last = newLines[len(newLines)-1]
        if last.tag == "":
            sgm = last.b
            srcDir = os.path.dirname(differ.srcIndexPath)
            trackIndexDir = os.path.dirname(differ.trackIndexPath)
            srcVidPath = os.path.join(srcDir, sgm)
            trackVidPath = os.path.join(trackIndexDir, sgm)
            info.video, info.handleResults = handleVideo(trackVidPath, srcVidPath, handler)

            base = os.path.basename(srcVidPath)
            noext, _ = os.path.splitext(base)
            info.trackPath = os.path.join(trackDir, noext+".json")

    return info, None


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

    lastRes, warmup, err = differ.init()
    if err:
        return err
    lastHandleRes = ""
    if lastRes != "":
        lastHandleRes = os.path.join(cfg["TrackDir"], lastRes+".json")
    logging.info("lastHandleRes \"%s\"", lastHandleRes)
    handler = Handler(cfg, height, width, lastHandleRes)

    # Warmup the tracker, so that it does not count objects in the first frame as new objects.
    logging.info("warmup \"%s\"", warmup)
    if warmup != "":
        handleVideo("", warmup, handler)
        handler.afterWarmup()

    firstVideo = True
    stdinR = newStdinReader()
    threads = []
    while True:
        info, err = process(differ, cfg["TrackDir"], handler)
        if err:
            return err

        qu = queue.Queue(maxsize=1)
        stillRunning = list(filter(lambda t: t.is_alive(), threads))
        # For the first video, do things synchronously, so that we get a video as soon as possible.
        # If not, we may spawn too many threads, leading to resource contention and slow first video.
        if firstVideo:
            firstVideo = False
            runBackground(qu, stillRunning, info)
        else:
            thrd = threading.Thread(target=runBackground, args=(qu, stillRunning, info))
            thrd.start()
            threads = [t for t in stillRunning]
            threads.append(ThreadQueue(thrd, qu))

        if info.isEOF():
            break

        stdin = stdinR.readline(info.sleep())
        if len(stdin) > 0:
            break

    for t in threads:
        t.wait()

    return None


if __name__ == "__main__":
        main()
