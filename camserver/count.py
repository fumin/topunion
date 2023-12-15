# Example usage:
# python count.py -dst=out.ts -src=testing/shilin20230826_sd.mp4 -c='{"Height": 480, "Width": 640, "Device": "cuda:0", "Mask": {"Enable": true, "X": 0, "Y": 160, "Width": 640,"Height": 70, "Shift": 5}, "Yolo": {"Weights": "yolo_best.pt", "Size": 640}}'

import argparse
import datetime
# import http
# from http import server as httpserver
import inspect
import json
import logging
# import urllib
# import threading
import traceback

import av
import cv2
import numpy as np
import numpy.typing as npt
import torch
import torchvision
import ultralytics
from ultralytics import trackers as ultralytics_trackers

# import util


class Instances:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.pred_classes = np.empty([0])
        self.pred_boxes = np.empty([0, 0])
        self.scores = np.empty([0])

        self.frameObjs = -1
        self.objs = -1

    def plot(self, im):
        text_scale = 1
        text_thickness = 1
        line_thickness = 3

        for i, box in enumerate(self.pred_boxes):
            intbox = tuple(map(int, (box[0], box[1], box[2], box[3])))
            obj_id = int(self.pred_classes[i])
            id_text = '{} {}'.format(int(obj_id), int(self.scores[i]*100))
            color = self._get_color(abs(obj_id))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0),
                        thickness=text_thickness)

        msg = f"{self.frameObjs} {self.objs}"
        msg = ""
        text_scale *= 3
        cv2.putText(im, msg,
                    (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0), thickness=2)

        return im

    def _get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color


class AIFrameOutput:
    def __init__(self, masked: npt.NDArray[np.uint8], tracked: Instances):
        self.masked = masked
        self.tracked = tracked


class Frame:
    def __init__(self, time_base: float, pts: float, dts: float, aiout: AIFrameOutput):
        self.time_base = time_base
        self.pts = pts
        self.dts = dts
        self.aiout = aiout


class Packet:
    def __init__(self, time_base, pts, dts):
        self.time_base = time_base
        self.pts = pts
        self.dts = dts
        self.frames: list[Frame] = []


class Video:
    def __init__(self, time_base, rate, bit_rate, height, width):
        self.time_base = time_base
        self.rate = rate
        self.bit_rate = bit_rate
        self.height = height
        self.width = width
        self.packets: list[Packet] = []

        self.pIdx = 0
        self.fIdx = 0

    def seek(self, t):
        while True:
            if self.pIdx >= len(self.packets):
                return None, False
            if self.fIdx >= len(self.packets[self.pIdx].frames):
                self._incr()
                continue

            frm = self.packets[self.pIdx].frames[self.fIdx]
            if frm.pts >= t:
                return frm, True

            self._incr()

    def _incr(self):
       self.fIdx += 1
       if self.fIdx >= len(self.packets[self.pIdx].frames):
           self.fIdx = 0
           self.pIdx += 1


class AIOutput:
    def __init__(self):
        self.Passed = -1


class AI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.masker = newMasker(cfg["Mask"], cfg["Height"], cfg["Width"], cfg["Yolo"]["Size"], cfg["Device"])
        # Initialize Yolo.
        numChannels = 3
        img = torch.from_numpy(np.zeros([cfg["Height"], cfg["Width"], numChannels], dtype=np.uint8)).to(cfg["Device"])
        _, masked, _ = self.masker.run(img)
        maskedShape = [masked.shape[0], masked.shape[1]]
        self.yolo = Yolo(cfg["Yolo"], cfg["Device"], [maskedShape])

    def run(self, dst: str, src: str) -> (AIOutput, str):
        tracker = Tracker()

        # Read video.
        rMux = av.open(src)
        rMux.streams.video[0].thread_type = "AUTO"
        rStream = rMux.streams.video[0]

        # Write video.
        wMux = av.open(dst, mode="w", format="mpegts")
        wStream = wMux.add_stream("h264", rate=rStream.average_rate)
        wStream.time_base = rStream.time_base
        wStream.height = self.masker.mask.shape[0]
        wStream.width = self.masker.mask.shape[1]
        wStream.bit_rate = int(rMux.bit_rate * (wStream.height*wStream.width) / (rStream.height*rStream.width))
        wStream.bit_rate = min(wStream.bit_rate, 2048*1024)

        interval = 1 / wStream.time_base / wStream.codec_context.framerate
        pts = -1
        for packet in rMux.demux(rStream):
            for frame in packet.decode():
                if frame.pts < pts:
                    continue
                img = frame.to_rgb().to_ndarray()
                aiout = self._analyzeImg(tracker, img)
                tracked = aiout.tracked.plot(aiout.masked.cpu().numpy())

                wFrame = av.VideoFrame.from_ndarray(tracked, format="rgb24")
                wFrame.pts = int(pts)
                for pkt in wStream.encode(wFrame):
                    wMux.mux(pkt)

                pts += interval

        rMux.close()
        for pkt in wStream.encode():
            wMux.mux(pkt)
        wMux.close()

        out = AIOutput()
        out.Passed = tracker.passed()
        return out, ""

    def _analyzeImg(self, tracker, im):
        with torch.no_grad():
            img = torch.from_numpy(im).to(self.cfg["Device"])
            cropped, masked1, maskedViz = self.masker.run(img)
            masked = [masked1]

            yoloOut = self.yolo.run(masked)
            tracked = tracker.run(yoloOut[0])
            aiout = AIFrameOutput(maskedViz, tracked)
            return aiout


class Tracker:
    def __init__(self):
        arg = argparse.Namespace()
        # Detections over high_thresh are considered high priority.
        arg.track_high_thresh = 0.75
        # Detections lower than low_thresh are ignored.
        arg.track_low_thresh = 0.1
        # 1 - match_thresh is the minimum IOU.
        arg.match_thresh = 0.99
        # Threshold to activate a new track.
        arg.new_track_thresh = arg.track_high_thresh + 0.1
        # Objects before track_buffer are considered lost.
        arg.track_buffer = 30
        self.tracker = ultralytics_trackers.BYTETracker(arg)

        self.frameObjs = {}
        self.objs = {}

    def reset(self):
        self.tracker.reset()
        self.frameObjs = {}
        self.objs = {}

    def run(self, outputs: dict[str, Instances]):
        instances = outputs["instances"]

        out = Instances(instances.height, instances.width)
        numObjs = len(instances.pred_classes)
        if numObjs == 0:
            return out

        tin = argparse.Namespace()
        tin.conf = instances.scores
        tin.xyxy = instances.pred_boxes
        tin.cls = instances.pred_classes

        tout = self.tracker.update(tin)

        self.frameObjs = {}
        for i in range(tout.shape[0]):
            trackID = tout[i][4]
            box = tout[i][:4]
            self.frameObjs[trackID] = box
            self.objs[trackID] = box

        if tout.shape[0] > 0:
            out.pred_classes = tout[:, 4]
            out.pred_boxes = tout[:, :4]
            out.scores = tout[:, 5]
        out.frameObjs = len(self.frameObjs)
        out.objs = len(self.objs)
        return out

    def passed(self):
        n = 0
        for tid, _ in self.objs.items():
            if tid in self.frameObjs:
                continue
            n += 1
        return n


class Masker:
    def __init__(self):
        self.config = {}
        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.mask = None
        self.vizMask = None

    def run(self, img):
        if not self.config["Enable"]:
            return img, img, img

        cropped = img[self.y1:self.y2, self.x1:self.x2]
        masked = cropped * self.mask
        maskedViz = (cropped.float() * self.vizMask).to(torch.uint8)
        return cropped, masked, maskedViz


def newMasker(config, imgH, imgW, minW, device):
    m = Masker()
    m.config = config

    h = config["Height"] + abs(config["Shift"])
    w = int(max(config["Width"], h, minW))
    centerX = int(config["X"] + config["Width"]/2)
    centerY = int(config["Y"] + min(0, config["Shift"]) + h/2)

    m.x1 = centerX - int(w/2)
    m.y1 = centerY - int(w/2)
    m.x2 = m.x1 + w
    m.y2 = m.y1 + w

    m.x1 = min(max(m.x1, 0), imgW)
    m.y1 = min(max(m.y1, 0), imgH)
    m.x2 = min(max(m.x2, 0), imgW)
    m.y2 = min(max(m.y2, 0), imgH)

    croppedH, croppedW = m.y2-m.y1, m.x2-m.x1
    numChannels = 3
    mask = np.zeros([croppedH, croppedW, numChannels], dtype=np.uint8)
    slope = config["Shift"] / croppedW
    for x in range(croppedW):
        yS = config["Y"] + int(x*slope)
        yE = yS + config["Height"]
        mask[yS:yE, x] = 1
    m.mask = torch.from_numpy(mask).to(device)

    m.vizMask = torch.clip(m.mask.float() + 0.25, 0, 1)

    return m


class Yolo:
    def __init__(self, cfg, device, shapes):
        self.yolo = ultralytics.YOLO(cfg["Weights"])
        self.yoloSize = cfg["Size"]
        # Coerce yolo to setup itself by running predict once.
        bchw = torch.zeros([len(shapes), 3, self.yoloSize, self.yoloSize], dtype=float)
        bchw = bchw.to(device)
        half = False
        if device.startswith("cuda"):
            half = True
        self.yolo.predict(bchw, half=half, verbose=False)

        self.functionals = []
        for shape in shapes:
            h, w = shape
            if w > h:
                resizedH, resizedW = int(self.yoloSize*h/w), self.yoloSize
                if (resizedH % 2) != 0:
                    resizedH -= 1
                resize = torchvision.transforms.Resize([resizedH, resizedW], antialias=True)
                padWidth = int((self.yoloSize-resizedW)/2)
                padHeight = int((self.yoloSize-resizedH)/2)
                pad = torchvision.transforms.Pad([padWidth, padHeight])
            else:
                resizedH, resizedW = self.yoloSize, int(self.yoloSize*w/h)
                if (resizedW % 2) != 0:
                    resizedW -= 1
                resize = torchvision.transforms.Resize([resizedH, resizedW], antialias=True)
                padWidth = int((self.yoloSize-resizedW)/2)
                padHeight = int((self.yoloSize-resizedH)/2)
                pad = torchvision.transforms.Pad([padWidth, padHeight])
            fn = argparse.Namespace()
            fn.h, fn.w = h, w
            fn.resize = resize
            fn.padHeight = padHeight
            fn.padWidth = padWidth
            fn.pad = pad
            self.functionals.append(fn)

    def run(self, imgs):
        chws = []
        for i, img in enumerate(imgs):
            hwc = img
            hwc = hwc.float()
            chw = torch.permute(hwc, [2, 0, 1])
            chw = chw / 255
            chw = torch.clamp(chw, min=0, max=0.9999)
            fn = self.functionals[i]
            chw = fn.resize(chw)
            chw = fn.pad(chw)
            chws.append(chw)
        bchw = torch.stack(chws, dim=0)

        # Run torch model.
        # We don't follow the usual Yolo usage since it is not thread safe.
        model = self.yolo.predictor.model
        if model.fp16:
            bchw = bchw.half()
        preds = model(bchw)
        results = self.yolo.predictor.postprocess(preds, bchw, bchw)

        batch = []
        for i, result in enumerate(results):
            fn = self.functionals[i]

            xyxy = result.boxes.xyxy.cpu().numpy()
            if fn.w > fn.h:
                xyxy[:, 1] -= fn.padHeight
                xyxy[:, 3] -= fn.padHeight
                xyxy *= (fn.w / self.yoloSize)
            else:
                xyxy[:, 0] -= fn.padWidth
                xyxy[:, 2] -= fn.padWidth
                xyxy *= (fn.h / self.yoloSize)

            instances = Instances(fn.h, fn.w)
            instances.pred_classes = result.boxes.cls.cpu().numpy()
            instances.pred_boxes = xyxy
            instances.scores = result.boxes.conf.cpu().numpy()

            o = {"instances": instances}
            batch.append(o)
        return batch


# def Multipart(handler: httpserver.BaseHTTPRequestHandler):
#     forms, files = util.ReadMultipart(handler)
#     logging.info("req %s", forms["myname"])
#     logging.info("req %s", files["f"].filename)
#     logging.info("req %s", files["f"].file)
# 
# 
# def Analyze(handler: httpserver.BaseHTTPRequestHandler, ai: AI):
#     parsed = urllib.parse.urlparse(handler.path)
#     v = urllib.parse.parse_qs(parsed.query)
#     dst = v.get("dst", [""])[0]
#     src = v.get("src", [""])[0]
# 
#     try:
#         aiout, err = ai.run(dst, src)
#     except Exception as e:
#         aiout, err = AIOutput(), traceback.format_exc()
#     if err:
#         util.HTTPRespJ(handler, 400, {"Error": ("%s %s" % (v, err))})
#         return
# 
#     util.HTTPRespJ(handler, 200, vars(aiout))
# 
# 
# class MyServerHandler(httpserver.BaseHTTPRequestHandler):
#     def do_POST(self):
#         if self.path.startswith("/Quit"):
#             threading.Thread(target=self.server.shutdown).start()
#         elif self.path.startswith("/Analyze"):
#             Analyze(self, self.server.ai)
#         elif self.path.startswith("/Multipart"):
#             Multipart(self)
#         else:
#             util.HTTPRespJ(self, 200, {"hello": "world"})
# 
#     def log_message(self, format, *args):
#         return
# 
# 
# class MyServer(httpserver.ThreadingHTTPServer):
#     def __init__(self, server_address, RequestHandlerClass, cfg):
#         super().__init__(server_address, RequestHandlerClass)
#         self.ai = AI(cfg)


class StructuredMessage:
    def __init__(self, event, /, **kwargs):
        stack = inspect.stack()
        self.longfile = stack[1][1] + ":" + str(stack[1][2])
        self.event = event
        self.kwargs = kwargs

    def __str__(self):
        self.kwargs["Ltime"] = datetime.datetime.utcnow().isoformat()+"Z"
        self.kwargs["Llongfile"] = self.longfile
        self.kwargs["Levent"] = self.event
        return json.dumps(self.kwargs)

_l = StructuredMessage


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c")
    parser.add_argument("-dst")
    parser.add_argument("-src")
    args = parser.parse_args()

    err = mainWithErr(args)
    if err:
        logging.fatal("%s", err)


def mainWithErr(args):
    cfg = json.loads(args.c)
    logging.info(_l("config", **cfg))

    dst = args.dst
    src = args.src
    ai = AI(cfg)
    try:
        aiout, err = ai.run(dst, src)
    except Exception as e:
        aiout, err = AIOutput(), traceback.format_exc()
    if err:
        print(json.dumps({"Status": 400, "Body": {"Error": ("%s %s" % (v, err))}}))
        return

    print(json.dumps({"Status": 200, "Body": vars(aiout)}))

    # Since both pyav and ultralytics don't support multithreading, don't use HTTP.
    # Plus, there are serious memory leaks.
    #
    # httpd = MyServer(("localhost", 0), MyServerHandler, cfg)
    # port = httpd.server_address[1]
    # host = "http://localhost:%d" % port
    # logging.info(_l("host", Host=host))
    # httpd.serve_forever()
    # return None


if __name__ == "__main__":
        main()
