import argparse
import datetime
import logging
import json
import inspect
import os
import queue
import subprocess
import threading
import time

import cv2
import matplotlib as plt
import numpy as np
import torch
import torchvision

import detectron2
from detectron2 import model_zoo
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances

import ultralytics

# BYTETracker needs this numpy fix...
np.float = float
from yolox.tracker.byte_tracker import BYTETracker


class Handler:
    def __init__(self, saveDir, maskers, predictor, trackers):
        self.disker = Disker(saveDir)
        self.maskers = maskers
        self.predictor = predictor
        self.trackers = trackers
        self.isShow = True

        self.pauseOnNext = False

        self.debug = ""

    def h(self, frames, ts):
        for i, f in enumerate(frames):
            img = torch.from_numpy(f.img).cuda()
            masker = self.maskers[i]
            f.cropped, f.masked, f.maskedViz = masker.run(img, ts)
        masked = [f.masked for f in frames]
        ts.append({"name": "masker", "t": time.perf_counter()})
    
        outputs_batch = self.predictor.predict(self.predictor, masked, ts)
        ts.append({"name": "predictor", "t": time.perf_counter()})
    
        for i, outputs in enumerate(outputs_batch):
            instances = outputs["instances"]
            detBoxes = instances.pred_boxes.tensor.cpu()
            detBoxesLTWH = np.copy(detBoxes)
            detBoxesLTWH[:, 2:] -= detBoxesLTWH[:, :2]
            detIDs = np.zeros([detBoxes.shape[0]], dtype=np.int32)
            detScores = instances.scores.cpu()
            predImg = plot(frames[i].maskedViz.cpu(), detBoxesLTWH, detIDs, detScores, "")
    
            if self.debug != "":
                base = os.path.basename(frames[i].fpath)
                outPath = os.path.join(self.debug, f"det{i}", base)
                cv2.imwrite(outPath, predImg)
    
            if self.isShow:
                predImg = cv2.resize(predImg, (0,0), fx=0.5, fy=0.5)
                if self.pauseOnNext:
                    predImg = self._addPause(predImg)
                cv2.imshow(f"frame{i}", predImg)
        ts.append({"name": "visualize", "t": time.perf_counter()})
    
        for i, tracker in enumerate(self.trackers):
            outputs = outputs_batch[i]
            im = frames[i].maskedViz.cpu()
            trackPredImg = track(tracker, outputs, im)

            self.disker.add(f"track{i}", trackPredImg)
    
            if self.debug != "":
                base = os.path.basename(frames[i].fpath)
                outPath = os.path.join(self.debug, f"track{i}", base)
                cv2.imwrite(outPath, trackPredImg)
    
            if self.isShow:
                trackPredImg = cv2.resize(trackPredImg, (0,0), fx=0.5, fy=0.5)
                if self.pauseOnNext:
                    trackPredImg = self._addPause(trackPredImg)
                cv2.imshow(f"track{i}", trackPredImg)
        ts.append({"name": "track", "t": time.perf_counter()})

        if self.pauseOnNext:
            self.pauseOnNext = False
            self.isShow = False

        for i, frm in enumerate(frames):
            self.disker.add(f"masked{i}", frm.masked.cpu().numpy())

        if self.debug:
            for i, frm in enumerate(frames):
                base = os.path.basename(frm.fpath)
                outPath = os.path.join(self.debug, f"masked{i}", base)
                cv2.imwrite(outPath, frm.masked.cpu().numpy())

        self.disker.save()
        ts.append({"name": "save", "t": time.perf_counter()})
    
        durs = []
        for i, t in enumerate(ts[:len(ts)-1]):
            nextT = ts[i+1]
            durs.append({"name": nextT["name"], "t": nextT["t"] - t["t"]})
        # logging.info("%s", durs)
    
        output = outputs_batch[0]
        output["fpath"] = frames[0].fpath
        return output

    def togglePause(self):
        if self.isShow:
            self.pauseOnNext = True
        else:
            self.isShow = True

    def setDebug(self, dPath, frames):
        self.debug = dPath

        for i, _ in enumerate(frames):
            dirs = [
                os.path.join(self.debug, f"masked{i}"),
                os.path.join(self.debug, f"det{i}"),
                os.path.join(self.debug, f"track{i}"),
                ]
            for d in dirs:
                os.makedirs(d, exist_ok=True)

    def _addPause(self, img):
        h, w = img.shape[0], img.shape[1]
        cv2.putText(img, "Paused", (int(w/8), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), thickness=5)
        return img


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
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

    text_scale *= 3
    cv2.putText(im, msg,
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    return im


class Disker:
    def __init__(self, dir):
        self.dir = dir
        self.frames = []

    def add(self, name, img):
        f = argparse.Namespace()
        f.name = name
        f.img = img
        self.frames.append(f)

    def save(self):
        t = datetime.datetime.now(datetime.timezone.utc)
        base = t.strftime("%Y%m%d_%H%M%S_") + str(int(t.microsecond/1000)).zfill(3)
        base += ".jpg"

        for f in self.frames:
            fdir = os.path.join(self.dir, f.name, t.strftime("%Y%m%d"), t.strftime("%H"), t.strftime("%M"))
            os.makedirs(fdir, exist_ok=True)

            fpath = os.path.join(fdir, base)
            cv2.imwrite(fpath, f.img)

        self.frames = []


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
    def __init__(self, config, frame):
        self.x1 = -1
        if config["crop"]["x"] < 0:
            return

        imgH, imgW = frame.img.shape[:2]
        self.x1 = config["crop"]["x"]
        self.y1 = config["crop"]["y"]
        self.x2 = self.x1 + config["crop"]["w"]
        if self.x2 > imgW:
            self.x2 = imgW
        self.y2 = self.y1 + config["crop"]["w"]
        if self.y2 > imgH:
            self.y2 = imgH

        croppedH, croppedW = self.y2-self.y1, self.x2 - self.x1
        numChannels = frame.img.shape[2]
        mask = np.zeros([croppedH, croppedW, numChannels], dtype=frame.img.dtype)
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


def newYolov8(weights, yoloSize, frames):
    model = ultralytics.YOLO(weights)

    myPredictor = argparse.Namespace()
    myPredictor.model = model

    functionals = []
    for frm in frames:
        h, w = frm.masked.shape[:2]
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
            # BGR to RGB.
            chw[0], chw[2] = chw[2], chw[0]
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


def newDetectron2(metadata, modelWeights, device, frame):
  cfg = detectron2.config.get_cfg()
  cfg.MODEL.DEVICE = device
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
  cfg.MODEL.WEIGHTS = modelWeights
  cfg.TEST.DETECTIONS_PER_IMAGE = 128
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
  predictor = DefaultPredictor(cfg)

  predictor.model = predictor.model.half()

  myPredictor = argparse.Namespace()
  myPredictor.p = predictor
  myPredictor.transform = predictor.aug.get_transform(frame)
  def predictFn(predictor, frames, ts):
      with torch.no_grad():
          if predictor.p.input_format == "RGB":
              frames = frames[:, :, :, ::-1]
          height, width = frames.shape[1], frames.shape[2]
          imgs = []
          for i, fm in enumerate(frames):
              img = predictor.transform.apply_image(fm)
              imgs.append(img)
          imgs = np.stack(imgs)
          imgs = torch.as_tensor(imgs.astype("float32").transpose(0, 3, 1, 2))
          if next(predictor.p.model.parameters()).dtype == torch.half:
              imgs = imgs.half()
          ts.append({"name": "transform", "t": time.perf_counter()})

          inputs = []
          for i, fm in enumerate(imgs):
              inpt = {"image": fm, "height": height, "width": width}
              inputs.append(inpt)
          predictions = predictor.p.model(inputs)
          return predictions
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


class ViaRecorder():
    def __init__(self):
        self.outputs = []
    def add(self, outputs):
        self.outputs.append(outputs)
    def save(self, outPath, inPath):
        with open(inPath) as f:
            viaSetting = json.load(f)

        metadata = {}
        imgList = []
        for o in self.outputs:
            fpath = o["fpath"]
            base = os.path.basename(fpath)
            filesize = os.stat(fpath).st_size

            boxes = o["instances"].pred_boxes.tensor.numpy()
            regions = []
            for box in boxes:
                shapeAttr = {}
                shapeAttr["name"] = "rect"
                shapeAttr["x"] = float(box[0])
                shapeAttr["y"] = float(box[1])
                shapeAttr["width"] = float(box[2] - box[0])
                shapeAttr["height"] = float(box[3] - box[1])

                rg = {}
                rg["shape_attributes"] = shapeAttr
                rg["region_attributes"] = {}
                regions.append(rg)

            metadata = {}
            metadata["filename"] = base
            metadata["size"] = filesize
            metadata["file_attributes"] = {}
            metadata["regions"] = regions

            k = base + str(filesize)
            if k in viaSetting["_via_img_metadata"]:
                continue
            viaSetting["_via_img_metadata"][k] = metadata
            viaSetting["_via_image_id_list"].append(k)

        with open(outPath, "w") as f:
            f.write(json.dumps(viaSetting))


class DummyVia:
    def __init__(self):
        pass
    def add(self, outputs):
        pass
    def save(self, outPath, inPath):
        pass


class UVCCam:
    def __init__(self, port):
        self.cap, self.err = self._newCap(port)

    def grab(self):
        ok = self.cap.grab()
        if not ok:
            return f"not ok {inspect.getframeinfo(inspect.currentframe())}"
        return None

    def retrieve(self):
        ok, frame = self.cap.retrieve()
        if not ok:
            return None, f"not ok {inspect.getframeinfo(inspect.currentframe())}"

        ret = argparse.Namespace()
        ret.eof = False
        ret.img = frame
        ret.fpath = f"{int(time.time() * 1000)}.jpg"
        return ret, None

    def close(self):
        self.cap.release()

    def _newCap(self, port):
        self.port = port
        os.system(f"""sh -c "echo 0 > /sys/bus/usb/devices/{port}/authorized" """)
        os.system(f"""sh -c "echo 1 > /sys/bus/usb/devices/{port}/authorized" """)
        videos = self.usbVideos(port)
        if len(videos) == 0:
            return None, f"no videos {port} {inspect.getframeinfo(inspect.currentframe())}"
        self.videoID = videos[0]
        cap = cv2.VideoCapture(self.videoID)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 120)
        if not cap.isOpened():
            return None, f"not opened 0 {videos} {inspect.getframeinfo(inspect.currentframe())}"

        return cap, None

    def usbVideos(self, port):
        for i in range(20):
            videos = self.usbVideosOnce(port)
            if len(videos) > 0:
                return videos
    
            time.sleep(0.1)
        return []
    
    def usbVideosOnce(self, port):
        sysDir = "/sys/bus/usb/devices"
        portDirs = []
        sysDirSubs = os.listdir(sysDir)
        sysDirSubs.sort()
        for d in sysDirSubs:
            if d.startswith(port):
                portDirs.append(os.path.join(sysDir, d))
        hasVideo = []
        for d in portDirs:
            vidDir = os.path.join(d, "video4linux")
            if os.path.isdir(vidDir):
                hasVideo.append(vidDir)
    
        videos = []
        for d in hasVideo:
            for vd in os.listdir(d):
                vID = int(vd[len(vd)-1])
                videos.append(vID)
        videos.sort()
    
        return videos

    def __str__(self):
        return f"UVCCam port: {self.port}, videoID: {self.videoID}"


class RTSPCam:
    def __init__(self, interface, addr, username, password, port, urlPath):
        self.interface = interface
        self.addr = addr
        self.username = username
        self.password = password
        self.port = port
        self.urlPath = urlPath

        self.err = None
        self.link = ""
        self.lock = threading.Lock()
        self.kill = False
        self.ok = False
        self.frame = None

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        try:
            self.err = self.queue.get(True, 15)
        except queue.Empty as e:
            self.err = f"queue empty {inspect.getframeinfo(inspect.currentframe())}"

    def grab(self):
        return None

    def retrieve(self):
        with self.lock:
            if not self.ok:
                return None, f"not ok link=\"{self.link}\" {inspect.getframeinfo(inspect.currentframe())}"
            frame = self.frame.copy()

        ret = argparse.Namespace()
        ret.eof = False
        ret.img = frame
        ret.fpath = f"{int(time.time() * 1000)}.jpg"
        return ret, None

    def close(self):
        with self.lock:
            self.kill = True
        self.thread.join()

    def _run(self):
        self.link, err = self._getLink(self.addr)
        if err:
            self.queue.put(err, timeout=1)
            return
        cap = cv2.VideoCapture(self.link, apiPreference=cv2.CAP_FFMPEG)
        for i in range(10):
            with self.lock:
                self.ok, self.frame = cap.read()
                if self.ok:
                    break
            time.sleep(1)
        with self.lock:
            if not self.ok:
                self.queue.put(f"not ok {self.link} {inspect.getframeinfo(inspect.currentframe())}", timeout=1)
                cap.release()
                return
            else:
                self.queue.put(None, timeout=1)

        while True:
            with self.lock:
                if self.kill:
                    break

            cap.grab()
            with self.lock:
                self.ok, self.frame = cap.retrieve()

        cap.release()

    def _getLink(self, macAddr):
        arps, err = self._getARP()
        if err:
            return "", err
        if macAddr not in arps:
            return "", f"{macAddr} {arps} {inspect.getframeinfo(inspect.currentframe())}"

        ip = arps[macAddr].ip
        link = f"rtsp://{self.username}:{self.password}@{ip}:{self.port}{self.urlPath}"
        return link, None

    def _getARP(self):
        outStr = subprocess.run(["arp-scan", f"--interface={self.interface}", "-l", "-x"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = outStr.split("\n")
        # Skip header.
        # lines = lines[1:]
        # Skip last empty line.
        lines = lines[:len(lines)-1]

        arps = {}
        for line in lines:
            records = line.split("\t")
            records = list(filter(None, records))
            if len(records) != 3:
                return None, f"{records} {inspect.getframeinfo(inspect.currentframe())}"
            arp = argparse.Namespace()
            arp.ip = records[0]
            arp.hw = records[1]
            arps[arp.hw] = arp
        return arps, None


class Video:
    def __init__(self, fpath):
        self.cap = cv2.VideoCapture(fpath)
        self.i = -1
        self.err = None

    def grab(self):
        return None

    def retrieve(self):
        ret = argparse.Namespace()
        self.i += 1
        ok, frm = self.cap.read()
        if not ok:
            ret.eof = True
            return ret, None

        ret.eof = False
        ret.img = frm
        ret.fpath = f"{self.i}.jpg"
        return ret, None

    def close(self):
        self.cap.release()


class ImgDir:
    def __init__(self, imgdir):
        bases = os.listdir(imgdir)
        basets = []
        for base in bases:
            noext = os.path.splitext(base)[0]
            t = float(noext)

            bt = argparse.Namespace()
            bt.base = base
            bt.t = t
            basets.append(bt)
        basets.sort(key=lambda bt: bt.t)

        self.err = None
        self.dir = imgdir
        self.bases = [bt.base for bt in basets]
        self.i = -1

    def grab(self):
        return None

    def retrieve(self):
        ret = argparse.Namespace()
        self.i += 1
        if self.i >= len(self.bases):
            ret.eof = True
            return ret, None

        base = self.bases[self.i]
        fpath = os.path.join(self.dir, base)
        frm = cv2.imread(fpath)

        ret.eof = False
        ret.img = frm
        ret.fpath = fpath
        return ret, None

    def close(self):
        pass


class MultiCam:
    def __init__(self, config):
        self.config = config
        self.cams, self.err = self._newCameras(self.config)

    def get(self):
        while True:
            frames, err = self._getOnce(self.cams)
            if not err:
                return frames

            logging.info("%s", err)
            while True:
                for c in self.cams:
                    c.close()
                self.cams, err = self._newCameras(self.config)
                if not err:
                    break

                logging.info("%s", err)
                time.sleep(1)

    def close(self):
        for cam in self.cams:
            cam.close()

    def _getOnce(self, cams):
        for c in cams:
            err = c.grab()
            if err:
                return err

        frames = []
        for c in cams:
            frm, err = c.retrieve()
            if err:
                return None, err
            frames.append(frm)
        return frames, None

    def _newCameras(self, config):
        cams = []
        for cfg in config:
            c, err = self._newCam(cfg)
            if err:
                return [], err
            cams.append(c)
        return cams, None

    def _newCam(self, config):
        if config["type"] == "uvc":
            cam = UVCCam(config["port"])
        elif config["type"] == "rtsp":
            cam = RTSPCam(config["interface"], config["addr"], config["username"], config["password"], config["port"], config["path"])
        elif config["type"] == "video":
            cam = Video(config["path"])
        elif config["type"] == "img":
            cam = ImgDir(config["path"])
        else:
            return None, f"unknown camera type {config} {inspect.getframeinfo(inspect.currentframe())}"
        if cam.err:
            return None, cam.err
        return cam, None


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]

    camCfg = [
            # {"type": "uvc", "port": "1-1"},
            # {"type": "uvc", "port": "1-5"},

            # 喬安科技
            # {"type": "rtsp", "addr": "00:e2:1f:4a:0b:c2", "username": "admin", "password": "123456", "port": 554, "path": "/mpeg4"},

            # Redmi Note 4X
            {"type": "rtsp", "interface": "wlx00ebd8c3df3d", "addr": "4c:49:e3:3a:87:4a", "username": "admin", "password": "0000", "port": 8080, "path": "/h264_ulaw.sdp"},
            # Redmi 12C
            {"type": "rtsp", "interface": "wlx00ebd8c3df3d", "addr": "46:31:36:46:1c:29", "username": "admin", "password": "0000", "port": 8080, "path": "/h264_ulaw.sdp"},
            # {"type": "video", "path": "/home/topunion/a/count/want/motianlun1.mp4"}
            # {"type": "video", "path": "/home/topunion/a/count/want/rukou1.mp4"}
            # {"type": "img", "path": "/home/topunion/a/count/data/tianxin20230901/img"},
    ]
    camera = MultiCam(camCfg)
    if camera.err:
        raise Exception(camera.err)
    frames = camera.get()

    maskers = [
            Masker({"crop": {"x": -1}}, frames[0]),
            Masker({"crop": {"x": -1}}, frames[0]),
            # Masker({
            #     "crop": {"x": 400, "y": 0, "w": 1200},
            #     "mask": {"slope": 75, "y": 400, "h": 1000},
            #     }, frames[0]),
            # Masker({
            #     "crop": {"x": 0, "y": 0, "w": 99999},
            #     "mask": {"slope": -150, "y": 350, "h": 200},
            #     }, frames[1]),
            ]
    for i, f in enumerate(frames):
        img = torch.from_numpy(f.img).cuda()
        masker = maskers[i]
        f.cropped, f.masked, f.maskedViz = masker.run(img, [])

    metadata = Metadata()
    metadata.thing_classes = ["蛋"]
    device = "cuda"
    # predictor = newDetectron2(metadata, "detectron2_best.pth", device, frames[0])
    predictor = newYolov8("yolo_best.pt", 640, frames)

    trackers = []
    for i in range(len(frames)):
        t = newTracker()
        trackers.append(t)

    handler = Handler("nvr", maskers, predictor, trackers)
    # handler.setDebug("motianlun1", frames)

    viaRecorder = ViaRecorder()
    viaRecorder = DummyVia()

    durations = []
    while True:
        startT = time.perf_counter()

        ts = [{"t": time.perf_counter()}]
        frames = camera.get()
        hasEOF = False
        for f in frames:
            if f.eof:
                hasEOF = True
                break
        if hasEOF:
            break
        ts.append({"name": "camera", "t": time.perf_counter()})

        outputs = handler.h(frames, ts)
        viaRecorder.add(outputs)
        
        duration = time.perf_counter() - startT
        durations.append(duration)
        if len(durations) >= 20:
            logging.info("fps %f", len(durations)/sum(durations))
            durations = []
        
        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == ord('q'):
            break
        if keyboard == ord('s'):
            handler.togglePause()
    
    viaRecorder.save("/home/topunion/a/count/data/tianxin20230901/labelQQQ.json", "/home/topunion/a/count/data/tianxin20230901/label.json")
    logging.info("%d eggs", len(trackers[0].ids))
    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
