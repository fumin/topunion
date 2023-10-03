import argparse
import logging
import os
import subprocess
import time
import cv2
import matplotlib as plt
import numpy as np
import torch

import detectron2
from detectron2 import model_zoo
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor

# BYTETracker needs this numpy fix...
np.float = float
from yolox.tracker.byte_tracker import BYTETracker


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot(inImg, tlwhs, obj_ids, scores, isBad, msg):
    im = np.copy(inImg)

    text_scale = 2
    text_thickness = 2
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

        if isBad[i] != 0:
            pt1 = [intbox[0]+10, intbox[1]+10]
            pt2 = [intbox[2]-10, intbox[3]-10]
            im[intbox[1]+10:intbox[3]-10, intbox[0]+10:intbox[2]-10, 2] = 255

    cv2.putText(im, msg,
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    return im


def predict(predictor, frames, ts):
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

    pred_classes = instances.pred_classes.cpu()
    trackPredImg = plot(im, trackBoxes, trackIDs, trackScores, pred_classes, f"")
    return trackPredImg


def handleFrame(predictor, trackers, frames):
    # mask = np.zeros(frames.shape, dtype=frames.dtype)
    # mask[:, 140:1400, :135] = 1
    # frames = frames * mask

    ts = [{"t": time.perf_counter()}]

    outputs_batch = predict(predictor, frames, ts)
    ts.append({"name": "detectron2", "t": time.perf_counter()})

    for i, outputs in enumerate(outputs_batch):
        instances = outputs["instances"]
        detBoxes = instances.pred_boxes.tensor.cpu()
        detBoxesLTWH = np.copy(detBoxes)
        detBoxesLTWH[:, 2:] -= detBoxesLTWH[:, :2]
        detIDs = instances.pred_classes.cpu()
        detScores = instances.scores.cpu()
        predImg = plot(frames[i], detBoxesLTWH, detIDs, detScores, np.zeros(detIDs.shape), "")

        # if detScores.shape[0] > 1:
        #     t = time.time() * 1000
        #     cv2.imwrite(f"spurious/{t}_ori.jpg", im)
        #     cv2.imwrite(f"spurious/{t}_pred.jpg", predImg)

        predImg = cv2.resize(predImg, (0,0), fx=0.5, fy=0.5)
        cv2.imshow(f"frame{i}", predImg)
    ts.append({"name": "visualize", "t": time.perf_counter()})

    for i, tracker in enumerate(trackers):
        outputs = outputs_batch[i]
        im = frames[i]

        trackPredImg = track(tracker, outputs, im)
        trackPredImg = cv2.resize(trackPredImg, (0,0), fx=0.5, fy=0.5)
        cv2.imshow(f"track{i}", trackPredImg)
    ts.append({"name": "track", "t": time.perf_counter()})

    durs = []
    for i, t in enumerate(ts[:len(ts)-1]):
        nextT = ts[i+1]
        durs.append({"name": nextT["name"], "t": nextT["t"] - t["t"]})
    # logging.info("%s", durs)


def preparePredictor(metadata, modelWeights, device, frame):
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

  return myPredictor


def newTracker():
    trackerArg = argparse.Namespace()
    trackerArg.track_thresh = 0.5
    trackerArg.track_buffer = 30
    trackerArg.match_thresh = 0.8
    trackerArg.mot20 = False
    tracker = BYTETracker(trackerArg)

    t = argparse.Namespace()
    t.t = tracker
    return t


def usbVideos(port):
    sysDir = "/sys/bus/usb/devices"
    portDirs = []
    for d in os.listdir(sysDir):
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


def newCameras(cap0, cap1):
    if cap0:
        cap0.release()
    if cap1:
        cap1.release()

    port0 = "1-1"
    os.system(f"""sh -c "echo 0 > /sys/bus/usb/devices/{port0}/authorized" """)
    os.system(f"""sh -c "echo 1 > /sys/bus/usb/devices/{port0}/authorized" """)
    videos0 = usbVideos(port0)
    if len(videos0) == 0:
        return None, None, f"no videos {port0}"
    vid0ID = videos0[0]
    os.system(f"""v4l2-ctl --set-ctrl power_line_frequency=2 -d {vid0ID}""")
    os.system(f"""v4l2-ctl --set-ctrl backlight_compensation=0 -d {vid0ID}""")
    cap0 = cv2.VideoCapture(vid0ID)
    cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap0.set(cv2.CAP_PROP_FPS, 60)
    cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap0.set(cv2.CAP_PROP_EXPOSURE, 78)
    if not cap0.isOpened():
        return None, None, f"not opened 0 {videos0}"

    port1 = "1-5"
    os.system(f"""sh -c "echo 0 > /sys/bus/usb/devices/{port1}/authorized" """)
    os.system(f"""sh -c "echo 1 > /sys/bus/usb/devices/{port1}/authorized" """)
    videos1 = usbVideos(port1)
    if len(videos1) == 0:
        return None, None, f"no videos {port1}"
    vid1ID = videos1[0]
    os.system(f"""v4l2-ctl --set-ctrl power_line_frequency=2 -d {vid1ID}""")
    os.system(f"""v4l2-ctl --set-ctrl backlight_compensation=0 -d {vid1ID}""")
    cap1 = cv2.VideoCapture(vid1ID)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap1.set(cv2.CAP_PROP_FPS, 60)
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap1.set(cv2.CAP_PROP_EXPOSURE, 78)
    if not cap1.isOpened():
        return None, None, f"not opened 0 {videos1}"

    return cap0, cap1, None


def getFramesOnce(cap0, cap1):
    ok0 = cap0.grab()
    ok1 = cap1.grab()
    if not ok0:
        return None, "cap0 grab fail"
    if not ok1:
        return None, "cap1 grab fail"
    
    ret, frame0 = cap0.retrieve()
    if not ret:
        return None, "cap0 retrieve fail"
    ret, frame1 = cap1.retrieve()
    if not ret:
        return None, "cap1 retrieve fail"

    frames = np.stack([frame0, frame1])
    return frames, None
 

def getFrames(cap0, cap1):
    while True:
        frames, err = getFramesOnce(cap0, cap1)
        if not err:
            return cap0, cap1, frames
        logging.info("%s", err)

        while True:
            cap0, cap1, err = newCameras(cap0, cap1)
            if not err:
                break

            logging.info("%s", err)
            time.sleep(1)


class Camera:
    def __init__(self):
        self.cap0, self.cap1, self.err = newCameras(None, None)
    def get(self):
        self.cap0, self.cap1, frames = getFrames(self.cap0, self.cap1)
        return frames, False
    def close(self):
        self.cap0.release()
        self.cap1.release()


class Video:
    def __init__(self, fpath):
        self.cap = cv2.VideoCapture(fpath)
        self.err = None
    def get(self):
        ret, frm = self.cap.read()
        if not ret:
            return None, True
        frames = np.stack([frm, frm])
        return frames, False
    def close(self):
        self.cap.release()


class Dir:
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.paths = os.listdir(dirpath)
        self.i = -1
        self.err = None
    def get(self):
        self.i += 1
        if self.i >= len(self.paths):
            return None, True

        fpath = os.path.join(self.dirpath, self.paths[self.i])
        frm = cv2.imread(fpath)
        frames = np.stack([frm, frm])
        return frames, False
    def close(self):
        pass


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]

    camera = Camera()
    # camera = Video("20230721_110115.mp4")
    # camera = Dir("data/fushangsheng20230519/img")
    if camera.err:
        raise Exception(camera.err)
    frames, _ = camera.get()

    metadata = Metadata()
    metadata.thing_classes = ["好", "壞", "髒污"]
    device = "cuda"
    predictor = preparePredictor(metadata, "model_best.pth", device, frames[0])

    trackers = []
    for i in range(2):
        t = newTracker()
        trackers.append(t)

    durations = []
    while True:
        startT = time.perf_counter()

        frames, eof = camera.get()
        if eof:
            break
        handleFrame(predictor, trackers, frames)
        
        duration = time.perf_counter() - startT
        durations.append(duration)
        if len(durations) >= 20:
            logging.info("fps %f", len(durations)/sum(durations))
            durations = []
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
