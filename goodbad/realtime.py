import logging
import os
import time
import cv2
import matplotlib as plt

import detectron2
from detectron2 import model_zoo
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor


def handleFrame(predictor, metadata, im):
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predImg = out.get_image()[:, :, ::-1]

    cv2.imshow("frame", predImg)


def preparePredictor(metadata, modelWeights, device):
  cfg = detectron2.config.get_cfg()
  cfg.MODEL.DEVICE = device
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
  cfg.MODEL.WEIGHTS = modelWeights
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
  predictor = DefaultPredictor(cfg)
  return predictor


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]
    
    metadata = Metadata()
    metadata.thing_classes = ["好蛋", "壞蛋", "髒污"]
    device = "cuda"
    predictor = preparePredictor(metadata, "model_best.pth", device)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)
    if not cap.isOpened():
        raise Exception("not opened")
    
    durations = []
    while True:
        startT = time.perf_counter()
    
        ret, frame = cap.read()
        handleFrame(predictor, metadata, frame)
        
        duration = time.perf_counter() - startT
        durations.append(duration)
        if len(durations) >= 100:
            logging.info("fps %f", len(durations)/sum(durations))
            durations = []
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
