import logging
import json
import os
import random

import cv2
import numpy as np
import torch

import detectron2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import BestCheckpointer
from detectron2.engine import DefaultPredictor


def getDataDicts(dataDir, imgIDStart):
  json_file = os.path.join(dataDir, "label.json")
  with open(json_file) as f:
    viaSetting = json.load(f)

  # categories = list(viaSetting["_via_attributes"]["region"]["klass"]["options"].keys())
  categories = ["egg"]
  categoryIDMap = {}
  for idx, c in enumerate(categories):
    categoryIDMap[c] = idx

  dataset_dicts = []
  for idx, k in enumerate(viaSetting["_via_image_id_list"]):
    v = viaSetting["_via_img_metadata"][k]
    
    filename = os.path.join(dataDir, "img", v["filename"])
    height, width = cv2.imread(filename).shape[:2]

    record = {}
    record["file_name"] = filename
    record["image_id"] = imgIDStart + idx
    record["height"] = height
    record["width"] = width

    objs = []
    for _, anno in enumerate(v["regions"]):
      s = anno["shape_attributes"]
      obj = {
          "bbox_mode": BoxMode.XYWH_ABS,
          "bbox": [s["x"], s["y"], s["width"], s["height"]],
          "category_id": 0  # categoryIDMap[anno["region_attributes"]["klass"]],
      }
      objs.append(obj)
    record["annotations"] = objs

    dataset_dicts.append(record)

  return categories, dataset_dicts


def registerData(eggCategories, trainDataDicts, validationDataDicts):
  try:
    DatasetCatalog.remove("eggTrain")
  except Exception as e:
    pass
  DatasetCatalog.register("eggTrain", lambda _=None: trainDataDicts)
  try:
    MetadataCatalog.remove("eggTrain")
  except Exception as e:
    pass
  MetadataCatalog.get("eggTrain").set(thing_classes=eggCategories)

  try:
    DatasetCatalog.remove("eggValidation")
  except Exception as e:
    pass
  DatasetCatalog.register("eggValidation", lambda _=None: validationDataDicts)
  try:
    MetadataCatalog.remove("eggValidation")
  except Exception as e:
    pass
  MetadataCatalog.get("eggValidation").set(thing_classes=eggCategories)


def getDataset(dataDir):
    eggCategories = []
    dataDicts = []
    imgIDStart = 0
    for i, dir in enumerate(os.listdir(dataDir)):
      eggCategories, ds = getDataDicts(os.path.join(dataDir, dir), imgIDStart)
      imgIDStart += len(ds)
      dataDicts += ds
    trainDataDicts = []
    validationDataDicts = []
    for idx, d in enumerate(dataDicts):
      if idx % 10 >= 8:
        validationDataDicts.append(d)
      else:
        trainDataDicts.append(d)
    
    registerData(eggCategories, trainDataDicts, validationDataDicts)

    return eggCategories, trainDataDicts, validationDataDicts


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
      if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
      return COCOEvaluator(dataset_name, output_dir=output_folder, max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE)


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    expDir = "exp"
    dataDir = "data"

    eggCategories, trainDataDicts, validationDataDicts = getDataset(dataDir)
    logging.info("train: %d, validation: %d", len(trainDataDicts), len(validationDataDicts))

    datasampleDir = os.path.join(expDir, "datasample")
    os.makedirs(datasampleDir, exist_ok=True)
    for i, d in enumerate(random.sample(trainDataDicts, 10)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get("eggTrain"),
            scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(datasampleDir, f"{i}.jpg"), out.get_image()[:, :, ::-1])

    cfg = detectron2.config.get_cfg()
    cfg.OUTPUT_DIR = expDir
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("eggTrain",)
    cfg.DATASETS.TEST = ("eggValidation",)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(eggCategories)  # see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # filter proposals with score larger than this to speed up NMS
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 15000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER / 10
    cfg.TEST.DETECTIONS_PER_IMAGE = 512

    trainer = Trainer(cfg)
    trainer.register_hooks(
        [BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer, "bbox/AP", mode="max")])
    trainer.resume_or_load(resume=False)
    trainer.train()

    # tensorboard --logdir output

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25   # filter proposals with score larger than this to speed up NMS
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    valResDir = os.path.join(expDir, "validationResult")
    os.makedirs(valResDir, exist_ok=True)
    for i, d in enumerate(random.sample(validationDataDicts, min(1, len(validationDataDicts)))):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("eggValidation"),
            scale=0.5,
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        predImg = out.get_image()[:, :, ::-1]
        
        logging.info("%s 實際 %d，預測 %d", d["file_name"], len(d["annotations"]), outputs["instances"].pred_classes.shape[0])
        
        imgPath = os.path.join(valResDir, os.path.basename(d["file_name"]))
        cv2.imwrite(imgPath, predImg)


if __name__ == "__main__":
        main()
