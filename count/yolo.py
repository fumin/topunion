import json
import logging
import os

import cv2
import matplotlib as plt
import ultralytics
import yaml

def newDataset(expDir, dataDir):
    imgDirTrain = os.path.join(expDir, "data", "images", "train")
    os.makedirs(imgDirTrain, exist_ok=True)
    imgDirVal = os.path.join(expDir, "data", "images", "val")
    os.makedirs(imgDirVal, exist_ok=True)
    labelDirTrain = os.path.join(expDir, "data", "labels", "train")
    os.makedirs(labelDirTrain, exist_ok=True)
    labelDirVal = os.path.join(expDir, "data", "labels", "val")
    os.makedirs(labelDirVal, exist_ok=True)

    for i, dirBase in enumerate(os.listdir(dataDir)):
        dDir = os.path.join(dataDir, dirBase)

        json_file = os.path.join(dDir, "label.json")
        with open(json_file) as f:
            viaSetting = json.load(f)

        for idx, k in enumerate(viaSetting["_via_image_id_list"]):
            v = viaSetting["_via_img_metadata"][k]

            fpath = os.path.join(dDir, "img", v["filename"])
            imgH, imgW = cv2.imread(fpath).shape[:2]

            labels = []
            for _, anno in enumerate(v["regions"]):
                s = anno["shape_attributes"]

                category = 0  # categoryIDMap[anno["region_attributes"]["klass"]]
                centerX = (s["x"] + s["width"] / 2) / imgW
                centerY = (s["y"] + s["height"] / 2) / imgH
                width = s["width"] / imgW
                height = s["height"] / imgH

                labels.append("%g %.6f %.6f %.6f %.6f" % (category, centerX, centerY, width, height))
            labelStr = "\n".join(labels)

            outImages, outLabels = imgDirTrain, labelDirTrain
            if idx % 10 >= 8:
                outImages, outLabels = imgDirVal, labelDirVal

            base = os.path.basename(fpath)
            outImg = os.path.join(outImages, f"{dirBase}_{base}")
            os.symlink(fpath, outImg)
            noext = os.path.splitext(base)[0]
            outLabel = os.path.join(outLabels, f"{dirBase}_{noext}.txt")
            with open(outLabel, "w") as f:
                f.write(labelStr)

    cfg = {}
    cfg["path"] = ""
    cfg["train"] = imgDirTrain
    cfg["val"] = imgDirVal
    cfg["test"] = ""
    cfg["names"] = ["egg"]
    yamlStr = yaml.safe_dump(cfg)
    cfgPath = os.path.join(expDir, f"data.yaml")
    with open(cfgPath, "w") as f:
        f.write(yamlStr)
    return cfgPath


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]

    expDir = os.path.join(os.getcwd(), "exp")
    dataDir = os.path.join(os.getcwd(), "data")

    dataCfg = newDataset(expDir, dataDir)
    model = ultralytics.YOLO("yolov8n.pt")
    model.train(data=dataCfg, epochs=10, project=expDir)


if __name__ == "__main__":
    main()
