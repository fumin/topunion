import sys
sys.path.insert(0, "/usr/local/unimatch")

import logging
import os
import time
import cv2
import matplotlib as plt
import numpy as np
import torch
from PIL import Image

from unimatch.unimatch import UniMatch
from dataloader.stereo import transforms

from detectron2.data import Metadata
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from segment_anything import sam_model_registry, SamPredictor

import pyrealsense2 as rs


def getCamInfo():
    depthToDisp = rs.disparity_transform(True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    profile = pipeline.start(config)

    depth_profile = profile.get_stream(rs.stream.depth)
    intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # https://dev.intelrealsense.com/docs/api-how-to#get-disparity-baseline
    ir1Stream = profile.get_stream(rs.stream.infrared, 1)
    ir2Stream = profile.get_stream(rs.stream.infrared, 2)
    extrinsics = ir1Stream.get_extrinsics_to(ir2Stream)
    baseline = -1e3 * extrinsics.translation[0]

    pipeline.stop()

    return baseline, intrinsics


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def newUniMatch():
    model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task="stereo")
    modelPath = "/usr/local/unimatch/checkpoints/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth"
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = "cuda"
    model.to(device)
    val_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    val_transform = transforms.Compose(val_transform_list)

    return model, val_transform


def unimatchRun(model, val_transform, left, right):
    sample = {'left': left, 'right': right}

    sample = val_transform(sample)

    device = "cuda"
    left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
    right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
    
    padding_factor = 32
    nearest_size = [
            int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
            int(np.ceil(left.size(-1) / padding_factor)) * padding_factor
    ]
    inference_size = nearest_size
    ori_size = left.shape[-2:]
    logging.info("left.shape %s, inference_size %s", left.shape, inference_size)
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        left = F.interpolate(left, size=inference_size,
                             mode='bilinear',
                             align_corners=True)
        right = F.interpolate(right, size=inference_size,
                              mode='bilinear',
                              align_corners=True)

    with torch.no_grad():
        pred_disp = model(
                left, right,
                attn_type="self_swin2d_cross_1d",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task="stereo"
                )["flow_preds"][-1]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        # resize back
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                  mode='bilinear',
                                  align_corners=True).squeeze(1)  # [1, H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    return pred_disp


def newSam(frame):
    cpPath = "/usr/local/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=cpPath)
    sam = sam.to(device="cuda").eval()

    img_size = sam.image_encoder.img_size

    # sam.image_encoder = torch.jit.script(sam.image_encoder, example_inputs=[frame])
    # sam.image_encoder = torch.jit.optimize_for_inference(sam.image_encoder)

    # input_signature = ([torch_tensorrt.Input(shape=frame.shape, dtype=torch.half)])
    # enabled_precisions = {torch.half,}
    # sam.image_encoder = torch_tensorrt.compile(sam.image_encoder, input_signature=input_signature, enabled_precisions=enabled_precisions)

    sam.image_encoder.img_size = img_size

    predictor = SamPredictor(sam)
    return predictor


def samRun(sam, im, boxesNP):
    instances = Instances(
        (im.shape[0], im.shape[1]),
        pred_classes=np.zeros(boxesNP.shape[0], dtype=np.int64),
        scores=np.ones(boxesNP.shape[0]),
        pred_boxes=Boxes(boxesNP))
    boxes = sam.transform.apply_boxes_torch(instances.pred_boxes.tensor, im.shape[:2]).to("cuda")
    sam.set_image(im)
    masks, scores, logits = sam.predict_torch(
            boxes=boxes, multimask_output=False,
            point_coords=None, point_labels=None)

    masks = np.squeeze(masks, axis=1)
    instances.pred_masks = masks
    metadata = Metadata()
    metadata.thing_classes = ["egg"]
    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=1.0,
        )
    out = v.draw_instance_predictions(instances.to("cpu"))
    predImg = out.get_image()[:, :, ::-1]

    return masks[0], predImg


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]

    baseline, intrinsics = getCamInfo()
    focalLength = intrinsics.fx

    color_name = "output1111/1688385554944_c.png"
    left_name = "output1111/1688385554944_ir1.png"
    right_name = "output1111/1688385554944_ir2.png"
    dpath = "disp.png"

    colorImg = cv2.imread(color_name)
    lll = cv2.imread(left_name)
    rrr = cv2.imread(right_name)

    left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
    right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)

    left = cv2.cvtColor(lll, cv2.COLOR_BGR2RGB).astype(np.float32)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB).astype(np.float32)

    unimatchM, unimatchT = newUniMatch()
    sam = newSam(colorImg)

    pred_disp = unimatchRun(unimatchM, unimatchT, left, right)
    disp = pred_disp[0].cpu().numpy()
    dispViz = vis_disparity(disp)
    cv2.imwrite(dpath, dispViz)

    boxes = np.array([
        [180, 190, 270, 310],
        [275, 185, 355, 295],
        [355, 180, 440, 290],
    ])
    mask, predImg = samRun(sam, colorImg, boxes)
    cv2.imwrite("sam.png", predImg)

    eggDisp = np.multiply(disp, mask.cpu().numpy())

    for i, box in enumerate(boxes):
        # box = boxes[0]
        logging.info("%d", i)
        eggDepth = baseline*focalLength / np.max(disp[box[1]:box[3], box[0]:box[2]])
        planeDepth = baseline*focalLength / np.min(disp[box[1]:box[3], box[0]:box[2]])
        eggHeight = planeDepth - eggDepth
        logging.info("蛋高度 (mm): %f %f %f", eggHeight, eggDepth, planeDepth)

        circumferenceDepth = (planeDepth + eggDepth) / 2
        pixToWorld = circumferenceDepth / focalLength
        pixelArea = float(np.count_nonzero(eggDisp))
        pixelWorld = pixelArea * pixToWorld * pixToWorld
        logging.info("蛋面積 (mm*mm): %f", pixelWorld)
        logging.info("蛋體積 (mm^3): %f", pixelWorld*4/3*eggHeight/2)


if __name__ == "__main__":
	main()
