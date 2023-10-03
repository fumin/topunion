import sys
sys.path.insert(0, "/usr/local/ByteTrack")

import argparse
import logging
import os
import time
import cv2
import matplotlib as plt
import numpy as np
import torch
import torch_tensorrt
from typing import Tuple, List, Dict

import detectron2
from detectron2 import model_zoo
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.layers import batched_nms, cat, move_device_like

# BYTETracker needs this numpy fix...
np.float = float
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking


def preprocess(predictor, predictorFields, im):
    original_image = im
    if predictor.input_format == "RGB":
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = predictor.aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    infnIn = [{"image": image, "height": height, "width": width}]
    
    batched_inputs = infnIn
    images = [predictor.model._move_to_current_device(x["image"]) for x in batched_inputs]
    images = [(x - predictor.model.pixel_mean) / predictor.model.pixel_std for x in images]
    images = ImageList.from_tensors(
        images,
        predictorFields["size_divisibility"],
        padding_constraints=predictorFields["padding_constraints"],
    )
    return infnIn, images 


def handleFrame(predictor, predictorFields, tracker, metadata, im):
    ts = [{"t": time.perf_counter()}]

    # outputs = predictor(im)
    with torch.no_grad():
        infnIn, images = preprocess(predictor, predictorFields, im)
        ts.append({"name": "detectron2.1", "t": time.perf_counter()})

        # features = predictor.model.backbone(images.tensor)
        # proposals, _ = predictor.model.proposal_generator(images, features, None)
        features, pred_objectness_logits, pred_anchor_deltas, anchors = predictorFields["detectron2"](images.tensor)
        # torch.cuda.synchronize()
        ts.append({"name": "detectron2.2", "t": time.perf_counter()})
        feats = [features[f] for f in predictor.model.proposal_generator.in_features]
        pre_nms_topk = predictor.model.proposal_generator.pre_nms_topk[predictor.model.proposal_generator.training]
        post_nms_topk = predictor.model.proposal_generator.post_nms_topk[predictor.model.proposal_generator.training]
        proposals, _ = choose_topk(predictor.model.proposal_generator.anchor_generator, feats, pred_objectness_logits, pred_anchor_deltas, anchors, predictorFields["image_sizes"], predictor.model.proposal_generator.anchor_generator.box_dim, predictor.model.proposal_generator.min_box_size, predictor.model.proposal_generator.nms_thresh, pre_nms_topk, post_nms_topk, predictor.model.proposal_generator.box2box_transform)

        ts.append({"name": "detectron2.3", "t": time.perf_counter()})
        infnOut, _ = predictor.model.roi_heads(images, features, proposals, None)
        ts.append({"name": "detectron2.4", "t": time.perf_counter()})
        ppOut = GeneralizedRCNN._postprocess(infnOut, infnIn, images.image_sizes)
        outputs = ppOut[0]
    ts.append({"name": "detectron2", "t": time.perf_counter()})

    instances = outputs["instances"]
    detBoxes = instances.pred_boxes.tensor.cpu()
    detBoxesLTWH = np.copy(detBoxes)
    detBoxesLTWH[:, 2:] -= detBoxesLTWH[:, :2]
    detIDs = np.zeros([detBoxes.shape[0]], dtype=np.int32)
    detScores = instances.scores.cpu()
    predImg = plot_tracking(im, detBoxesLTWH, detIDs, detScores)
    cv2.imshow("frame", predImg)
    ts.append({"name": "visualize", "t": time.perf_counter()})

    trackInput = np.zeros([
        outputs["instances"].pred_classes.shape[0],
        5,
        ], dtype=np.float32)
    trackInput[:, 4] = detScores
    trackInput[:, :4] = detBoxes
    w, h = im.shape[1], im.shape[0]
    trackOutput = tracker.update(trackInput, [h, w], [h, w])
    trackBoxes = []
    trackIDs = []
    trackScores = []
    for t in trackOutput:
        b = t.tlwh
        trackBoxes.append(b)
        trackIDs.append(t.track_id)
        trackScores.append(t.score)
    trackPredImg = plot_tracking(im, trackBoxes, trackIDs, trackScores)
    cv2.imshow("track", trackPredImg)
    ts.append({"name": "track", "t": time.perf_counter()})

    durs = []
    for i, t in enumerate(ts[:len(ts)-1]):
        nextT = ts[i+1]
        durs.append({"name": nextT["name"], "t": nextT["t"] - t["t"]})
    # logging.info("%s", durs)


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor
):
    grid_height, grid_width = size
    shifts_x = move_device_like(
        torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )
    shifts_y = move_device_like(
        torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class DefaultAnchorGenerator(torch.nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    def __init__(self, cell_anchors, strides, offset):
        super().__init__()
        self.cell_anchors = cell_anchors
        self.strides = strides
        self.offset = offset

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps
        # return [Boxes(x) for x in anchors_over_all_feature_maps]


def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return torch.jit.is_tracing()


class Detectron2(torch.nn.Module):
    def __init__(self, backbone, proposal_generator, images_tensor, num_images):
        super().__init__()
        self.backbone = backbone

        self.in_features = proposal_generator.in_features
        self.anchor_generator = proposal_generator.anchor_generator
        self.rpn_head = proposal_generator.rpn_head
        self.box_dim = proposal_generator.anchor_generator.box_dim
        self.wx = proposal_generator.box2box_transform.weights[0]
        self.wy = proposal_generator.box2box_transform.weights[1]
        self.ww = proposal_generator.box2box_transform.weights[2]
        self.wh = proposal_generator.box2box_transform.weights[3]
        self.scale_clamp = proposal_generator.box2box_transform.scale_clamp
        self.num_images = num_images
        self.pre_nms_topk = proposal_generator.pre_nms_topk[proposal_generator.training]

        ag = proposal_generator.anchor_generator
        self.anchor_generator = DefaultAnchorGenerator(ag.cell_anchors, ag.strides, ag.offset)

        features = self.backbone(images_tensor)
        feats = [features[f] for f in self.in_features]
        self.anchors = self.anchor_generator(feats)

    def forward(self, images_tensor):
        features = self.backbone(images_tensor)

        # feats = [features[f] for f in self.in_features]
        feats = list(features.values())
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(feats)
        # anchors = self.anchor_generator(feats)
        anchors = self.anchors

        return features, pred_objectness_logits, pred_anchor_deltas, anchors


def choose_topk(anchor_generator, feats, pred_objectness_logits, pred_anchor_deltas, anchors, image_sizes, box_dim, min_box_size, nms_thresh, pre_nms_topk, post_nms_topk, box2box_transform):
    ts = [{"t": time.perf_counter()}]

    # anchors = anchor_generator(feats)
    anchors = [Boxes(x) for x in anchors]
    ts.append({"name": "anchor_generator", "t": time.perf_counter()})

    # Transpose the Hi*Wi*A dimension to the middle:
    pred_objectness_logits = [
        # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        score.permute(0, 2, 3, 1).flatten(1)
        for score in pred_objectness_logits
    ]
    pred_anchor_deltas = [
        # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        x.view(x.shape[0], -1, box_dim, x.shape[-2], x.shape[-1])
        .permute(0, 3, 4, 1, 2)
        .flatten(1, -2)
        for x in pred_anchor_deltas
    ]

    weights, scale_clamp = box2box_transform.weights, box2box_transform.scale_clamp
    wx, wy, ww, wh = weights
    pred_proposals = _decode_proposals(anchors, pred_anchor_deltas, wx, wy, ww, wh, scale_clamp)

    proposals = pred_proposals
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )
    ts.append({"name": "gen_proposals", "t": time.perf_counter()})

    num_images = len(image_sizes)
    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        # topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4
        topk_proposals_i = proposals_i[batch_idx.view(-1, 1), topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)
    ts.append({"name": "concat", "t": time.perf_counter()})

    losses = {}

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        # boxes = Boxes(topk_proposals[n])
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        lvl = level_ids
        ts.append({"name": "box_topk", "t": time.perf_counter()})

        # valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores_per_img)
        ts.append({"name": "box_clip1", "t": time.perf_counter()})
        # if not valid_mask.all():
        #     if training:
        #         raise FloatingPointError(
        #             "Predicted boxes or scores contain Inf/NaN. Training has diverged."
        #         )
        #     boxes = boxes[valid_mask]
        #     scores_per_img = scores_per_img[valid_mask]
        #     lvl = lvl[valid_mask]
        boxes = boxes[valid_mask]
        scores_per_img = scores_per_img[valid_mask]
        lvl = lvl[valid_mask]
        ts.append({"name": "box_clip2", "t": time.perf_counter()})
        # boxes.clip(image_size)
        h, w = image_size[0], image_size[1]
        ts.append({"name": "box_clip3", "t": time.perf_counter()})
        x1 = boxes[:, 0].clamp(min=0, max=w)
        y1 = boxes[:, 1].clamp(min=0, max=h)
        x2 = boxes[:, 2].clamp(min=0, max=w)
        y2 = boxes[:, 3].clamp(min=0, max=h)
        ts.append({"name": "box_clip", "t": time.perf_counter()})

        # filter empty boxes
        # keep = boxes.nonempty(threshold=min_box_size)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > min_box_size) & (heights > min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        ts.append({"name": "filter_empty", "t": time.perf_counter()})
        # keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted
        ts.append({"name": "nms", "t": time.perf_counter()})

        res = Instances(image_size)
        res.proposal_boxes = Boxes(boxes[keep])
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    ts.append({"name": "pg_forward", "t": time.perf_counter()})

    durs = []
    for i, t in enumerate(ts[:len(ts)-1]):
        nextT = ts[i+1]
        durs.append({"name": nextT["name"], "t": nextT["t"] - t["t"]})
    # logging.info("%s", durs)

    return results, losses

def _decode_proposals(anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor], wx: float, wy: float, ww: float, wh: float, scale_clamp: float):
    """
    Transform anchors into proposals by applying the predicted anchor deltas.

    Returns:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape
            (N, Hi*Wi*A, B)
    """
    N = pred_anchor_deltas[0].shape[0]
    proposals = []
    # For each feature map
    for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
        B = anchors_i.tensor.size(1)
        pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
        # Expand anchors to shape (N*Hi*Wi*A, B)
        anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
        proposals_i = apply_deltas(pred_anchor_deltas_i, anchors_i, wx, wy, ww, wh, scale_clamp)
        # Append feature map proposals with shape (N, Hi*Wi*A, B)
        proposals.append(proposals_i.view(N, -1, B))
    return proposals

def apply_deltas(deltas, boxes, wx: float, wy: float, ww: float, wh: float, scale_clamp: float):
    deltas = deltas.float()  # ensure fp32 for decoding precision
    boxes = boxes.to(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    # logging.info("%s %s", widths, widths[:, None])
    pred_ctr_x = dx * widths.view(-1, 1) + ctr_x.view(-1, 1)
    pred_ctr_y = dy * heights.view(-1, 1) + ctr_y.view(-1, 1)
    pred_w = torch.exp(dw) * widths.view(-1, 1)
    pred_h = torch.exp(dh) * heights.view(-1, 1)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return pred_boxes.reshape(deltas.shape)


def preparePredictor(metadata, modelWeights, device, frame):
  cfg = detectron2.config.get_cfg()
  cfg.MODEL.DEVICE = device
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
  cfg.MODEL.WEIGHTS = modelWeights
  cfg.TEST.DETECTIONS_PER_IMAGE = 512
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
  predictor = DefaultPredictor(cfg)


  predictorFields = {
          "size_divisibility": predictor.model.backbone.size_divisibility,
          "padding_constraints": predictor.model.backbone.padding_constraints,
          }
  _, images = preprocess(predictor, predictorFields, frame)
  features = predictor.model.backbone(images.tensor)

  predictorFields["image_sizes"] = torch.tensor(images.image_sizes, device="cpu")
  predictorFields["detectron2"] = Detectron2(predictor.model.backbone, predictor.model.proposal_generator, images.tensor, len(images.image_sizes)).eval()
  # return predictor, predictorFields

  # trt_backbone_path = "trt_backbone.jit"
  # if not os.path.exists(trt_backbone_path):
  #   bbShape = images.tensor.shape
  #   inputs = [
  #           torch_tensorrt.Input(min_shape=bbShape, opt_shape=bbShape, max_shape=bbShape, dtype=images.tensor.dtype)]
  #   enabled_precisions = {images.tensor.dtype}
  #   logging.info("torch script start")
  #   ts_backbone = torch.jit.script(predictor.model.backbone)
  #   logging.info("tensorrt compilation start")
  #   trt_backbone = torch_tensorrt.compile(ts_backbone, inputs=inputs, enabled_precisions=enabled_precisions)
  #   logging.info("tensorrt compilation done")
  #   torch.jit.save(trt_backbone, trt_backbone_path)
  # trt_backbone = torch.jit.load(trt_backbone_path)
  # predictor.model.backbone = trt_backbone

  det2_path = "det2.jit"
  if not os.path.exists(det2_path):
      bbShape = images.tensor.shape
      inputs = [
              torch_tensorrt.Input(min_shape=bbShape, opt_shape=bbShape, max_shape=bbShape, dtype=images.tensor.dtype)
              ]
      enabled_precisions = {images.tensor.dtype}
      with torch_tensorrt.logging.debug():
        det2 = torch_tensorrt.compile(predictorFields["detectron2"], inputs=inputs, enabled_precisions=enabled_precisions)
      torch.jit.save(det2, det2_path)
  predictorFields["detectron2"] = torch.jit.load(det2_path)

  return predictor, predictorFields


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    plt.rcParams["font.family"]=["Noto Serif CJK JP"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)
    if not cap.isOpened():
        raise Exception("not opened")
    ret, frame = cap.read()
    if not ret:
        raise Exception("no frame")
 
    metadata = Metadata()
    metadata.thing_classes = ["蛋"]
    device = "cuda"
    predictor, predictorFields = preparePredictor(metadata, "model_best.pth", device, frame)

    trackerArg = argparse.Namespace()
    trackerArg.track_thresh = 0.5
    trackerArg.track_buffer = 30
    trackerArg.match_thresh = 0.8
    trackerArg.mot20 = False
    tracker = BYTETracker(trackerArg)

    durations = []
    while True:
        startT = time.perf_counter()
    
        ret, frame = cap.read()
        if not ret:
            logging.info("no frame")
            continue
        handleFrame(predictor, predictorFields, tracker, metadata, frame)
        
        duration = time.perf_counter() - startT
        durations.append(duration)
        if len(durations) >= 20:
            logging.info("fps %f", len(durations)/sum(durations))
            durations = []
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
