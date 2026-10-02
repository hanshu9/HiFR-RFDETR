"""Boxes use normalized coordinates relative to the complete, unpadded image."""
import torch
from torchvision.ops import roi_align


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(boxes):
    lo, hi = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((lo + hi) / 2, hi - lo), dim=-1)


def valid_xyxy(boxes, eps=1e-5):
    lo = torch.minimum(boxes[..., :2], boxes[..., 2:]).clamp(0, 1 - eps)
    hi = torch.maximum(boxes[..., :2], boxes[..., 2:]).clamp(eps, 1)
    hi = torch.maximum(hi, lo + eps)
    return torch.cat((lo, hi), dim=-1)


def search_regions(proposals_xyxy, expansion):
    center = (proposals_xyxy[:, :2] + proposals_xyxy[:, 2:]) / 2
    half = (proposals_xyxy[:, 2:] - proposals_xyxy[:, :2]) * expansion / 2
    return valid_xyxy(torch.cat((center - half, center + half), dim=1))


def relative_edges(boxes_xyxy, regions):
    origin = regions[:, :2].repeat(1, 2)
    size = (regions[:, 2:] - regions[:, :2]).repeat(1, 2).clamp_min(1e-6)
    return (boxes_xyxy - origin) / size


def absolute_edges(edges, regions):
    origin = regions[:, :2].repeat(1, 2)
    size = (regions[:, 2:] - regions[:, :2]).repeat(1, 2)
    return edges * size + origin


def crop_features(features, regions, batch_indices, roi_size):
    # Scale normalized ROIs into this feature map's own coordinate system.
    # Each branch can have a different H/W without a hardcoded stride.
    scale = regions.new_tensor([features.shape[-1], features.shape[-2]] * 2)
    rois = torch.cat((batch_indices[:, None].to(regions), regions * scale), dim=1)
    # Keep feature/ROI dtypes identical under CUDA mixed precision. Coordinate
    # math and sampling use float32; gradients still flow into both branches.
    with torch.autocast(device_type=features.device.type, enabled=False):
        return roi_align(features.float(), rois.float(), (roi_size, roi_size),
                         spatial_scale=1.0, sampling_ratio=2, aligned=True)


def paired_giou(boxes1, boxes2):
    lo = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    hi = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter = (hi - lo).clamp_min(0).prod(dim=1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=1)
    union = area1 + area2 - inter
    enclosing_lo = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_hi = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing = (enclosing_hi - enclosing_lo).clamp_min(0).prod(dim=1)
    return inter / union.clamp_min(1e-8) - (enclosing - union) / enclosing.clamp_min(1e-8)
