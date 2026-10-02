from contextlib import nullcontext
import torch
from torch import nn

from .config import ModelConfig
from .geometry import (absolute_edges, crop_features, cxcywh_to_xyxy,
                       relative_edges, search_regions, valid_xyxy, xyxy_to_cxcywh)
from .modules import (BoundaryDistributionHead, HighResolutionStem, ROIFusion,
                      RegressionHead, SpectralAdapter)


class HSIBoundaryDetector(nn.Module):
    """RF-DETR wrapper with spectral adaptation and local box refinement.

    cubes are fixed-size, unpadded Bx16xHxW tensors in [0,1]. Boxes are
    normalized cxcywh. The original classification/auxiliary/encoder outputs
    are retained. Selected final boxes are refined in one forward pass.
    """
    def __init__(self, detector, config: ModelConfig, matcher=None, detector_input_builder=None):
        super().__init__()
        self.detector, self.config, self.matcher = detector, config, matcher
        self.detector_input_builder = detector_input_builder
        self.adapter = SpectralAdapter(config.input_channels)
        self.refiner_enabled = config.mode != "baseline"
        self.use_highres = config.mode in {"highres", "combined"}
        if self.refiner_enabled:
            if not hasattr(detector, "backbone"):
                raise TypeError("Expected an RF-DETR nn.Module exposing .backbone")
            c = config.fusion_channels
            self.global_projection = nn.Conv2d(config.global_channels, c, 1)
            self.highres_stem = HighResolutionStem(config.input_channels, c) if self.use_highres else None
            self.fusion = ROIFusion(c, self.use_highres)
            self.head = (RegressionHead(c) if config.mode == "highres" else
                         BoundaryDistributionHead(c, config.boundary_bins, config.prior_sigma))
        if config.freeze_detector:
            self.detector.requires_grad_(False).eval()
            self.adapter.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_detector:
            self.detector.eval()
            self.adapter.eval()
        return self

    def _coarse_forward(self, images):
        cache = []

        def capture(_module, _inputs, output):
            features = output[0]
            feature = features[0]
            tensor = feature.tensors if hasattr(feature, "tensors") else feature
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise RuntimeError("Unsupported RF-DETR backbone output; expected a 4D feature tensor")
            cache.append(tensor)

        handle = self.detector.backbone.register_forward_hook(capture) if self.refiner_enabled else None
        try:
            context = torch.no_grad() if self.config.freeze_detector else nullcontext()
            with context:
                samples = self.detector_input_builder(images) if self.detector_input_builder else images
                output = self.detector(samples)
        finally:
            if handle is not None:
                handle.remove()
        if self.refiner_enabled and len(cache) != 1:
            raise RuntimeError("Backbone was not observed exactly once; this RF-DETR interface is unsupported")
        return output, cache[0] if cache else None

    def _select(self, coarse, matches):
        logits = coarse["pred_logits"]
        start, stop = self.config.class_offset, self.config.class_offset + self.config.num_classes
        if logits.shape[-1] < stop:
            raise ValueError("Checkpoint classification head does not match class_offset/num_classes")
        scores = logits[..., start:stop].detach().sigmoid().amax(dim=-1)
        topk = min(self.config.train_topk if self.training else self.config.eval_topk, scores.shape[1])
        pairs, assignments = [], []
        for b in range(scores.shape[0]):
            selected = scores[b].topk(topk).indices
            gt_for_query = torch.full((scores.shape[1],), -1, device=scores.device, dtype=torch.long)
            if matches is not None:
                q, g = matches[b]
                q, g = q.to(scores.device), g.to(scores.device)
                # Keep every matched positive, even when low-confidence or
                # duplicated across Group-DETR training groups.
                selected = torch.unique(torch.cat((selected, q)), sorted=True)
                gt_for_query[q] = g
            pairs.append(torch.stack((torch.full_like(selected, b), selected), dim=1))
            assignments.append(gt_for_query[selected])
        return torch.cat(pairs), torch.cat(assignments)

    def forward(self, cubes, targets=None):
        if cubes.ndim != 4 or cubes.shape[1] != self.config.input_channels:
            raise ValueError(f"Expected Bx{self.config.input_channels}xHxW cubes")
        if cubes.shape[0] < 1 or min(cubes.shape[-2:]) < 4:
            raise ValueError("Empty batch or image too small")
        images = self.adapter(cubes, self.config.detector_resolution)
        coarse, global_features = self._coarse_forward(images)
        if not isinstance(coarse, dict) or not {"pred_logits", "pred_boxes"} <= coarse.keys():
            raise TypeError("Expected unexported RF-DETR dictionary outputs")
        output = dict(coarse)
        output["coarse_outputs"] = coarse
        if not self.refiner_enabled:
            output["refinement"] = None
            return output
        if targets is not None and self.matcher is None:
            raise ValueError("Training with targets requires the RF-DETR Hungarian matcher")
        matches = None
        if targets is not None:
            group = self.config.group_detr if self.detector.training else 1
            with torch.no_grad():
                matches = self.matcher(coarse, targets, group_detr=group)
        pairs, gt_indices = self._select(coarse, matches)
        batch_indices, query_indices = pairs.unbind(1)
        proposals = valid_xyxy(cxcywh_to_xyxy(coarse["pred_boxes"][batch_indices, query_indices].detach().float()))
        regions = search_regions(proposals, self.config.search_expansion)
        proposal_edges = relative_edges(proposals, regions)
        if global_features.shape[1] != self.config.global_channels:
            raise RuntimeError("global_channels does not match the projected RF-DETR backbone feature")
        global_features = self.global_projection(global_features)
        highres_features = self.highres_stem(cubes) if self.use_highres else None
        chunks = []
        for first in range(0, len(pairs), self.config.roi_chunk_size):
            last = first + self.config.roi_chunk_size
            local_global = crop_features(global_features, regions[first:last],
                                         batch_indices[first:last], self.config.roi_size)
            features = [local_global]
            if highres_features is not None:
                features.append(crop_features(highres_features, regions[first:last],
                                              batch_indices[first:last], self.config.roi_size))
            fused = self.fusion(torch.cat(features, dim=1))
            chunks.append(self.head(fused, proposal_edges[first:last]))
        refinement = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
        predicted = valid_xyxy(absolute_edges(refinement["edges"], regions))
        # Convex blending of valid xyxy boxes preserves edge ordering.
        blended = proposals + self.config.refine_blend * (predicted - proposals)
        refined_boxes = coarse["pred_boxes"].clone()
        refined_boxes[batch_indices, query_indices] = xyxy_to_cxcywh(blended).to(refined_boxes)
        output["pred_boxes"] = refined_boxes
        refinement.update({"pairs": pairs, "gt_indices": gt_indices, "regions": regions,
                           "raw_refined_xyxy": predicted, "blended_xyxy": blended})
        output["refinement"] = refinement
        return output
