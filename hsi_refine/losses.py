import torch
from torch import nn
from torch.nn import functional as F
from .geometry import cxcywh_to_xyxy, paired_giou, relative_edges


def interpolated_boundary_loss(logits, coordinates):
    """Two-neighbor soft labels; the coordinate remains continuous."""
    bins = logits.shape[-1]
    position = coordinates.clamp(0, 1) * (bins - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(bins - 1)
    right_weight = position - left
    log_probs = logits.float().log_softmax(-1)
    loss_left = -log_probs.gather(-1, left[..., None]).squeeze(-1)
    loss_right = -log_probs.gather(-1, right[..., None]).squeeze(-1)
    return ((1 - right_weight) * loss_left + right_weight * loss_right).mean()


class RefinementCriterion(nn.Module):
    """Original RF-DETR losses on coarse outputs + positive ROI refinement."""
    def __init__(self, detector_criterion, boundary_weight=1.0, l1_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.detector_criterion = detector_criterion
        self.weight_dict = dict(detector_criterion.weight_dict)
        self.weight_dict.update(loss_boundary=boundary_weight, loss_refine_l1=l1_weight,
                                loss_refine_giou=giou_weight)

    def forward(self, outputs, targets):
        ref = outputs["refinement"]
        assignments = ref.get("has_target_assignments") if ref is not None else None
        if ref is not None and (assignments is None or not assignments.all()):
            raise ValueError(
                "Refinement loss requires targets in the model forward pass. "
                "Call model(cubes, targets) before criterion(outputs, targets). "
                "For target-free inference, use model.eval() and model(cubes) "
                "without computing a loss."
            )
        losses = self.detector_criterion(outputs["coarse_outputs"], targets)
        zero = outputs["pred_boxes"].sum() * 0
        stats = {"matched_rois": 0, "supervised_rois": 0}
        if ref is None:
            return losses, stats
        # Connect the graph even for all-empty batches and zero covered positives.
        zero = zero + ref["raw_refined_xyxy"].sum() * 0
        if "boundary_logits" in ref:
            zero = zero + ref["boundary_logits"].sum() * 0
        losses.update(loss_boundary=zero, loss_refine_l1=zero, loss_refine_giou=zero)
        positive = ref["gt_indices"] >= 0
        stats["matched_rois"] = int(positive.sum().item())
        if not positive.any():
            return losses, stats
        pairs = ref["pairs"][positive]
        gt_ids = ref["gt_indices"][positive]
        # Flatten per-image targets once; grouped matches index them on-device.
        counts = pairs.new_tensor([len(target["boxes"]) for target in targets])
        offsets = counts.cumsum(0) - counts
        boxes = torch.cat([target["boxes"] for target in targets], dim=0)
        gt_boxes = boxes[offsets[pairs[:, 0]] + gt_ids]
        gt_xyxy = cxcywh_to_xyxy(gt_boxes.float())
        relative = relative_edges(gt_xyxy, ref["regions"][positive])
        covered = ((relative >= 0) & (relative <= 1)).all(dim=1)
        if "boundary_logits" in ref:
            # Decoding subtracts the prior mean to retain exact initial coarse
            # boxes, so labels need the same change of coordinates.
            encoded = relative - ref["proposal_edges"][positive] + ref["prior_mean"][positive]
            covered &= ((encoded >= 0) & (encoded <= 1)).all(dim=1)
        else:
            encoded = relative
        stats["supervised_rois"] = int(covered.sum().item())
        if not covered.any():
            return losses, stats
        if "boundary_logits" in ref:
            losses["loss_boundary"] = interpolated_boundary_loss(ref["boundary_logits"][positive][covered], encoded[covered])
        # Supervise the raw refined prediction so the training objective stays
        # independent of the coarse/refined blend used in the model output.
        refined = ref["raw_refined_xyxy"][positive][covered]
        target = gt_xyxy[covered]
        losses["loss_refine_l1"] = F.l1_loss(refined, target)
        losses["loss_refine_giou"] = (1 - paired_giou(refined, target)).mean()
        return losses, stats

    def total(self, losses):
        terms = [value * self.weight_dict[key] for key, value in losses.items() if key in self.weight_dict]
        if not terms:
            raise RuntimeError("No weighted losses produced")
        return torch.stack(terms).sum()
