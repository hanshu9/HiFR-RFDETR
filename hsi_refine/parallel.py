"""DataParallel with image-wise detection targets and global ROI indices."""
import torch
from torch import nn
from torch.nn.parallel.scatter_gather import gather, scatter

from .model import HSIBoundaryDetector


def _move_target(value, device):
    """Move complete target tensors, including empty boxes and scalar metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_target(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_target(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_target(item, device) for item in value)
    return value


class HSIDataParallel(nn.DataParallel):
    """Wrap an HSIBoundaryDetector for batch-dimension CUDA data parallelism.

    The model must be on device_ids[0]. Compute RefinementCriterion once on
    gathered outputs and the original complete targets on output_device.
    Only images are split as tensors: each image's targets stay together.
    """
    def __init__(self, module, device_ids=None, output_device=None):
        if not isinstance(module, HSIBoundaryDetector):
            raise TypeError("HSIDataParallel requires an HSIBoundaryDetector")
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=0)

    def forward(self, cubes, targets=None):
        if not isinstance(cubes, torch.Tensor) or cubes.ndim != 4 or cubes.shape[0] < 1:
            raise ValueError("Expected a nonempty BxCxHxW cube batch")
        if targets is not None:
            if (not isinstance(targets, (list, tuple)) or len(targets) != len(cubes)
                    or not all(isinstance(target, dict) for target in targets)):
                raise ValueError("targets must contain one dictionary per image")
        # Normalize positional and keyword calls before the parent scatters.
        return super().forward(cubes, targets)

    def scatter(self, inputs, kwargs, device_ids):
        cubes, targets = inputs
        shards = scatter(cubes, device_ids, dim=0)
        split_inputs = []
        first = 0
        for shard in shards:
            count = len(shard)
            if count == 0:
                continue
            local_targets = (None if targets is None else
                             [_move_target(target, shard.device)
                              for target in targets[first:first + count]])
            split_inputs.append((shard, local_targets))
            first += count
        return tuple(split_inputs), tuple({} for _ in split_inputs)

    def replicate(self, module, device_ids):
        replicas = super().replicate(module, device_ids)
        for replica in replicas:
            replica._hsi_parallel_replica = True
        return replicas

    def gather(self, outputs, output_device):
        rebased = []
        batch_offset = 0
        for output in outputs:
            refinement = output["refinement"]
            if refinement is not None:
                pairs = refinement["pairs"].clone()
                pairs[:, 0] += batch_offset
                # Keep replica outputs intact and preserve autograd on every
                # floating tensor; only integer image indices need rewriting.
                output = {**output, "refinement": {**refinement, "pairs": pairs}}
            rebased.append(output)
            batch_offset += output["pred_boxes"].shape[0]
        return gather(rebased, output_device, dim=0)
