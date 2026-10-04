"""Small CPU example using synthetic inputs and randomly initialized weights.

From the repository root, after installing the package:
    python -m examples.quickstart --phase infer
    python -m examples.quickstart --phase train
    python -m examples.quickstart --phase validate
"""
import argparse
import torch

from hsi_refine import ModelConfig
from hsi_refine.factory import build_rfdetr_model
from hsi_refine.geometry import cxcywh_to_xyxy


def make_batch(device):
    cubes = torch.rand(1, 16, 64, 96, device=device, dtype=torch.float32)
    targets = [{
        "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.4]], device=device, dtype=torch.float32),
        "labels": torch.tensor([1], device=device, dtype=torch.int64),
    }]
    return cubes, targets


def run(phase="infer", device="cpu"):
    torch.manual_seed(7)
    if torch.device(device).type == "cpu":
        torch.set_num_threads(2)
    config = ModelConfig(
        num_classes=3, detector_resolution=64, group_detr=1,
        fusion_channels=16, roi_size=8, boundary_bins=32,
        train_topk=8, eval_topk=8,
    )
    model, criterion = build_rfdetr_model(config, device=device, initialize_pretrained=False)
    cubes, targets = make_batch(device)

    if phase == "train":
        model.train()
        criterion.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(cubes, targets)
        losses, roi_stats = criterion(outputs, targets)
        loss = criterion.total(losses)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        print(f"train: loss={loss.detach().item():.4f}, {roi_stats}")

    elif phase == "validate":
        model.eval()
        criterion.eval()
        with torch.no_grad():
            # The same forward supplies predictions and validation losses.
            outputs = model(cubes, targets)
            losses, roi_stats = criterion(outputs, targets)
            loss = criterion.total(losses)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        print(f"validate: loss={loss.item():.4f}, {roi_stats}")

    elif phase == "infer":
        model.eval()
        with torch.no_grad():
            outputs = model(cubes)
        # Select valid class channels, then decode one label per query.
        start = config.class_offset
        probabilities = outputs["pred_logits"][..., start:start + config.num_classes].sigmoid()
        scores, labels = probabilities.max(dim=-1)
        labels = labels + start
        height, width = cubes.shape[-2:]
        scale = outputs["pred_boxes"].new_tensor([width, height, width, height])
        boxes = cxcywh_to_xyxy(outputs["pred_boxes"]).clamp(0, 1) * scale
        keep = scores[0] >= 0.5
        detections = {"boxes": boxes[0, keep], "labels": labels[0, keep], "scores": scores[0, keep]}
        print("infer: detections=", {key: value.detach().cpu().tolist() for key, value in detections.items()})
    else:
        raise ValueError(f"Unknown phase: {phase}")

    print("output shapes:", {key: tuple(outputs[key].shape) for key in ("pred_logits", "pred_boxes")})
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validate", "infer"), default="infer")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    args = parser.parse_args()
    run(args.phase, args.device)
