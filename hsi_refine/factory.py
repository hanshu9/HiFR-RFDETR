"""RF-DETR construction, component weights and loss integration."""
from importlib.metadata import version
from pathlib import Path
import torch
from .weights import extract_state, load_checkpoint
from .model import HSIBoundaryDetector
from .losses import RefinementCriterion


def _fixed_size_nested_tensor(images):
    """Use RF-DETR's explicit input contract for fixed-size, unpadded batches.

    This also bypasses 1.3.0's tensor-list padding loop, whose in-place writes
    through unbound views are incompatible with autograd in recent PyTorch.
    The image tensor stays connected to the spectral adapter's graph.
    """
    from rfdetr.util.misc import NestedTensor
    mask = torch.zeros((images.shape[0], *images.shape[-2:]),
                       dtype=torch.bool, device=images.device)
    return NestedTensor(images, mask)


def _add_shared_bbox_aliases(state, expected):
    state = dict(state)
    alias = "transformer.decoder.bbox_embed."
    for key in expected:
        if key.startswith(alias) and key not in state:
            source = "bbox_embed." + key[len(alias):]
            if source in state:
                state[key] = state[source]
    return state


def load_detector_weights(detector, state, coco_pretrained=False):
    expected = detector.state_dict()
    state = _add_shared_bbox_aliases(state, expected)
    if not coco_pretrained:
        detector.load_state_dict(state, strict=True)
        return
    # Only the explicitly replaced class heads may be absent in COCO transfer.
    def classification_key(key):
        return key.startswith("class_embed.") or key.startswith("transformer.enc_out_class_embed.")
    for key in list(state):
        if classification_key(key):
            state.pop(key)
    for key in ("refpoint_embed.weight", "query_feat.weight"):
        if key in state and key in expected:
            if state[key].shape[0] < expected[key].shape[0]:
                raise ValueError("COCO weights contain fewer query groups than the requested model")
            state[key] = state[key][:expected[key].shape[0]]
    # Encoder projections, normalization and box heads follow the query groups.
    for prefix in ("transformer.enc_output.", "transformer.enc_output_norm.",
                   "transformer.enc_out_bbox_embed."):
        last_group = max((int(key[len(prefix):].split(".", 1)[0])
                          for key in expected if key.startswith(prefix)), default=-1)
        for key in list(state):
            if key.startswith(prefix):
                group, _, suffix = key[len(prefix):].partition(".")
                reference = prefix + "0." + suffix
                if (group.isdecimal() and int(group) > last_group and
                        reference in expected and state[key].shape == expected[reference].shape):
                    state.pop(key)
    # Unused alias tensors can appear in full-iterative weights loaded into a
    # lite model. Remove only this documented alias, never arbitrary keys.
    for key in list(state):
        if key.startswith("transformer.decoder.bbox_embed.") and key not in expected:
            state.pop(key)
    for key, value in state.items():
        if key in expected and value.shape != expected[key].shape:
            raise ValueError(f"Unexpected COCO tensor shape for {key}: {value.shape} vs {expected[key].shape}")
    result = detector.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not classification_key(k)]
    if bad_missing or result.unexpected_keys:
        raise ValueError(f"Unsupported checkpoint: missing={bad_missing}, unexpected={result.unexpected_keys}")


def build_rfdetr_model(config, device="cpu", initialize_pretrained=False,
                       coco_weights="rf-detr-small.pth", detector_checkpoint=None,
                       detector_state_key="model", detector_prefix="",
                       adapter_checkpoint=None, adapter_state_key="model", adapter_prefix=""):
    try:
        installed = version("rfdetr")
    except Exception as exc:
        raise RuntimeError("Install requirements.txt first; this environment has no RF-DETR") from exc
    if installed != "1.3.0":
        raise RuntimeError(f"This factory requires rfdetr==1.3.0; found {installed}")
    from rfdetr.config import RFDETRSmallConfig
    from rfdetr.main import populate_args, download_pretrain_weights
    from rfdetr.models import build_model, build_criterion_and_postprocessors

    small = RFDETRSmallConfig(num_classes=config.num_classes,
                             resolution=config.detector_resolution,
                             group_detr=config.group_detr,
                             lite_refpoint_refine=not config.full_iterative,
                             pretrain_weights=None, device=torch.device(device).type)
    args = populate_args(**small.model_dump(), force_no_pretrain=True)
    args.device = str(device)
    detector = build_model(args)
    if detector_checkpoint:
        state = extract_state(load_checkpoint(detector_checkpoint), detector_state_key, detector_prefix)
        load_detector_weights(detector, state, coco_pretrained=False)
    elif initialize_pretrained:
        if not Path(coco_weights).is_file():
            download_pretrain_weights(coco_weights)
        state = extract_state(load_checkpoint(coco_weights))
        load_detector_weights(detector, state, coco_pretrained=True)
    base_criterion, _postprocessors = build_criterion_and_postprocessors(args)
    if config.freeze_detector:
        # A frozen detector runs in eval mode and emits one query group.
        # Its criterion must not try to split those queries into 13 groups.
        base_criterion.group_detr = 1
    model = HSIBoundaryDetector(detector, config, matcher=base_criterion.matcher,
                               detector_input_builder=_fixed_size_nested_tensor)
    if adapter_checkpoint:
        state = extract_state(load_checkpoint(adapter_checkpoint), adapter_state_key, adapter_prefix)
        # Accept either the projection Conv2d's weight/bias or the complete
        # SpectralAdapter state. No guessed prefix stripping is performed.
        if set(state) == {"weight", "bias"}:
            model.adapter.projection.load_state_dict(state, strict=True)
        else:
            model.adapter.load_state_dict(state, strict=True)
    return model.to(device), RefinementCriterion(base_criterion).to(device)
