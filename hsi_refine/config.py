from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    input_channels: int = 16
    num_classes: int = 18
    class_offset: int = 0
    detector_resolution: int = 640
    group_detr: int = 13
    full_iterative: bool = True
    mode: str = "combined"  # baseline, highres, locnet, combined
    adapter_init: str = "xavier"
    global_channels: int = 256
    fusion_channels: int = 48
    roi_size: int = 28
    boundary_bins: int = 128
    search_expansion: float = 1.6
    prior_sigma: float = 0.10
    refine_blend: float = 0.5
    train_topk: int = 64
    eval_topk: int = 200
    roi_chunk_size: int = 64
    freeze_detector: bool = False

    def __post_init__(self):
        if self.mode not in {"baseline", "highres", "locnet", "combined"}:
            raise ValueError("Unknown model mode")
        if self.adapter_init != "xavier":
            raise ValueError("adapter_init must be 'xavier' (Xavier uniform initialization)")
        if self.class_offset not in {0, 1}:
            raise ValueError("class_offset must be 0 or 1; match the original checkpoint")
        if self.num_classes < 1 or self.input_channels < 3:
            raise ValueError("Invalid channel/class count")
        if self.detector_resolution < 32 or self.detector_resolution % 32:
            raise ValueError("RF-DETR Small resolution must be divisible by 32")
        if self.search_expansion <= 1 or self.prior_sigma <= 0:
            raise ValueError("Search expansion must exceed 1; sigma must be positive")
        if not 0 < self.refine_blend <= 1:
            raise ValueError("refine_blend must be in (0, 1]")
        if min(self.fusion_channels, self.global_channels, self.group_detr,
               self.train_topk, self.eval_topk, self.roi_chunk_size) < 1:
            raise ValueError("Channels, groups and candidate limits must be positive")
        if self.roi_size < 4 or self.boundary_bins < 4:
            raise ValueError("ROI size and boundary bins must be at least 4")

    def to_dict(self):
        return asdict(self)
