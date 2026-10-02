import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class SpectralAdapter(nn.Module):
    """Xavier-initialized spectral projection with RF-DETR normalization."""
    def __init__(self, in_channels=16):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 3, 1, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, cube, resolution):
        features = self.projection(cube)
        features = F.interpolate(features, (resolution, resolution), mode="bilinear", align_corners=False)
        return (features - self.mean) / self.std


class HighResolutionStem(nn.Sequential):
    """Stride-4 feature map computed from all spectral bands."""
    def __init__(self, in_channels, channels):
        super().__init__(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1, bias=False),
            norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            norm(channels), nn.GELU(),
        )


class ROIFusion(nn.Sequential):
    def __init__(self, channels, use_highres):
        super().__init__(
            nn.Conv2d(channels * (2 if use_highres else 1), channels, 1, bias=False),
            norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm(channels), nn.GELU(),
        )


class BoundaryDistributionHead(nn.Module):
    """LocNet-inspired, class-agnostic distributions for L/T/R/B boundaries.

    Axis pooling retains horizontal/vertical position. A proposal-centered
    Gaussian log-prior plus zero-initialized residual logits gives an identity
    refinement at initialization. The prior expectation is subtracted during
    decoding, including at clipped image borders.
    """
    def __init__(self, channels, bins, sigma):
        super().__init__()
        self.bins, self.sigma = bins, sigma
        self.x_head = nn.Sequential(nn.Conv1d(channels * 2, channels, 3, padding=1),
                                    nn.GELU(), nn.Conv1d(channels, 2, 1))
        self.y_head = nn.Sequential(nn.Conv1d(channels * 2, channels, 3, padding=1),
                                    nn.GELU(), nn.Conv1d(channels, 2, 1))
        for head in (self.x_head, self.y_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer("positions", torch.linspace(0, 1, bins))

    def forward(self, features, proposal_edges):
        x = torch.cat((features.mean(dim=2), features.amax(dim=2)), dim=1)
        y = torch.cat((features.mean(dim=3), features.amax(dim=3)), dim=1)
        x = F.interpolate(self.x_head(x), self.bins, mode="linear", align_corners=True)
        y = F.interpolate(self.y_head(y), self.bins, mode="linear", align_corners=True)
        residual = torch.stack((x[:, 0], y[:, 0], x[:, 1], y[:, 1]), dim=1).float()
        positions = self.positions.float()
        prior = -0.5 * ((positions[None, None, :] - proposal_edges.float()[:, :, None]) / self.sigma).square()
        logits = prior + residual
        prior_mean = (prior.softmax(-1) * positions).sum(-1)
        predicted_mean = (logits.softmax(-1) * positions).sum(-1)
        edges = proposal_edges.float() + predicted_mean - prior_mean
        return {"edges": edges.clamp(0, 1), "boundary_logits": logits,
                "proposal_edges": proposal_edges.float(), "prior_mean": prior_mean}


class RegressionHead(nn.Module):
    """Coordinate-offset head for the highres configuration."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(channels, channels), nn.GELU(),
                                 nn.Linear(channels, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, proposal_edges):
        delta = self.net(features).float().tanh() * 0.25
        return {"edges": (proposal_edges.float() + delta).clamp(0, 1)}
