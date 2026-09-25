"""
Adaptive Instance Normalization (AdaIN) style transfer.

Reference: Huang & Belongie, "Arbitrary Style Transfer in Real-Time with
Adaptive Instance Normalization" (ICCV 2017).

This module implements the standard AdaIN architecture:
    encoder  = VGG19 truncated to relu4_1 (ImageNet-pretrained; downloaded
               automatically via torchvision, no manual step needed)
    AdaIN    = align content-feature channel statistics to style-feature
               channel statistics
    decoder  = mirror of the encoder, trained to invert AdaIN-normalized
               features back into an image

The encoder needs no external file. The decoder, however, is a generative
network that must be *trained* to invert VGG features into realistic
pixels -- that is a separate, non-trivial training run that is not the
point of this assignment. We therefore load the decoder from the
publicly released checkpoint that accompanies the original AdaIN
implementation, exactly as permitted by the assignment ("Use AdaIN or
another public style-transfer implementation").

    Public implementation used: https://github.com/naoto0804/pytorch-AdaIN
    Required file: decoder.pth (place at task1/checkpoints/decoder.pth)

See task1/README.md for the download step. If the checkpoint is not
found, `AdaINStyleTransfer` raises a clear error rather than silently
producing garbage images -- an untrained decoder would corrupt every
downstream cue-conflict result.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.models as tv_models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _vgg_normalize(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def _vgg_denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def build_vgg_encoder() -> nn.Sequential:
    """VGG19 features truncated at relu4_1 (layer index 21 in torchvision's
    features Sequential), matching the original AdaIN paper's encoder."""
    vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
    encoder = nn.Sequential(*list(vgg.children())[:21])  # up to and including relu4_1
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder


def build_decoder() -> nn.Sequential:
    """Mirror of the relu4_1 VGG encoder, matching naoto0804/pytorch-AdaIN's
    decoder architecture so its public pretrained weights load directly."""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, 3), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, 3), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, 3), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, 3), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, 3),
    )


def adaptive_instance_norm(content_feat: torch.Tensor, style_feat: torch.Tensor,
                            eps: float = 1e-5) -> torch.Tensor:
    """Align the per-channel mean/std of content_feat to style_feat's."""
    b, c = content_feat.shape[:2]
    c_mean = content_feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    c_std = content_feat.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
    s_mean = style_feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    s_std = style_feat.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


class AdaINStyleTransfer(nn.Module):
    """Full content -> stylized-image pipeline used for Task 1 cue conflicts."""

    def __init__(self, decoder_weights: str):
        super().__init__()
        if not os.path.isfile(decoder_weights):
            raise FileNotFoundError(
                f"AdaIN decoder weights not found at {decoder_weights!r}.\n"
                "Download decoder.pth from the public reference implementation "
                "(naoto0804/pytorch-AdaIN, linked in task1/README.md) and place "
                "it at that path. An untrained decoder produces meaningless "
                "images and must not be used to generate cue conflicts."
            )
        self.encoder = build_vgg_encoder()
        self.decoder = build_decoder()
        state = torch.load(decoder_weights, map_location="cpu")
        self.decoder.load_state_dict(state)
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def stylize(self, content01: torch.Tensor, style01: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        """
        content01, style01: [B, 3, 224, 224] in [0, 1].
        alpha in [0, 1] interpolates between the content feature (alpha=0)
        and the fully style-normalized feature (alpha=1); the assignment's
        "style strength" experimental-design choice maps directly to alpha.
        Returns a [0, 1] RGB tensor of the same shape.
        """
        c_feat = self.encoder(_vgg_normalize(content01))
        s_feat = self.encoder(_vgg_normalize(style01))
        t = adaptive_instance_norm(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat
        out = self.decoder(t)
        return _vgg_denormalize(out)
