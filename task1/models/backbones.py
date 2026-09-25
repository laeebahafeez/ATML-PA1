"""
Backbone wrappers for Task 1 (Inductive Biases and Feature Representations).

Design contract (important for the whole pipeline):
    Every wrapper accepts a *common* un-normalized RGB tensor of shape
    [B, 3, 224, 224] with values in [0, 1], and is responsible for applying
    its own required normalization internally. This is what lets
    interventions (data/transforms.py, data/make_cue_conflicts.py) be
    written once and fed identically to all three backbones, satisfying:

        "Construct interventions on a common 224x224 RGB image before
         applying each model's required normalization."

Backbones and required representations (per assignment spec):
    - ResNet-50 (ResNet50_Weights.IMAGENET1K_V2):
          global-average-pooled 2048-d feature (output of layer4 + GAP)
    - ViT-B/16 (ViT_B_16_Weights.IMAGENET1K_V1):
          final [CLS] token, 768-d
    - OpenCLIP ViT-B-32 (pretrained="openai"):
          L2-normalized image embedding, 512-d

All backbones are frozen (no gradient ever flows into them); only the
attached nn.Linear head is trained.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    import open_clip
except ImportError:  # pragma: no cover - exercised only when open_clip is missing
    open_clip = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


class FrozenBackbone(nn.Module):
    """Base class: frozen feature extractor + trainable linear head."""

    feature_dim: int = -1

    def __init__(self):
        super().__init__()
        self.head: Optional[nn.Linear] = None

    def attach_head(self, num_classes: int) -> nn.Linear:
        self.head = nn.Linear(self.feature_dim, num_classes)
        return self.head

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: [B, 3, 224, 224] in [0, 1]. Returns frozen [B, feature_dim] features."""
        raise NotImplementedError

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("call attach_head(num_classes) before classifying")
        return self.head(feats)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        feats = self.features(x01)
        return self.logits_from_features(feats)

    def trainable_parameters(self):
        if self.head is None:
            raise RuntimeError("call attach_head(num_classes) first")
        return self.head.parameters()


class ResNet50Backbone(FrozenBackbone):
    feature_dim = 2048

    def __init__(self):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        net = tv_models.resnet50(weights=weights)
        # Everything up to (and including) layer4, i.e. before avgpool+fc.
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.stem.parameters():
            p.requires_grad_(False)
        self.stem.eval()

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        self.stem.eval()
        with torch.no_grad():
            x = _normalize(x01, IMAGENET_MEAN, IMAGENET_STD)
            x = self.stem(x)
            x = self.pool(x).flatten(1)
        return x


class ViTB16Backbone(FrozenBackbone):
    feature_dim = 768

    def __init__(self):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
        self.net = tv_models.vit_b_16(weights=weights)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            x = _normalize(x01, IMAGENET_MEAN, IMAGENET_STD)
            # Mirror torchvision's VisionTransformer.forward up to the
            # encoder output, stopping before the classification head so we
            # can read out the final [CLS] token.
            x = self.net._process_input(x)
            n = x.shape[0]
            cls_token = self.net.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.net.encoder(x)
            cls = x[:, 0]
        return cls


class CLIPViTB32Backbone(FrozenBackbone):
    feature_dim = 512

    def __init__(self):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip_torch is required for the CLIP backbone "
                "(pip install open_clip_torch)"
            )
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = model
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._text_features_cache = None
        self._cached_prompt_key = None

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = _normalize(x01, CLIP_MEAN, CLIP_STD)
            feats = self.model.encode_image(x)
            feats = F.normalize(feats, dim=-1)
        return feats

    def zero_shot_logits(
        self,
        x01: torch.Tensor,
        class_names: List[str],
        prompt_template: str = "a photo of a {class}.",
    ) -> torch.Tensor:
        """
        Scaled cosine-similarity logits (pre-softmax) for CLIP zero-shot
        classification with a fixed prompt, as required by the assignment
        ("compute confidence from the softmax over scaled class similarities").
        """
        cache_key = (tuple(class_names), prompt_template)
        if self._text_features_cache is None or self._cached_prompt_key != cache_key:
            prompts = [
                prompt_template.format(**{"class": c.replace("_", " ")})
                for c in class_names
            ]
            with torch.no_grad():
                # the tokenizer always returns CPU tensors, regardless of
                # which device the model itself lives on -- move them to
                # match x01 (and therefore the model) before encoding.
                tokens = self.tokenizer(prompts).to(x01.device)
                text_feats = self.model.encode_text(tokens)
                text_feats = F.normalize(text_feats, dim=-1)
            self._text_features_cache = text_feats
            self._cached_prompt_key = cache_key
        img_feats = self.features(x01)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feats @ self._text_features_cache.t()
        return logits


def build_backbone(name: str) -> FrozenBackbone:
    if name == "resnet50":
        return ResNet50Backbone()
    if name == "vit_b16":
        return ViTB16Backbone()
    if name == "clip_vitb32":
        return CLIPViTB32Backbone()
    raise ValueError(f"unknown backbone name: {name!r}")


BACKBONE_NAMES = ["resnet50", "vit_b16", "clip_vitb32"]
