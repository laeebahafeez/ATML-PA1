"""
Common image interventions for Task 1.

Every transform here operates on a *common* representation: a float tensor
of shape [B, 3, H, W] (or [3, H, W]) with values in [0, 1], BEFORE any
model-specific normalization. This is what lets every backbone receive
byte-for-byte identical transformed pixels, per the assignment's
requirement that "every model must receive the same clean and transformed
images."

Included:
    - to_grayscale            : required color intervention (removes color)
    - hue_rotate               : chosen additional color intervention
                                  (changes color, preserves luminance/geometry)
    - translate_reflect /
      translate_cardinal       : reflection-padded shift by (dx, dy) pixels
    - patch_shuffle            : deterministic non-identity 4x4 grid permutation
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Color interventions
# --------------------------------------------------------------------------- #

def to_grayscale(x01: torch.Tensor) -> torch.Tensor:
    """Convert to grayscale, replicated across 3 channels.

    Removes all chromatic information while exactly preserving object
    geometry (edges, silhouette, spatial layout) -- the required "removing
    color" intervention.
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=x01.device, dtype=x01.dtype)
    weights = weights.view(1, 3, 1, 1)
    gray = (x01 * weights).sum(dim=1, keepdim=True)
    return gray.expand(-1, 3, -1, -1).clone()


def _rgb_to_hsv(x01: torch.Tensor) -> torch.Tensor:
    r, g, b = x01[:, 0], x01[:, 1], x01[:, 2]
    maxc, _ = x01.max(dim=1)
    minc, _ = x01.min(dim=1)
    v = maxc
    deltac = maxc - minc
    s = torch.where(maxc > 0, deltac / maxc.clamp(min=1e-8), torch.zeros_like(maxc))

    deltac_safe = deltac.clamp(min=1e-8)
    rc = (maxc - r) / deltac_safe
    gc = (maxc - g) / deltac_safe
    bc = (maxc - b) / deltac_safe

    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(deltac == 0, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.remainder(6).long()

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    rgb_options = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    for k, (rr, gg, bb) in enumerate(rgb_options):
        mask = (i == k)
        r = torch.where(mask, rr, r)
        g = torch.where(mask, gg, g)
        b = torch.where(mask, bb, b)
    return torch.stack([r, g, b], dim=1).clamp(0, 1)


def hue_rotate(x01: torch.Tensor, degrees: float = 90.0) -> torch.Tensor:
    """Rotate hue by a fixed amount in HSV space.

    Preserves saturation, value (brightness), and exact object geometry;
    changes only which color each surface appears as. This is the chosen
    "additional color intervention" that tests sensitivity to *changing*
    color, complementing grayscale's test of *removing* it.
    """
    hsv = _rgb_to_hsv(x01)
    hsv = hsv.clone()
    hsv[:, 0] = (hsv[:, 0] + degrees / 360.0) % 1.0
    return _hsv_to_rgb(hsv)


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #

CARDINAL_DIRECTIONS: Dict[str, Sequence[int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def translate_reflect(x01: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift by (dx, dy) pixels using reflection padding + a shifted crop.

    Output has the same H x W as the input; no black border is introduced,
    isolating the effect of displacement from the effect of a new border
    artifact.
    """
    if dx == 0 and dy == 0:
        return x01.clone()
    _, _, h, w = x01.shape
    pad = max(abs(dx), abs(dy))
    xp = F.pad(x01, (pad, pad, pad, pad), mode="reflect")
    top = pad - dy
    left = pad - dx
    return xp[:, :, top:top + h, left:left + w]


def translate_cardinal(x01: torch.Tensor, pixels: int, direction: str) -> torch.Tensor:
    dxu, dyu = CARDINAL_DIRECTIONS[direction]
    return translate_reflect(x01, dx=dxu * pixels, dy=dyu * pixels)


# --------------------------------------------------------------------------- #
# Patch shuffle
# --------------------------------------------------------------------------- #

def _random_non_identity_permutation(n: int, generator: torch.Generator) -> torch.Tensor:
    identity = torch.arange(n)
    perm = torch.randperm(n, generator=generator)
    while torch.equal(perm, identity):
        perm = torch.randperm(n, generator=generator)
    return perm


def patch_shuffle(
    x01: torch.Tensor,
    grid: int = 4,
    seed: int = 6304,
    image_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Divide each image into a grid x grid pixel-space grid and apply one
    non-identity patch permutation per image.

    `image_ids` should be a stable, globally unique identifier per image
    (e.g. the dataset index) so that regenerating the shuffle for the same
    image -- regardless of which batch or model pass it appears in --
    always yields the identical permutation ("reuse exactly the same
    shuffled images across models").
    """
    b, c, h, w = x01.shape
    assert h % grid == 0 and w % grid == 0, "image size must be divisible by grid"
    if image_ids is None:
        image_ids = list(range(b))
    assert len(image_ids) == b

    ph, pw = h // grid, w // grid
    n_patches = grid * grid
    out = torch.empty_like(x01)

    for i in range(b):
        gen = torch.Generator().manual_seed(seed + int(image_ids[i]))
        # generated on CPU (torch.Generator() defaults to CPU); must move to
        # x01's device before using it to index a possibly-CUDA tensor below,
        # or PyTorch raises a device-mismatch error identical to the CLIP
        # tokenizer bug fixed in models/backbones.py.
        perm = _random_non_identity_permutation(n_patches, gen).to(x01.device)

        patches = x01[i].unfold(1, ph, ph).unfold(2, pw, pw)  # [C, grid, grid, ph, pw]
        patches = patches.contiguous().view(c, n_patches, ph, pw)
        shuffled = patches[:, perm, :, :]
        shuffled = shuffled.view(c, grid, grid, ph, pw)
        shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
        out[i] = shuffled
    return out
