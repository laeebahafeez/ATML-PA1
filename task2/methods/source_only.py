"""Source-only ERM: plain cross-entropy over the three labeled source
domains, with domain-balanced batches (handled upstream by
SourceTargetBatchIterator with use_target=False). This is also the exact
checkpoint reused, unchanged, as Task 3's ERM baseline."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_loss(backbone, head, images, labels, domain_ids, n_source,
                  progress, cfg, discriminator=None) -> Tuple[torch.Tensor, Dict]:
    feats = backbone(images[:n_source])
    logits = head(feats)
    loss = F.cross_entropy(logits, labels)
    return loss, {"cls_loss": loss.item()}
