"""
PROSER -- classifier + data placeholders (Zhou et al. 2021, "Learning
Placeholders for Open-Set Recognition").

Implementation notes / attribution (documented at length here, and again in
task4/README.md, since the assignment explicitly leaves exact loss
construction to "the paper/reference code"):

Classifier-placeholder loss (weight beta, assignment sets beta=1):
  L1 = CE(logits_full, y)
       -- ordinary cross-entropy over ALL K+num_dummy logits with the true
       label y. This is exactly "the correct known-class response should
       remain the largest" -- softmax over the full logit vector already
       forces y to outrank every dummy logit too, with no extra machinery.
  L2 = CE(concat(logits_known_with_y_masked_to_-inf, max(logits_dummy)), target=K)
       -- after excluding y (masked to -inf so it cannot win), collapse the
       five dummy logits to a single virtual "reject" column via max, and
       train that virtual column to outrank the remaining K-1 real classes.
       This is "once the correct class is excluded, encourage one of the
       dummy classifiers to become the strongest remaining response."
  L_classifier_placeholder = L1 + beta * L2

Data-placeholder loss (weight gamma, assignment sets gamma=0.1):
  For a manifold-mixed feature between two different-class examples (no
  ground-truth class of its own), collapse the dummy logits to one virtual
  "reject" column the same way, and train that virtual column to outrank
  ALL K known classes (no masking needed -- there is no true class to
  exclude):
  L_data_placeholder = CE(concat(logits_known, max(logits_dummy)), target=K)

Total per-batch loss (first half of the batch -> classifier placeholders,
second half -> data placeholders, per the assignment's explicit split):
  L = L_classifier_placeholder(first_half) + gamma * L_data_placeholder(second_half)

This "collapse dummies via max into one virtual column, then plain
cross-entropy" construction is a direct, literal reading of the assignment's
prose description of the mechanism; it is not copied from the official
PROSER codebase line-for-line (not available at build time), so minor
differences from the reference implementation's exact tensor bookkeeping are
possible. This is called out again in task4/README.md's attribution section.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from classifier_head import ClassifierHead  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from manifold_mixup import manifold_mixup_pairs  # noqa: E402


def classifier_placeholder_loss(all_logits: torch.Tensor, labels: torch.Tensor,
                                 num_classes: int, beta: float) -> torch.Tensor:
    B = all_logits.shape[0]
    known_logits = all_logits[:, :num_classes]
    dummy_logits = all_logits[:, num_classes:]

    l1 = F.cross_entropy(all_logits, labels)

    masked_known = known_logits.clone()
    masked_known.scatter_(1, labels.unsqueeze(1), float("-inf"))
    dummy_max = dummy_logits.max(dim=1, keepdim=True).values
    reduced = torch.cat([masked_known, dummy_max], dim=1)  # [B, K+1]
    target_dummy = torch.full((B,), num_classes, dtype=torch.long, device=all_logits.device)
    l2 = F.cross_entropy(reduced, target_dummy)

    return l1 + beta * l2


def data_placeholder_loss(mixed_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    B = mixed_logits.shape[0]
    known_logits = mixed_logits[:, :num_classes]
    dummy_logits = mixed_logits[:, num_classes:]
    dummy_max = dummy_logits.max(dim=1, keepdim=True).values
    reduced = torch.cat([known_logits, dummy_max], dim=1)  # [B, K+1]
    target_dummy = torch.full((B,), num_classes, dtype=torch.long, device=mixed_logits.device)
    return F.cross_entropy(reduced, target_dummy)


def placeholder_detection_score(all_logits, num_classes: int):
    """u(x): larger => more novel. "Combining the strongest dummy response
    with the known-class responses" (assignment's phrasing) as the margin by
    which the best dummy classifier beats the best known class -- a positive
    margin means the network itself would rather reject than commit to a
    known class.

    Operates on numpy arrays. The only real call site is evaluate_osr.py,
    which reads cached logits back from .npz files written by
    extract_outputs.py (always numpy, never torch) -- this is never called
    during training on live tensors, so numpy is the correct and only
    contract here (torch.Tensor.max uses `dim=`, numpy uses `axis=`; mixing
    the two APIs under one function was the bug fixed here)."""
    all_logits = np.asarray(all_logits)
    known_logits = all_logits[:, :num_classes]
    dummy_logits = all_logits[:, num_classes:]
    return dummy_logits.max(axis=1) - known_logits.max(axis=1)


def train_proser(cfg: dict, vanilla_checkpoint_path: str, prepare_data_fn,
                  build_model_bundle_fn, evaluate_accuracy_fn, device):
    proser_cfg = cfg["proser"]
    num_classes = cfg["dataset"]["num_classes"]
    num_dummy = proser_cfg["num_dummy_classifiers"]

    torch.manual_seed(proser_cfg["seed"])

    # PROSER fine-tunes with the SAME base augmentation as vanilla/gcsc training
    # (crop+pad+hflip only -- RandAugment is GCSC-specific, not part of PROSER's
    # recipe per the assignment).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import vanilla as vanilla_method  # noqa: E402
    train_transform = vanilla_method.build_transform(cfg)

    train_subset, val_subset, _, _ = prepare_data_fn(cfg, train_transform)
    train_loader = DataLoader(train_subset, batch_size=proser_cfg["batch_size"],
                               shuffle=True, num_workers=2, drop_last=True,
                               generator=torch.Generator().manual_seed(proser_cfg["seed"]))
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)

    backbone, head = build_model_bundle_fn(cfg, num_dummy=num_dummy)

    vanilla_ckpt = torch.load(vanilla_checkpoint_path, map_location=device)
    backbone.load_state_dict(vanilla_ckpt["backbone"])
    # Copy the known-class rows of Vanilla's fc weight/bias into the new,
    # wider head; the extra `num_dummy` rows keep their fresh random init.
    with torch.no_grad():
        head.fc.weight[:num_classes].copy_(vanilla_ckpt["head"]["fc.weight"])
        head.fc.bias[:num_classes].copy_(vanilla_ckpt["head"]["fc.bias"])

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=proser_cfg["lr"], momentum=proser_cfg["momentum"],
                                 weight_decay=proser_cfg["weight_decay"])
    epochs = proser_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(proser_cfg["seed"])

    beta = proser_cfg["beta_classifier_placeholder"]
    gamma = proser_cfg["gamma_data_placeholder"]
    mix_alpha = proser_cfg["mixup_beta_alpha"]
    mix_beta = proser_cfg["mixup_beta_beta"]

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        backbone.train()
        head.train()
        epoch_logs = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.shape[0]
            half = B // 2
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:], y[half:]

            optimizer.zero_grad()

            # --- first half: classifier placeholders (ordinary forward) ---
            feats1 = backbone(x1)
            logits1 = head(feats1)
            loss_cls_ph = classifier_placeholder_loss(logits1, y1, num_classes, beta)

            # --- second half: data placeholders (manifold mixup) ---
            if x2.shape[0] >= 2:
                images_i, images_j, _, lam = manifold_mixup_pairs(x2, y2, mix_alpha, mix_beta, gen)
                h_i = backbone.forward_pre(images_i)
                h_j = backbone.forward_pre(images_j)
                h_mix = lam * h_i + (1 - lam) * h_j
                feats_mix = backbone.forward_post(h_mix)
                logits_mix = head(feats_mix)
                loss_data_ph = data_placeholder_loss(logits_mix, num_classes)
            else:
                loss_data_ph = torch.zeros((), device=device)

            loss = loss_cls_ph + gamma * loss_data_ph
            loss.backward()
            optimizer.step()
            epoch_logs.append({"loss_cls_ph": loss_cls_ph.item(),
                                "loss_data_ph": loss_data_ph.item(),
                                "total_loss": loss.item()})
        scheduler.step()

        val_acc = evaluate_accuracy_fn(backbone, head, val_loader, num_classes)
        mean_logs = {k: float(np.mean([l[k] for l in epoch_logs])) for k in epoch_logs[0]}
        history.append({"epoch": epoch, "val_accuracy": val_acc, **mean_logs})
        print(f"[proser] epoch {epoch:03d}  val_accuracy={val_acc:.4f}  "
              f"train_logs={mean_logs}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "epoch": epoch, "val_accuracy": val_acc,
            }

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    return backbone, head, best_state, history
