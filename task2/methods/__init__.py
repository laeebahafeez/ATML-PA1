"""
Common interface every method module implements, so scripts/train.py can
run all four methods through one identical loop (per the assignment:
"all methods must run through the same training and evaluation pipeline"):

    compute_loss(backbone, head, images, labels, domain_ids, n_source,
                 progress, cfg, discriminator=None) -> (total_loss, logs: dict)

Inputs (all already on the training device):
    images:     [B, 3, 224, 224], the first `n_source` rows are labeled
                source examples (8 per source domain), any remaining rows
                are unlabeled target examples (adaptation methods only).
    labels:     [n_source] class labels for the source rows only.
    domain_ids: [B] integer domain id (0/1/2 = the three source domains in
                cfg.dataset.source_domains order, 3 = target).
    n_source:   int, number of source rows at the front of `images`.
    progress:   float in [0, 1], training progress (epoch fraction) used by
                the gradient-reversal schedule in DANN/CDAN.

Every method's classification loss is computed ONLY on the source rows;
target rows never contribute a label-supervised term, matching the
assignment's transductive-but-unsupervised UDA protocol.
"""
