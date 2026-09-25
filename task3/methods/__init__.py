"""
Common interface for Task 3's three methods (erm, dan_dg, sam).

Unlike Task 2's methods (which all share a single-backward-pass training
step and can return a scalar loss for train.py to call .backward() on),
SAM requires two full forward/backward passes and its own optimizer.step()
call, so Task 3's interface gives every method full control of the
optimizer step instead:

    train_step(backbone, head, images, labels, domain_ids, cfg, optimizer)
        -> logs: dict[str, float]

Each method's train_step is responsible for optimizer.zero_grad(),
computing whatever forward/backward pass(es) it needs, and calling
optimizer.step() itself, so train.py's epoch loop never branches on
method name. `domain_ids` takes values 0/1/2 for
[photo, art_painting, cartoon] respectively -- Task 3 batches never
contain a target/Sketch row (see shared/pacs_protocol.py's
SourceTargetBatchIterator with use_target=False).

erm.py's train_step is provided for completeness/testability, but the
assignment requires the ERM checkpoint to be Task 2's Source-only run
loaded unchanged rather than retrained -- see erm.py::load_source_only_checkpoint
and train.py's handling of method="erm".
"""
