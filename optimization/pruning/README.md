# P20 Pruning

This directory contains the P20 pruning-compatible module definition, pruned-model runtime script, and TensorRT export utility.

`pruned_modules.py` must remain available when loading checkpoints serialized with `C2fPruningFriendly`.
