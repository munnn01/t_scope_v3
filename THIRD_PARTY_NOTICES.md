# Attribution and research lineage

T-SCOPE v3 continues the T-SCOPE v2 source and historical artifacts from
https://github.com/wagur1/Tscope-v2 (base commit
919dc012f512c05f268c256e39c3615f801d6c8b).
Git history is retained when publishing this continuation.

The conditioning basis, FiLM image editor, foreground/background decomposition,
and frozen-detector supervision are based on:

Silvia Dani, Leonardo Galteri, Marco Bertini. **SCOPE: Semantic COdec Pre-Editing
for Object-Preserving Compression**, arXiv:2610.39778v1, 2026.
https://arxiv.org/abs/2610.39778

The authors' implementation is available at https://github.com/miccunifi/SCOPE
under its own GPL-3.0 license. `models/scope_reference.py` is a newly written,
paper-derived adaptation to this repository's MOT17/x265 interface; it does not
vendor the authors' source or original weights and is not an official reproduction
of their COCO experiments. The v3 shared objective is also an adaptation.

MOT17, pretrained torchvision/Ultralytics models, FFmpeg, x265, PyTorch, SciPy and
the Kaggle SDK remain subject to their respective licenses and dataset terms.
No MOT images, SCOPE PDF, account credentials or pretrained detector weights are
included in this repository. Historical `release_ckpts/` weights belong to v2;
they are not newly trained v3 models.
