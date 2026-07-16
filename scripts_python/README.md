# Python scripts

Python utilities for converting and validating SRNet model weights between
PyTorch and the Julia/Flux reimplementation in `src/Experiments/src/srnet.jl`.

## Requirements

- Python ≥ 3.10
- [Poetry](https://python-poetry.org/) for dependency management

## Setup

Run the preparation script once to download the pretrained weights, clone the
reference implementation, and install all Python dependencies:

```bash
cd scripts_python
bash prepare.sh
```

This will:
1. Download `JIN_SRNet.pt` from [dde.binghamton.edu](https://dde.binghamton.edu/download/feature_extractors/) into `data/`
2. Clone [brijeshiitg/Pytorch-implementation-of-SRNet](https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet) into `vendor/srnet/`
3. Run `poetry install`

## Usage

### Step 1 — Export weights and generate parity reference

```bash
cd scripts_python
poetry run python prepare_srnet_weights.py \
    --input  ../data/JIN_SRNet.pt \
    --output ../data/pretrained/srnet.h5
```

This produces:
- `data/pretrained/srnet.h5` — 80 weight tensors (full model: backbone + fc head,
  `n_out=2`). This is the file loaded at training time when `pretrained = true`.
- `data/pretrained/srnet.h5.io.h5` — reference input/output pair for the Julia
  parity test.

Optionally pass `--images img1.png img2.png` to use real images instead of the
default random tensors for the reference I/O.

### Step 2 — Verify the Julia implementation

```bash
julia scripts/test_srnet.jl
```

A successful run prints:

```
PASS — outputs agree within tolerance (0.0001)
```

### Step 3 — Use in training

```toml
[model]
type = "SRNet"
pretrained = true
in_channels = 3
n_out = 2
```

## File layout

```
scripts_python/
├── prepare.sh                  # one-time setup (download weights + poetry install)
├── pyproject.toml              # Poetry dependency manifest
├── prepare_srnet_weights.py    # checkpoint → HDF5 + reference I/O
└── vendor/
    └── srnet/                  # cloned brijeshiitg/Pytorch-implementation-of-SRNet
                                # (git-ignored, populated by prepare.sh)
```
