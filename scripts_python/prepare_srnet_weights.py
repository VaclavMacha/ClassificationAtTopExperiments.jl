"""
Converts the JIN pretrained SRNet checkpoint (JIN_SRNet.pt from
https://dde.binghamton.edu/download/feature_extractors/) into an HDF5 file
loadable by load_srnet_weights! in the Julia Experiments framework.

All weights are exported — backbone AND the fc classification head (n_out=2).
A reference I/O file is also written so that scripts/test_srnet.jl can verify
that the Julia model produces identical outputs to this PyTorch model.

The checkpoint was saved as a full model object, so the original SRNet codebase
does not need to be installed — a stub unpickler handles the deserialization.

Setup:
    cd scripts_python && poetry install

Usage:
    uv run python prepare_srnet_weights.py \\
        --input  path/to/JIN_SRNet.pt \\
        --output ../data/pretrained/srnet.h5 \\
        [--images img1.png img2.png ...]

Output:
    <output>         HDF5 with 80 weight tensors (backbone + fc head)
    <output>.io.h5   HDF5 with reference input/output for Julia parity test

Then verify with Julia:
    julia scripts/test_srnet.jl
"""

import argparse
import os
import pickle
import types

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Robust loader — stubs any missing class (checkpoint saved as full model)
# ---------------------------------------------------------------------------


class _Stub(nn.Module):
    def __setstate__(self, state: dict) -> None:
        if "_parameters" not in self.__dict__:
            nn.Module.__init__(self)
        self.__dict__.update(state)


class _RobustUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            return _Stub


_robust_pickle = types.ModuleType("_robust_pickle")
_robust_pickle.__dict__.update(pickle.__dict__)
_robust_pickle.Unpickler = _RobustUnpickler  # type: ignore[attr-defined]


def load_state_dict(path: str) -> dict:
    ckpt = torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_robust_pickle
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt.state_dict()


# ---------------------------------------------------------------------------
# Minimal PyTorch SRNet matching JIN checkpoint key names exactly.
# Used only for the forward pass that generates reference I/O.
# ---------------------------------------------------------------------------


class _Layer1(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class Block1Unit(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)))


class Block2Unit(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.layer1 = _Layer1(c, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.layer1(x))) + x


class Block3Unit(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.resconv = nn.Conv2d(in_c, out_c, 1, stride=2, bias=False)
        self.resnorm = nn.BatchNorm2d(out_c)
        self.layer1 = _Layer1(in_c, out_c)
        self.conv = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.resnorm(self.resconv(x))
        main = F.avg_pool2d(
            self.norm(self.conv(self.layer1(x))), 3, stride=2, padding=1
        )
        return shortcut + main


class Block4Unit(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.layer1 = _Layer1(in_c, out_c)
        self.conv = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.layer1(x)))


class JINSRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(Block1Unit(3, 64), Block1Unit(64, 16))
        self.block2 = nn.Sequential(*[Block2Unit(16) for _ in range(5)])
        self.block3 = nn.Sequential(
            Block3Unit(16, 16),
            Block3Unit(16, 64),
            Block3Unit(64, 128),
            Block3Unit(128, 256),
        )
        self.block4 = nn.Sequential(Block4Unit(256, 512))
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.fc(x.mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# Weight order — full model in Flux.params traversal order.
# BatchNorm: β (bias) before γ (weight) — Flux field declaration order.
# Conv (out,in,H,W) → (H,W,in,out) permutation for Flux.
# fc.weight has the same (n_out, 512) layout in both frameworks — no permute.
# ---------------------------------------------------------------------------
WEIGHT_ORDER = (
    ("block1.0.conv.weight", True),
    ("block1.0.norm.bias", False),
    ("block1.0.norm.weight", False),
    ("block1.1.conv.weight", True),
    ("block1.1.norm.bias", False),
    ("block1.1.norm.weight", False),
    ("block2.0.layer1.conv.weight", True),
    ("block2.0.layer1.norm.bias", False),
    ("block2.0.layer1.norm.weight", False),
    ("block2.0.conv.weight", True),
    ("block2.0.norm.bias", False),
    ("block2.0.norm.weight", False),
    ("block2.1.layer1.conv.weight", True),
    ("block2.1.layer1.norm.bias", False),
    ("block2.1.layer1.norm.weight", False),
    ("block2.1.conv.weight", True),
    ("block2.1.norm.bias", False),
    ("block2.1.norm.weight", False),
    ("block2.2.layer1.conv.weight", True),
    ("block2.2.layer1.norm.bias", False),
    ("block2.2.layer1.norm.weight", False),
    ("block2.2.conv.weight", True),
    ("block2.2.norm.bias", False),
    ("block2.2.norm.weight", False),
    ("block2.3.layer1.conv.weight", True),
    ("block2.3.layer1.norm.bias", False),
    ("block2.3.layer1.norm.weight", False),
    ("block2.3.conv.weight", True),
    ("block2.3.norm.bias", False),
    ("block2.3.norm.weight", False),
    ("block2.4.layer1.conv.weight", True),
    ("block2.4.layer1.norm.bias", False),
    ("block2.4.layer1.norm.weight", False),
    ("block2.4.conv.weight", True),
    ("block2.4.norm.bias", False),
    ("block2.4.norm.weight", False),
    ("block3.0.resconv.weight", True),
    ("block3.0.resnorm.bias", False),
    ("block3.0.resnorm.weight", False),
    ("block3.0.layer1.conv.weight", True),
    ("block3.0.layer1.norm.bias", False),
    ("block3.0.layer1.norm.weight", False),
    ("block3.0.conv.weight", True),
    ("block3.0.norm.bias", False),
    ("block3.0.norm.weight", False),
    ("block3.1.resconv.weight", True),
    ("block3.1.resnorm.bias", False),
    ("block3.1.resnorm.weight", False),
    ("block3.1.layer1.conv.weight", True),
    ("block3.1.layer1.norm.bias", False),
    ("block3.1.layer1.norm.weight", False),
    ("block3.1.conv.weight", True),
    ("block3.1.norm.bias", False),
    ("block3.1.norm.weight", False),
    ("block3.2.resconv.weight", True),
    ("block3.2.resnorm.bias", False),
    ("block3.2.resnorm.weight", False),
    ("block3.2.layer1.conv.weight", True),
    ("block3.2.layer1.norm.bias", False),
    ("block3.2.layer1.norm.weight", False),
    ("block3.2.conv.weight", True),
    ("block3.2.norm.bias", False),
    ("block3.2.norm.weight", False),
    ("block3.3.resconv.weight", True),
    ("block3.3.resnorm.bias", False),
    ("block3.3.resnorm.weight", False),
    ("block3.3.layer1.conv.weight", True),
    ("block3.3.layer1.norm.bias", False),
    ("block3.3.layer1.norm.weight", False),
    ("block3.3.conv.weight", True),
    ("block3.3.norm.bias", False),
    ("block3.3.norm.weight", False),
    ("block4.0.layer1.conv.weight", True),
    ("block4.0.layer1.norm.bias", False),
    ("block4.0.layer1.norm.weight", False),
    ("block4.0.conv.weight", True),
    ("block4.0.norm.bias", False),
    ("block4.0.norm.weight", False),
    ("fc.weight", False),
    ("fc.bias", False),
)


def convert(input_path: str, output_path: str, image_paths: list[str]) -> None:
    state_dict = load_state_dict(input_path)

    missing = [k for k, _ in WEIGHT_ORDER if k not in state_dict]
    if missing:
        print("ERROR: missing keys in checkpoint:")
        for k in missing:
            print(f"  {k}")
        import sys

        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    param_keys = {k for k, _ in WEIGHT_ORDER}

    # Export weights
    with h5py.File(output_path, "w") as f:
        for idx, (key, conv_permute) in enumerate(WEIGHT_ORDER):
            arr = state_dict[key].numpy().astype(np.float32)
            if conv_permute:
                arr = np.transpose(arr, (3, 2, 1, 0))
            f.create_dataset(f"{idx:04d}", data=arr)
            print(f"  [{idx:04d}] {key:55s}  {arr.shape}")
    print(f"\nSaved {len(WEIGHT_ORDER)} tensors (backbone + fc) → {output_path}")

    # Build model and load weights for reference I/O generation
    model = JINSRNet()
    filtered_sd = {k: v for k, v in state_dict.items() if k in param_keys}
    model.load_state_dict(filtered_sd, strict=False)
    model.train()  # train mode: batch stats, so Julia (also train mode) matches

    if image_paths:
        import imageio

        frames = []
        for p in image_paths:
            img = imageio.imread(p)
            t = torch.tensor(np.array(img), dtype=torch.float32)
            if t.ndim == 2:
                t = t.unsqueeze(-1).expand(-1, -1, 3)
            elif t.shape[-1] == 4:
                t = t[..., :3]
            frames.append(t.permute(2, 0, 1))
        x = torch.stack(frames)
        print(f"Loaded {len(image_paths)} image(s), shape: {tuple(x.shape)}")
    else:
        torch.manual_seed(42)
        x = torch.randn(2, 3, 256, 256)
        print("No images provided — using 2 random 256×256 RGB tensors (seed=42).")

    with torch.no_grad():
        y = model(x)

    io_path = output_path + ".io.h5"
    x_flux = np.transpose(x.numpy(), (2, 3, 1, 0))  # (H,W,C,N)
    y_flux = y.numpy().T  # (n_out,N)

    with h5py.File(io_path, "w") as f:
        f.create_dataset("input", data=x_flux)
        f.create_dataset("output", data=y_flux)

    print(f"Reference logits:\n{y.numpy()}")
    print(f"Saved reference I/O → {io_path}")
    print("\nVerify with Julia:\n  julia scripts/test_srnet.jl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export JIN SRNet weights to HDF5 and generate Julia parity reference."
    )
    parser.add_argument("--input", required=True, help="Path to JIN_SRNet.pt")
    parser.add_argument("--output", required=True, help="Destination .h5 file")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[],
        help="Optional RGB images for reference I/O (PNG/JPEG)",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.images)
