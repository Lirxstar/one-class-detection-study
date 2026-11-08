"""Frozen DINOv2 backbone, offline, and two-stage class-token features (PROTOCOL.md Section 4)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

import config
from config import ROOT


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_backbone(role: str = "primary", verify_sha: bool = True) -> torch.nn.Module:
    """Build the model from the local clone and load the pinned weights with strict=True."""
    enc = config.load("core")["encoder"]
    spec = enc[role]
    torch.set_num_threads(int(enc["torch_num_threads"]))
    weights = ROOT / spec["weights"]
    if verify_sha and sha256_file(weights) != spec["weights_sha256"]:
        raise RuntimeError(f"weight sha256 mismatch for {weights}")
    model = torch.hub.load(str(ROOT / spec["repo_dir"]), spec["name"], source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def to_input(views: np.ndarray) -> torch.Tensor:
    """uint8 (N, 224, 224, 3) -> normalized float32 (N, 3, 224, 224)."""
    if views.dtype != np.uint8 or views.ndim != 4 or views.shape[-1] != 3:
        raise ValueError(f"expected uint8 (N,H,W,3), got {views.dtype} {views.shape}")
    norm = config.load("core")["encoder"]["normalize"]
    x = torch.from_numpy(np.ascontiguousarray(views)).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(norm["std"], dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract(model: torch.nn.Module, views: np.ndarray, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass per batch; returns (h_pre, h_post) as float32 arrays.

    h_pre  = forward_features(x)["x_prenorm"][:, 0, :]   (M0 only)
    h_post = forward_features(x)["x_norm_clstoken"]       (B1, O1, O2 after L2 norm)
    """
    if batch_size is None:
        batch_size = int(config.load("core")["encoder"]["batch_size"])
    pre, post = [], []
    for i in range(0, len(views), batch_size):
        chunk = np.asarray(views[i:i + batch_size])
        n = len(chunk)
        if n < batch_size:  # zero-pad so every forward pass has the same shape; padded rows discarded
            chunk = np.concatenate([chunk, np.zeros((batch_size - n, *chunk.shape[1:]), np.uint8)])
        out = model.forward_features(to_input(chunk))
        pre.append(out["x_prenorm"][:n, 0, :].numpy().astype(np.float32, copy=True))
        post.append(out["x_norm_clstoken"][:n].numpy().astype(np.float32, copy=True))
    return np.concatenate(pre), np.concatenate(post)


def l2_normalize(h_post: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h_post, dtype=np.float64)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)
