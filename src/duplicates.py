"""Exact and near-duplicate groups over all rows (PROTOCOL.md Section 2.5).

Exact link: identical sha1 of the decoded RGB array. Near link: 64-bit dHash
Hamming distance <= max_hamming. Groups are the connected components of the
union of both link types (union-find), over all rows regardless of split,
label, or generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def dhash_to_uint64(hex_values) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_values], dtype=np.uint64)


def near_pairs(hashes: np.ndarray, max_hamming: int, chunk: int = 2048) -> np.ndarray:
    """All index pairs (i < j) with popcount(h_i XOR h_j) <= max_hamming."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    out = []
    for s in range(0, len(hashes), chunk):
        block = hashes[s:s + chunk]
        dist = np.bitwise_count(block[:, None] ^ hashes[None, :])
        i, j = np.nonzero(dist <= max_hamming)
        i = i + s
        keep = i < j
        out.append(np.stack([i[keep], j[keep]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2), dtype=np.int64)


def duplicate_groups(df: pd.DataFrame, max_hamming: int = 4) -> pd.DataFrame:
    """Return a frame aligned with df: dup_group (int), dup_group_size, and
    flags for groups crossing splits or labels, plus per-row link counts."""
    n = len(df)
    uf = UnionFind(n)
    n_exact = np.zeros(n, dtype=np.int64)
    n_near = np.zeros(n, dtype=np.int64)

    sha = df["rgb_sha1"].to_numpy()
    for _, idx in pd.Series(np.arange(n)).groupby(sha).groups.items():
        idx = np.asarray(idx)
        if len(idx) > 1:
            n_exact[idx] += len(idx) - 1
            for k in idx[1:]:
                uf.union(int(idx[0]), int(k))

    pairs = near_pairs(dhash_to_uint64(df["dhash"]), max_hamming)
    for i, j in pairs:
        uf.union(int(i), int(j))
    np.add.at(n_near, pairs[:, 0], 1)
    np.add.at(n_near, pairs[:, 1], 1)

    roots = np.array([uf.find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    out = pd.DataFrame({"dup_group": group, "n_exact_links": n_exact, "n_near_links": n_near}, index=df.index)
    g = pd.DataFrame({"group": group, "split": df["split"].to_numpy(), "label": df["label"].to_numpy()})
    size = g.groupby("group")["split"].transform("size").to_numpy()
    n_split = g.groupby("group")["split"].transform("nunique").to_numpy()
    n_label = g.groupby("group")["label"].transform("nunique").to_numpy()
    out["dup_group_size"] = size
    out["group_crosses_split"] = n_split > 1
    out["group_crosses_label"] = n_label > 1
    return out
