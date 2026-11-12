#!/usr/bin/env python3
"""
Convert the shared float32 SNN weights into the int8 / sparse-CSR binary that
the C++ fusion core loads.

The conversion is deliberately explicit:
  * every layer's weights are quantised to int8 with a per-layer symmetric
    scale;
  * a weight is *dropped* (treated as a structural zero) only when the float
    weight is exactly 0.0 -- the random initialisation has no exact zeros, so
    the sparse structure is created deliberately below;
  * the sparsity pattern is deterministic: for each row a fixed fraction of
    the smallest-magnitude weights is pruned to zero, which is what "sparse
    arrays" means for this network and is applied identically to every
    configuration (the dense backends use the same pruned matrix, so all
    configurations still evaluate the same function).

The report records: nnz per layer, the int8 payload size, and the size of the
sparse index structure.
"""

import os
import json
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sparsify(w: np.ndarray, density: float, seed: int = 7) -> np.ndarray:
    """Deterministically prune a fraction of each row's weights to zero."""
    rng = np.random.default_rng(seed + w.shape[0])
    out = w.copy()
    n_out, n_in = w.shape
    for r in range(n_out):
        k = int(round(n_in * (1.0 - density)))
        if k <= 0:
            continue
        # prune the k smallest-magnitude entries of this row
        idx = np.argsort(np.abs(out[r]))[:k]
        out[r, idx] = 0.0
    return out


def write_layer(f, w: np.ndarray, b: np.ndarray, density: float, deq_out: dict, key: str):
    """Write one layer: dims, scale, bias, CSR structure, int8 values.

    `w` has shape (n_in, n_out).  The CSR is built over the TRANSPOSED
    matrix, i.e. one row per output unit, which is the layout the C++ kernel
    consumes directly (x[col] * val accumulated into y[row]).
    """
    n_in, n_out = w.shape
    wt = np.ascontiguousarray(w.T)          # (n_out, n_in)
    sp = sparsify(wt, density)
    amax = np.abs(sp).max()
    scale = float(amax / 127.0) if amax > 0 else 1.0
    q = np.rint(sp / scale).clip(-127, 127).astype(np.int8)

    row_ptr = [0]
    cols = []
    vals = []
    for r in range(n_out):
        nz = np.nonzero(q[r])[0]
        cols.extend(nz.tolist())
        vals.extend(q[r][nz].tolist())
        row_ptr.append(len(cols))

    f.write(struct.pack("<i", n_in))
    f.write(struct.pack("<i", n_out))
    f.write(struct.pack("<f", scale))
    f.write(b.astype(np.float32).tobytes())
    f.write(struct.pack("<i", len(row_ptr)))
    f.write(struct.pack("<i", len(cols)))
    f.write(np.asarray(row_ptr, dtype=np.int32).tobytes())
    f.write(np.asarray(cols, dtype=np.int32).tobytes())
    f.write(np.asarray(vals, dtype=np.int8).tobytes())

    # the de-quantised sparse matrix (stored back as (n_in, n_out) so the dense
    # backends can use it with x @ W), so that the dense backends evaluate
    # exactly the same function as the int8/CSR core
    deq_out[key] = np.ascontiguousarray((q.astype(np.float32) * scale).T)
    return {"n_in": n_in, "n_out": n_out, "nnz": len(cols),
            "density": len(cols) / (n_in * n_out), "scale": scale,
            "int8_bytes": len(cols),
            "index_bytes": len(cols) * 4 + (n_out + 1) * 4,
            "bias_bytes": n_out * 4}


def main():
    import workload as WL
    wpath = os.path.join(HERE, "weights.npz")
    if not os.path.exists(wpath):
        WL.export_weights(wpath)
    W = WL.load_weights(wpath)

    density = 0.75          # 75% of the weights are kept
    out_bin = os.path.join(HERE, "fusion_weights_int8_sparse.bin")
    deq = {}
    with open(out_bin, "wb") as f:
        info = []
        info.append(write_layer(f, W["w1"], W["b1"], density, deq, "w1"))
        info.append(write_layer(f, W["w2"], W["b2"], density, deq, "w2"))
        info.append(write_layer(f, W["w3"], W["b3"], density, deq, "w3"))

    # every backend (dense PyTorch / TensorRT / C++ int8 CSR) must evaluate the
    # SAME function, so the dense backends use the de-quantised sparse matrix
    np.savez(os.path.join(HERE, "weights_quantised.npz"),
             w1=deq["w1"], w2=deq["w2"], w3=deq["w3"],
             b1=W["b1"], b2=W["b2"], b3=W["b3"])

    total_nnz = sum(i["nnz"] for i in info)
    total_dense = sum(i["n_in"] * i["n_out"] for i in info)
    report = {
        "binary": out_bin,
        "density": density,
        "layers": info,
        "total_nnz": total_nnz,
        "total_dense_weights": total_dense,
        "sparsity_percent": (1 - total_nnz / total_dense) * 100.0,
        "int8_weight_bytes": total_nnz,
        "int32_index_bytes": sum(i["index_bytes"] for i in info),
        "bias_bytes": sum(i["bias_bytes"] for i in info),
        "total_bytes": total_nnz + sum(i["index_bytes"] for i in info) + sum(i["bias_bytes"] for i in info),
        "note": ("This is the on-CPU representation of the FUSION CORE only. "
                 "It is not the Loihi 2 on-chip footprint; that number comes "
                 "from the Lava/Loihi compiler output (see lookups/)."),
    }
    with open(os.path.join(HERE, "..", "results", "cpp_weights_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
