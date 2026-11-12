#!/usr/bin/env python3
"""Full-bag S2/S3: quantisation effect + spike stats + decoded trajectory.

Compares float32-dense vs int8-dequantised-sparse on EVERY frame of the
recorded bag (2746 /scan messages), reporting mean +/- SD:
  - output spike mismatch rate (with threshold-tie attribution)
  - per-layer spike counts per frame
  - decoded odometry trajectories and their drift (RMSE between the two)
Output: results/s2s3_fullbag.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
EXP = HERE

import workload as WL

TIE_MARGIN = 0.05

def run_full(x, W, cfg):
    """Returns (out_spikes, margin_all_layers, s1tot, s2tot, s3tot)."""
    beta, thr, T = cfg["beta"], cfg["spike_threshold"], cfg["time_steps"]
    w1, w2, w3 = (W[k].astype(np.float64) for k in ("w1", "w2", "w3"))
    m1 = np.zeros(cfg["hidden_size"]); m2 = np.zeros(cfg["hidden_size"])
    m3 = np.zeros(cfg["output_size"])
    x = np.asarray(x, dtype=np.float64)
    out = None
    margin = np.full(cfg["output_size"], np.inf)
    hidden_min = np.inf
    s1t = s2t = s3t = 0.0
    for _ in range(T):
        m1 = beta * m1 + (x @ w1); s1 = (m1 >= thr).astype(float)
        hidden_min = min(hidden_min, float(np.abs(m1 - thr).min()))
        m1 -= thr * s1; s1t += s1.sum()
        m2 = beta * m2 + (s1 @ w2); s2 = (m2 >= thr).astype(float)
        hidden_min = min(hidden_min, float(np.abs(m2 - thr).min()))
        m2 -= thr * s2; s2t += s2.sum()
        m3 = beta * m3 + (s2 @ w3); out = (m3 >= thr).astype(float)
        margin = np.minimum(margin, np.abs(m3 - thr)); m3 -= thr * out; s3t += out.sum()
    return out, min(float(margin.min()), hidden_min), margin, hidden_min, s1t, s2t, s3t

def main():
    Wq = WL.load_weights(os.path.join(HERE, "common", "weights_quantised.npz"))
    Wf = WL.load_weights(os.path.join(HERE, "common", "weights.npz"))
    cfg = dict(WL.NET)
    scans = WL.read_scans()
    n = len(scans)
    print(f"bag frames: {n}", flush=True)
    enc = np.stack([WL.encode_input(WL.profile_from_scan(a)) for _, a in scans])

    mm = []; mm_nontie = []
    s1q = np.zeros(n); s2q = np.zeros(n); s3q = np.zeros(n)
    s1f = np.zeros(n); s2f = np.zeros(n); s3f = np.zeros(n)
    tie_frac = np.zeros(n)
    oq = Odom(); of = Odom()
    traj_q = np.zeros((n, 2)); traj_f = np.zeros((n, 2))
    for i in range(n):
        o_q, mall_q, marg_q, h_q, a, b, c = run_full(enc[i], Wq, cfg)
        o_f, mall_f, marg_f, h_f, af, bf, cf = run_full(enc[i], Wf, cfg)
        bad = (o_q != o_f)
        mm.append(bad.mean()); tie = np.minimum(marg_q, marg_f) < TIE_MARGIN
        mm_nontie.append((bad & ~tie).mean()); tie_frac[i] = tie.mean()
        s1q[i], s2q[i], s3q[i] = a, b, c; s1f[i], s2f[i], s3f[i] = af, bf, cf
        traj_q[i] = oq.step(o_q); traj_f[i] = of.step(o_f)
        if (i + 1) % 500 == 0: print(f"  {i+1}/{n}", flush=True)

    drift = np.linalg.norm(traj_q - traj_f, axis=1)
    out = {
        "n_frames": n, "bag": WL.DEFAULT_BAG, "seed": 12345,
        "lava_nc": "0.10.0",
        "mismatch_rate_mean": float(np.mean(mm)), "mismatch_rate_sd": float(np.std(mm)),
        "nontie_mismatch_rate_mean": float(np.mean(mm_nontie)),
        "nontie_mismatch_rate_max": float(np.max(mm_nontie)),
        "tie_unit_frac_mean": float(np.mean(tie_frac)),
        "spikes_per_frame_quant": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
            for k, v in (("L1", s1q), ("L2", s2q), ("L3", s3q))},
        "spikes_per_frame_float": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
            for k, v in (("L1", s1f), ("L2", s2f), ("L3", s3f))},
        "traj_drift_final_m": float(drift[-1]),
        "traj_drift_rmse_m": float(np.sqrt((drift ** 2).mean())),
        "traj_length_quant_m": float(np.linalg.norm(np.diff(traj_q, axis=0), axis=1).sum()),
        "traj_length_float_m": float(np.linalg.norm(np.diff(traj_f, axis=0), axis=1).sum()),
    }
    p = os.path.join(EXP, "results", "s2s3_fullbag.json")
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)

class Odom:
    def __init__(self): self.x = self.y = self.th = 0.0
    def step(self, s):
        s = np.asarray(s, dtype=float)
        self.th += float((s[:32] - s[32:64]).mean()) * 0.01
        v = float(s.mean()) * 0.02
        self.x += v * np.cos(self.th); self.y += v * np.sin(self.th)
        return (self.x, self.y)

if __name__ == "__main__":
    main()
