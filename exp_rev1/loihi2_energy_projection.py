#!/usr/bin/env python3
"""Three-term Loihi projected energy model (all SIMULATED/PROJECTED, never measured).

CORRECTED fan-out (rev2): network is 256->512->512->64, T=25.
  s1 = L1 neuron spikes/frame (512 neurons x 25 steps)
  s2 = L2 neuron spikes/frame (512 neurons x 25 steps)
  s3 = L3/output spikes/frame (64 neurons x 25 steps, outputs: NO fan-out)
  Correct assignment:
    input graded events drive 256->512 (stored 98304, every step, graded);
    L1 spikes drive 512->512 (fanout 384 = 196608/512);
    L2 spikes drive 512->64  (fanout 48  = 24576/512);
    L3 spikes are outputs (fanout 0).
  The previous version wrongly used s1*384 + s2*384 + s3*48 and omitted the
  input term. Both totals are reported so a reviewer can see the effect.

Model per frame (T=25, N=1088 LIF neurons, nnz=319488):
  E_dyn = N_synops  x e_synop      (synaptic ops, sparse, corrected)
        + N_nupd    x e_nupd       (neuron updates: 1088 x 25 = 27200/frame)
        + N_spike   x e_spike      (spike events: measured mean +/- SD)
  E_frame = E_dyn + P_static x T_frame (static listed separately)

Coefficient source (literature, NOT measured here -- and Loihi 1, NOT Loihi 2):
  Davies et al. 2018 (FIRST Loihi chip, Intel 14nm, IEEE Micro 38(1), Table 2):
    caption "pre-silicon performance and energy measurements", i.e. SDF/SPICE
    simulation + early silicon characterisation, NOT a production-silicon
    meter reading. Re-checked against the PDF 2026-11-09:
      within-tile spike energy 1.7 pJ; tile hop E-W/N-S 3.0/4.0 pJ;
      synaptic spike op (min) 23.6 pJ; STDP update 120 pJ (excluded here,
      this network has no STDP); neuron update active/inactive 81/52 pJ.
    These are LOIHI-1 coefficients used as a PROXY because Loihi 2 (Intel 4)
    has no published same-definition silicon pJ/op. The text must say
    "Loihi 1 coefficients used as a proxy", not just the supplement.
    e_synop low = 23.6 pJ (optimistic floor), high = 100 pJ (conservative
    routing-inclusive upper bound, explicitly an assumption).
  N_synops / N_spike / N_nupd from results/s2s3_fullbag.json (full-bag counts).

Output: results/loihi2_energy_projection.json
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))

# --- 文献系数 (pJ),出处见模块 docstring ------------------------------------
E_SYNOP_LOW_PJ = 23.6    # Davies2018 Table2 min, pre-silicon, optimistic floor
E_SYNOP_HIGH_PJ = 100.0  # 含路由/多跳开销的保守上界 (假设,明确标注)
E_NUPD_ACTIVE_PJ = 81.0  # Davies2018 Table2, active neuron update
E_NUPD_QUIET_PJ = 52.0   # Davies2018 Table2, inactive neuron update
E_SPIKE_PJ = 1.7         # Davies2018 Table2, within-tile spike energy

# --- network structure ---------------------------------------------------------
N_LIF = 1088
T_STEPS = 25
N_NUPD_PER_FRAME = N_LIF * T_STEPS          # 27200
# Stored synapses per matrix (from weight_precision.json .bin parse,
# exactly 75% of dense): W1 98304 (256->512), W2 196608 (512->512),
# W3 24576 (512->64). Fan-out per SPIKE through each matrix:
FANOUT_W1 = 98304 / 256     # 384 targets per input event
FANOUT_W2 = 196608 / 512    # 384 targets per L1 spike
FANOUT_W3 = 24576 / 512     # 48 targets per L2 spike
# 256 graded inputs are active EVERY step (binned range profile in [0,1],
# see workload.encode_input): input events/frame = 256 x T_STEPS.
N_INPUT_EVENTS_PER_FRAME = 256 * T_STEPS    # 6400

def main():
    fb_path = os.path.join(HERE, "results", "s2s3_fullbag.json")
    if not os.path.exists(fb_path):
        raise SystemExit(f"run s2s3_fullbag.py first to generate {fb_path}")
    fb = json.load(open(fb_path))
    q = fb["spikes_per_frame_quant"]
    s1, s2, s3 = q["L1"]["mean"], q["L2"]["mean"], q["L3"]["mean"]
    s1sd, s2sd, s3sd = q["L1"]["sd"], q["L2"]["sd"], q["L3"]["sd"]

    # CORRECTED synaptic ops/frame:
    #   input graded events x W1 fanout + L1 spikes x W2 fanout
    #   + L2 spikes x W3 fanout. L3 spikes are outputs (fanout 0).
    syn_in = N_INPUT_EVENTS_PER_FRAME * FANOUT_W1
    syn_h1 = s1 * FANOUT_W2
    syn_h2 = s2 * FANOUT_W3
    synops = syn_in + syn_h1 + syn_h2
    synops_sd = ((s1sd * FANOUT_W2) ** 2 + (s2sd * FANOUT_W3) ** 2) ** 0.5
    # Old (wrong) assignment kept for audit: s1*384 + s2*384 + s3*48.
    synops_old = s1 * 384.0 + s2 * 384.0 + s3 * 48.0
    nupd = N_NUPD_PER_FRAME
    nspk = s1 + s2 + s3

    def dyn(e_syn):
        e_syn_j = synops * e_syn * 1e-12
        # Neuron updates use the all-active upper bound (81 pJ) and say so.
        e_nupd_j = nupd * E_NUPD_ACTIVE_PJ * 1e-12
        e_spk_j = nspk * E_SPIKE_PJ * 1e-12
        return {"synops_J": e_syn_j, "nupd_J": e_nupd_j, "spike_J": e_spk_j,
                "total_dyn_J": e_syn_j + e_nupd_j + e_spk_j}

    low = dyn(E_SYNOP_LOW_PJ)
    high = dyn(E_SYNOP_HIGH_PJ)
    out = {
        "provenance": "SIMULATED/PROJECTED -- CPU spike counts x literature pJ, NOT chip data",
        "network": {"lif_neurons": N_LIF, "time_steps": T_STEPS,
                    "stored_synapses": 319488,
                    "fanout_W1_input": FANOUT_W1,
                    "fanout_W2_from_L1": FANOUT_W2,
                    "fanout_W3_from_L2": FANOUT_W3,
                    "fanout_L3_output": 0,
                    "input_events_per_frame": N_INPUT_EVENTS_PER_FRAME},
        "counts_per_frame": {
            "spikes_L1_L2_L3": [s1, s2, s3],
            "spikes_total": nspk,
            "synops_terms": {"input_W1": syn_in, "L1_via_W2": syn_h1,
                             "L2_via_W3": syn_h2, "L3_output": 0.0},
            "synops": synops, "synops_sd": synops_sd,
            "synops_old_wrong_assignment": synops_old,
            "neuron_updates": nupd},
        "coefficients_pJ": {
            "e_synop_low": E_SYNOP_LOW_PJ,
            "e_synop_high": E_SYNOP_HIGH_PJ,
            "e_nupd_active": E_NUPD_ACTIVE_PJ,
            "e_nupd_quiet": E_NUPD_QUIET_PJ,
            "e_spike": E_SPIKE_PJ,
            "e_synop_note": "Davies2018 = FIRST Loihi chip (Intel 14nm), "
                            "Table2 is pre-silicon SDF/SPICE, NOT production-"
                            "silicon meter data; Loihi-1 used as PROXY for "
                            "Loihi-2 (Intel 4, no same-definition figure). "
                            "high=100pJ is a routing-inclusive assumption"},
        "E_dyn_low_J": low, "E_dyn_high_J": high,
        "static_note": "静态功耗 P_static 未在本机测得,按下式单独计入: "
                       "E_frame = E_dyn + P_static x T_frame; "
                       "Loihi2 待机/漏电请引用 Intel 官方或 INRC 实测,不得用 CPU 功耗代替",
        "lava_nc": "0.10.0",
    }
    p = os.path.join(HERE, "results", "loihi2_energy_projection.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"synops/frame = {synops:,.0f} +/- {synops_sd:,.0f}")
    print(f"nupd/frame   = {nupd:,}")
    print(f"spikes/frame = {nspk:,.1f}")
    print(f"E_dyn low    = {low['total_dyn_J']*1e6:.2f} uJ "
          f"(syn {low['synops_J']*1e6:.2f} + nupd {low['nupd_J']*1e6:.2f} + spk {low['spike_J']*1e6:.3f})")
    print(f"E_dyn high   = {high['total_dyn_J']*1e6:.2f} uJ "
          f"(syn {high['synops_J']*1e6:.2f} + nupd {high['nupd_J']*1e6:.2f} + spk {high['spike_J']*1e6:.3f})")
    print(f"wrote {p}")

if __name__ == "__main__":
    main()
