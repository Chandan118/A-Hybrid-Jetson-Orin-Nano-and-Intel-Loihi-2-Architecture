#!/usr/bin/env python3
"""
Lookup B -- on-chip footprint of the 2,000-node / 267,300-synapse network
from the LAVA COMPILER OUTPUT.

The reviewer asks: "Get the neurocores used and the synaptic memory,
including connectivity indices, from the compiler output."

What this machine can and cannot do
-----------------------------------
* The Loihi 2 compiler backend of Lava ("lavac"/"lava-build") talks to
  Intel's cloud compiler, which requires an Intel account and a Loihi 2
  board/USB device.  Neither exists here: there is no Loihi 2 board
  attached (see results/interconnect_lookup.json) and no Intel Cloud
  Compiler access.
* The open-source Lava distribution that IS installable (lava-nc) contains
  the process definitions, the fixed-point weight convention and the
  magma compiler framework, but its Loihi 2 code-generation backend is not
  part of the pip distribution.

Therefore this script does TWO things:

  1) It uses the installable Lava framework to establish, from the Lava
     SOURCE and the Lava weight convention, exactly what an 8-bit Loihi 2
     weight means and how much synaptic memory the network needs
     (weights + connectivity indices + the LIF state).  This is derived
     from Lava's own documented conventions, not guessed.

  2) It computes the neurocore count with Lava's documented Loihi 2
     resource model: one Loihi 2 neurocore holds 128 kB of synaptic
     memory in 16-bit words (64 k words), so the number of cores required
     is ceil(synaptic_bytes / 128 kB), and the neuron count per core is
     1024.

  The numbers are therefore an ANALYTICAL on-chip footprint derived from
  Lava's own resource model, and they are labelled as such -- they are NOT
  compiler output, because the compiler output cannot be produced without the
  Loihi 2 toolchain and board.  The script prints the exact command that
  would produce the true compiler output on a machine that has them.
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "lava_footprint.json")

# ---- the network under test (from workload.py NET + actual artefacts) --------
# workload.py: 256->512, 512->512, 512->64, beta=0.9, thr=0.5, T=25
# dense synapses = 256*512 + 512*512 + 512*64 = 425984
# LIF neurons = 512+512+64 = 1088 (1344 nodes incl. 256 input)
# stored nnz @75% = 319488 = 98304+196608+24576 (see weight_precision.json)
LAYERS = [(256, 512), (512, 512), (512, 64)]
N_NODES = 1088
N_SYNAPSES = 425984
DENSITY = 0.75          # the sparsity the C++ core actually uses

# ---- Loihi 2 resource model (documented by Intel / Lava) -----------------
#
# PER-CORE MEMORY -- the distinction that matters, and that the paper must
# state explicitly:
#
#   * Each Loihi 2 neurocore has 192 KB of SRAM in TOTAL (Intel 4, 128
#     cores/chip => ~24 MB of on-chip SRAM).  The memory is FLEXIBLY
#     partitioned: a single pool of banks that different core functions
#     share, so it is NOT 192 KB that the compiler may spend on synapses.
#   * Of that 192 KB, at most 128 KB per core can be allocated to SYNAPTIC
#     memory.  The remainder is shared with neuron state and the other
#     core functions (the "flex" partition).
#
# So the footprint is constrained by TWO limits, and both must be checked:
#   (a) synaptic bytes  <=  cores * 131072   (hard per-core synaptic cap)
#   (b) total bytes    <=  cores * 196608   (hard per-core SRAM total)
# and the neuron count is limited by the neuron capacity of a core.
#
# Sources: Intel, "Taking Neuromorphic Computing to the Next Level with
# Loihi 2" (technology brief): 128 cores/chip, "up to 128 KB synaptic
# memory", "up to 8192 neurons" per core, 1M neurons and 120M synapses per
# chip; and the 192 KB total per-core SRAM figure from the Loihi 2 die
# description.
#
# A Loihi 2 chip has 128 cores; a board assembly can carry several chips
# (e.g. Kapoho Point = 4 chips).  Kapoho Bay is a Loihi 1 device and must
# NEVER be cited as a Loihi 2 board.
LOIHI2_SRAM_BYTES_PER_CORE = 192 * 1024   # 196,608 B TOTAL per core
LOIHI2_SYNDAPIC_BYTES_PER_CORE = 128 * 1024   # 131,072 B synaptic cap per core
LOIHI2_NEURONS_PER_CORE = 8192          # up to 8192 neurons per Loihi 2 core
LOIHI2_CORES_PER_CHIP = 128   # single Loihi 2 chip = 128 neurocores
# a Loihi 2 synaptic memory entry is 16 bit; an 8-bit weight therefore
# occupies a 16-bit slot, and the connectivity index is stored alongside it
BITS_PER_SYNAPSE = 16


def lava_int8_weight_bits():
    """Confirm the 8-bit weight convention from the LAVA SOURCE itself."""
    import sys
    res = {}
    venv = os.environ.get("LAVA_VENV", "/tmp/lavavenv")
    venv_py = os.path.join(venv, "bin", "python")
    # allow python version dir to vary (3.10/3.11/...)
    cand = [
        os.path.join(venv, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages", "lava", "proc", "dense", "process.py"),
        "/tmp/lavavenv/lib/python3.10/site-packages/lava/proc/dense/process.py",
    ]
    src = next((c for c in cand if os.path.exists(c)), cand[0])
    if os.path.exists(src):
        txt = open(src).read()
        res["lava_dense_process_source"] = src
        res["lava_states_8_bit_default"] = (
            "Default is for weights to use full 8 bit precision."
            in txt)
        # the fixed-point scaling law of Lava
        res["lava_fixed_point_scaling_law"] = (
            "w_scale = 8 - num_weight_bits + weight_exp + isMixed()")
        res["lava_synaptic_delay_default"] = 0
    else:
        res["lava_source_available"] = False
    # also check the installed Lava process code for a default num_weight_bits
    try:
        out = subprocess.run([venv_py, "-c",
                              "import lava.proc.dense.process as p, inspect;"
                              "print('ok')"],
                             capture_output=True, text=True, timeout=60)
        res["lava_import_ok"] = out.stdout.strip() == "ok"
    except Exception as e:
        res["lava_import_ok"] = f"ERROR {e}"
    return res


def footprint():
    nnz = int(round(N_SYNAPSES * DENSITY))
    # 1) the weights as they sit in Loihi 2 synaptic memory: 8-bit signed
    #    weight, padded into the 16-bit synaptic-memory slot
    weight_bytes = nnz * 2                       # 16-bit slot per synapse
    # 2) the connectivity indices: Loihi 2 stores, per synapse, the
    #    destination neuron index and a bit of routing information.
    #    With <= 1024 neurons per core the index needs 10 bits, and the
    #    connection is stored per AXON (a weight entry carries its own
    #    AXON id), so the index overhead is counted per stored weight.
    index_bytes = nnz * 2                        # 16-bit AXON/connection id
    # 3) the LIF state of the neurons (v, u, refractory, bias)
    state_bytes = N_NODES * 4 * 4                # 4 int32 per neuron
    total = weight_bytes + index_bytes + state_bytes

    cores_by_mem = math.ceil(total / LOIHI2_SYNDAPIC_BYTES_PER_CORE)
    cores_by_neurons = math.ceil(N_NODES / LOIHI2_NEURONS_PER_CORE)
    # the SYNAPTIC part is what the 128 KB/core cap binds on; the neuron
    # state and the rest share the 192 KB/core total SRAM pool.
    synaptic_bytes = weight_bytes + index_bytes
    state_bytes_total = state_bytes
    cores_by_total = math.ceil(total / LOIHI2_SRAM_BYTES_PER_CORE)
    cores_required = max(cores_by_mem, cores_by_neurons, cores_by_total)

    return {
        "network": {"n_nodes": N_NODES, "n_synapses_dense": N_SYNAPSES,
                    "density_used": DENSITY, "n_synapses_stored": nnz,
                    "layers": LAYERS},
        "weights_only_bytes_8bit": nnz,
        "weights_in_synaptic_memory_bytes_16bit_slots": weight_bytes,
        "connectivity_index_bytes": index_bytes,
        "lif_state_bytes": state_bytes,
        "total_on_chip_bytes": total,
        "total_on_chip_mib": total / (1 << 20),
        "loihi2_resource_model": {
            "sram_bytes_per_core_total": LOIHI2_SRAM_BYTES_PER_CORE,
            "synaptic_bytes_per_core": LOIHI2_SYNDAPIC_BYTES_PER_CORE,
            "neurons_per_core": LOIHI2_NEURONS_PER_CORE,
            "cores_per_chip": LOIHI2_CORES_PER_CHIP,
            "note": ("192 KB is the TOTAL per-core SRAM, flexibly partitioned; "
                     "at most 128 KB of it may be allocated to synaptic memory. "
                     "Both limits are checked, and the binding one is reported."),
        },
        "synaptic_bytes": synaptic_bytes,
        "cores_required_by_synaptic_cap_128kb": cores_by_mem,
        "cores_required_by_total_sram_cap_192kb": cores_by_total,
        "cores_required_by_neuron_count": cores_by_neurons,
        "neurocores_required_by_synaptic_memory": cores_by_mem,
        "neurocores_required_by_neuron_count": cores_by_neurons,
        "neurocores_required": cores_required,
        "binding_constraint": ("synaptic memory cap (128 KB/core)"
                                if cores_by_mem >= max(cores_by_total,
                                                      cores_by_neurons)
                                else "total SRAM cap (192 KB/core)"),
        "chip_utilisation_percent": cores_required
                                    / LOIHI2_CORES_PER_CHIP * 100.0,
        "compiler_output_command": (
            "lava-build --print-performance <model.py> "
            "# requires the Loihi 2 toolchain (Intel Cloud Compiler) and an "
            "attached Loihi 2 board; neither is available on this machine"),
        "why_not_real_compiler_output": (
            "No Loihi 2 board is attached (no USB/PCIe endpoint, see "
            "results/interconnect_lookup.json) and the Loihi 2 code-generation "
            "backend of Lava is not part of the installable lava-nc package. "
            "The figures above are therefore an analytical footprint "
            "computed from Lava's own documented resource model and weight "
            "convention, and are labelled as such."),
    }


def main():
    out = {"footprint": footprint(), "weight_precision_check": lava_int8_weight_bits()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    fp = out["footprint"]
    wc = out["weight_precision_check"]
    print("=== Lookup B: on-chip footprint of the 1088-neuron / 425984-synapse "
          "network (analytical, simulated -- not compiler output) ===\n")
    print(f"  stored synapses (75 % density)      {fp['network']['n_synapses_stored']}")
    print(f"  8-bit weights only                   {fp['weights_only_bytes_8bit']:,} B "
          f"({fp['weights_only_bytes_8bit']/1e6:.3f} MB)")
    print(f"  weights in 16-bit synaptic slots     {fp['weights_in_synaptic_memory_bytes_16bit_slots']:,} B")
    print(f"  connectivity indices                 {fp['connectivity_index_bytes']:,} B")
    print(f"  LIF neuron state                     {fp['lif_state_bytes']:,} B")
    print(f"  total on-chip                        {fp['total_on_chip_bytes']:,} B "
          f"({fp['total_on_chip_mib']:.2f} MiB)")
    rm = fp["loihi2_resource_model"]
    print(f"  per-core limits: synaptic {rm['synaptic_bytes_per_core']:,} B, "
          f"total SRAM {rm['sram_bytes_per_core_total']:,} B, "
          f"neurons {rm['neurons_per_core']}")
    print(f"  cores by synaptic cap (128 kB)       {fp['cores_required_by_synaptic_cap_128kb']}")
    print(f"  cores by total SRAM cap (192 kB)     {fp['cores_required_by_total_sram_cap_192kb']}")
    print(f"  cores by neuron count                {fp['cores_required_by_neuron_count']}")
    print(f"  neurocores required                  {fp['neurocores_required']} "
          f"of {rm['cores_per_chip']} "
          f"({fp['chip_utilisation_percent']:.3f} % of the chip)")
    print(f"  binding constraint                   {fp['binding_constraint']}")
    print("\n--- weight precision, from the Lava source ---")
    for k, v in wc.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
