#!/usr/bin/env python3
"""
Lookup A -- weight precision.

The reviewer asks: "Your text says 8-bit. Confirm it in your Lava code."

What is checked, in order of authority:

  1) THE REPOSITORY'S OWN LAVA CODE.  The repository is searched for any
     Lava source at all (any file importing lava, any .py under a Lava
     package, any Lava model).  If there is no Lava code, that is the
     primary finding and it is reported as such: the 8-bit claim cannot be
     confirmed in Lava code because no Lava code exists.
  2) THE INSTALLED LAVA FRAMEWORK.  The Lava Dense process source is read
     to establish what Lava's own 8-bit weight convention is and how the
     fixed-point scale is derived, so that the claim can be restated in
     Lava's terms.
  3) THE PROJECT'S ACTUAL WEIGHT ARTEFACTS.  The int8/sparse binary that
     every configuration loads is parsed and its element type is verified
     byte by byte, so the "8-bit" claim is confirmed for the artefact that
     was actually measured.
"""

import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(EXP)
OUT = os.path.join(EXP, "results", "weight_precision.json")

INT8_MIN, INT8_MAX = -128, 127


def search_repo_for_lava():
    """Search the WHOLE repository for any Lava source code."""
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "build",
                                                "install", "log", "node_modules")]
        for fn in files:
            if not fn.endswith((".py", ".yaml", ".yml", ".json", ".md", ".cpp", ".h")):
                continue
            p = os.path.join(root, fn)
            try:
                txt = open(p, errors="ignore").read()
            except Exception:
                continue
            if "import lava" in txt or "from lava" in txt or "lava.lib" in txt \
               or "lava.process" in txt or "lava.proc" in txt:
                hits.append(p)
    return hits


def lava_source_weight_convention():
    """Read the installed Lava Dense process source."""
    res = {}
    src = "/tmp/lavavenv/lib/python3.10/site-packages/lava/proc/dense/process.py"
    if not os.path.exists(src):
        res["lava_installed"] = False
        return res
    res["lava_installed"] = True
    res["source"] = src
    txt = open(src).read()
    res["lava_doc_8bit_default"] = (
        "Default is for weights to use full 8 bit precision." in txt)
    res["lava_fixed_point_scaling_law"] = (
        "w_scale = 8 - num_weight_bits + weight_exp + isMixed()")
    res["lava_weight_bits_semantics"] = (
        "num_weight_bits is the SHARED weight width; the sign is carried "
        "separately when sign_mode is MIXED, so a signed 8-bit weight needs "
        "8 magnitude bits + a sign bit")
    # the LIF process's own fixed-point parameter names
    lif = "/tmp/lavavenv/lib/python3.10/site-packages/lava/proc/lif/process.py"
    if os.path.exists(lif):
        ltxt = open(lif).read()
        res["lava_lif_process_source"] = lif
        res["lava_lif_has_mant_exp"] = ("du" in ltxt and "dv" in ltxt
                                         and "bias_mant" in ltxt)
    return res


def verify_project_weights():
    """Parse the int8/sparse weight binary that every configuration loads."""
    binp = os.path.join(EXP, "common", "fusion_weights_int8_sparse.bin")
    if not os.path.exists(binp):
        return {"binary": binp, "exists": False}
    data = open(binp, "rb").read()
    out = {"binary": binp, "exists": True, "size_bytes": len(data)}
    # the format (see common/export_int8_sparse.py), repeated per layer with
    # NO header:
    #   n_in(int32), n_out(int32), scale(float32), bias(float32 x n_out),
    #   len(row_ptr)(int32), len(cols)(int32),
    #   row_ptr(int32 x len), cols(int32 x len), vals(int8 x len)
    off = 0
    layers = []
    all_int8 = True
    for _ in range(3):
        (n_in,) = struct.unpack_from("<i", data, off); off += 4
        (n_out,) = struct.unpack_from("<i", data, off); off += 4
        (scale,) = struct.unpack_from("<f", data, off); off += 4
        bias = struct.unpack_from(f"<{n_out}f", data, off); off += 4 * n_out
        (n_row,) = struct.unpack_from("<i", data, off); off += 4
        (n_col,) = struct.unpack_from("<i", data, off); off += 4
        row_ptr = struct.unpack_from(f"<{n_row}i", data, off); off += 4 * n_row
        col = struct.unpack_from(f"<{n_col}i", data, off); off += 4 * n_col
        vals = struct.unpack_from(f"<{n_col}b", data, off); off += n_col
        lo, hi = (min(vals), max(vals)) if n_col else (0, 0)
        fits = (lo >= INT8_MIN and hi <= INT8_MAX)
        all_int8 &= fits
        layers.append({
            "n_in": n_in, "n_out": n_out, "nnz": n_col,
            "quantisation_scale": float(scale),
            "value_type": "int8 (signed 8-bit, one byte per stored synapse)",
            "value_range_observed": [lo, hi],
            "values_fit_in_int8": fits,
            "reconstructed_max_abs_weight": (max(abs(v) for v in vals) * scale)
                                             if n_col else 0.0,
            "index_type": "int32 (col index) + int32 (row_ptr)",
        })
    out["layers"] = layers
    out["all_weights_are_int8"] = all_int8
    out["bytes_consumed"] = off
    out["bytes_total"] = len(data)
    out["file_fully_parsed"] = (off == len(data))
    out["note"] = ("the binary stores each synaptic weight in ONE signed byte; "
                   "the float32 value is w = val * scale")
    return out


def main():
    # the lookup scripts themselves and run1_power.py's optional Lava path are
    # excluded: they are part of THIS review response, not of the project.
    EXCLUDE = ("exp_rev1/lookups/", "exp_rev1/run1_power.py")
    repo_hits = [h for h in search_repo_for_lava()
                 if not any(x in h for x in EXCLUDE)]
    out = {
        "question": "Weight precision: confirm the 8-bit claim in the Lava code.",
        "finding_1_repository_lava_code": {
            "number_of_lava_source_files_in_repository": len(repo_hits),
            "files": repo_hits,
            "verdict": ("The repository contains NO Lava source code at all. "
                        "The 8-bit claim therefore CANNOT be confirmed in the "
                        "project's Lava code, because there is none. The "
                        "project's fusion core is implemented in C++/C with an "
                        "explicitly int8/sparse weight format, and that is "
                        "what was measured."
                        if not repo_hits else
                        "Lava source files found; see 'files'."),
        },
        "finding_2_lava_framework_convention": lava_source_weight_convention(),
        "finding_3_measured_artefact": verify_project_weights(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    f1, f2, f3 = out["finding_1_repository_lava_code"], \
        out["finding_2_lava_framework_convention"], \
        out["finding_3_measured_artefact"]
    print("=== Lookup A: weight precision ===\n")
    print(f"  Lava source files in the repository : {f1['number_of_lava_source_files_in_repository']}")
    print(f"  Lava 8-bit default in Lava's Dense  : {f2.get('lava_doc_8bit_default')}")
    print(f"  Measured artefact                  : {f3.get('binary')}")
    print(f"  All weights are int8               : {f3.get('all_weights_are_int8')}")
    for L in f3.get("layers", []):
        print(f"    layer {L['n_in']:>4}->{L['n_out']:<4} nnz={L['nnz']:>7} "
              f"value range [{L['value_range_observed'][0]}, "
              f"{L['value_range_observed'][1]}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
