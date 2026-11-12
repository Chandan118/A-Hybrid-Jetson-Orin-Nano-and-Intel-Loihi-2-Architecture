#!/usr/bin/env python3
"""
The Lava process graph for the fusion network, as a real importable module.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Lava's compiler finds a ProcessModel for a Process by searching the module
in which the Process is DEFINED:

    10|    magma/compiler/compiler_graphs.py:758
        classes = [cls for cls in module.__dict__.values()
                   if inspect.isclass(cls) and cls.__module__ == module.__name__]
        ... and cls.implements_process is not None
              and type(proc) is cls.implements_process

So the Process class and its ProcessModel must be top-level classes of the
same module, with `cls.__module__ == module.__name__`.  Defining them inside
a function (as an earlier draft did) makes the compiler unable to find the
model and it raises NoProcessModelFound.
    20|
This module therefore defines, at top level:
    FrameSource   / PyFrameSourceModel   -- the graded 256-dim frame input
    SpikeRecorder / PySpikeRecorderModel -- the output spike recorder
    DenseExact    / PyDenseExactModel     -- the reference dense product
    LIFExact      / PyLIFExactModel       -- the reference LIF recurrence

and `build_network()` wires them together.

TWO NETWORK VARIANTS, AND WHY BOTH EXIST
-----------------------------------------
Lava ships its own `Dense` and `LIF` processes, and `Loihi2SimCfg` is
Lava's own run configuration.  Using them is the closest thing to a Loihi 2
deployment that can be run without the board -- but they are NOT
numerically equivalent to the reference network that C1..C5 evaluate:

  * **Lava's `Dense` buffers its output by one time step.**  It sends the
    dendritic accumulation computed at step t-1 and computes the new one at
    step t ("this prevents deadlocking in networks with recurrent
    connectivity structures", lava/proc/dense/models.py).  The reference
    network feeds the CURRENT step's spikes into the next layer.  With a
    constant frame input the lag is invisible in layer 1, but in layers 2
    and 3 the reference network's layer-2 neuron sees s1(t) while Lava's
    sees s1(t-1).

  * **Lava's `LIF` resets v to 0 on spike; the reference subtracts the
    threshold.**  The reference recurrence is
        v <- beta*v + cur ;  spike if v >= thr ;  v <- v - thr
    Lava's LIF is
        u <- u*(1-du) + a_in ;  v <- v*(1-dv) + u ;  spike if v > vth ;
        v <- 0
    with du = 0 (the reference has no current state) and dv = 1 - beta.

  * **Lava's `Dense` has no bias term.**  The reference network's biases
    are all zero, so nothing is lost -- and this is checked, not assumed.

So this module provides BOTH:

  * `build_network(..., variant="lava")` -- Lava's OWN `Dense` and `LIF`
    processes.  This is the configuration that answers "what does Lava's
    Loihi 2 software stack do with this network?".  Its spike train
    deviates from the reference, and the deviation is MEASURED and reported
    (see test_backends_equivalent.py), not hidden.

  * `build_network(..., variant="exact")` -- the reference dense product and
    the reference LIF recurrence, each as a Lava ProcessModel, so the
    network runs through Lava's compiler, Lava's runtime and Lava's
    `Loihi2SimCfg` run configuration, but computes the SAME function as
    C1/C2/C3/C4/C5.  This is the configuration that goes into the
    energy-per-frame comparison, because it isolates the SOFTWARE STACK
    (Lava's compiler + runtime + process-model framework) as the only
    variable.  Its spike train must agree with the float64 reference
    exactly, and that is asserted by the equivalence test.

IMPORTANT -- this module is imported ONLY by the Lava virtual environment
(`/tmp/lavavenv/bin/python`), because it needs `lava.magma`.  The system
Python does not have it.
"""

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


# --------------------------------------------------------------- the input

class FrameSource(AbstractProcess):
    """Supplies the frame's graded 256-dim range profile to the network.

    The frame input is CONTINUOUS (a binned range profile), not a spike
    train, so this port carries a float value per bin.  The first Dense is
    therefore run as a GRADED dense, which is Lava's own way of computing a
    matrix product against a graded input.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.out = OutPort(shape=(256,))
        self.frame = Var(shape=(256,), init=0.0)


@implements(proc=FrameSource, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyFrameSourceModel(PyLoihiProcessModel):
    frame: np.ndarray = LavaPyType(np.ndarray, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.out.send(self.frame)


# -------------------------------------------------------------- the output

class SpikeRecorder(AbstractProcess):
    """Records the output spikes of the last layer, one vector per step."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.inp = InPort(shape=(64,))
        self.last = Var(shape=(64,), init=0.0)


@implements(proc=SpikeRecorder, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySpikeRecorderModel(PyLoihiProcessModel):
    last: np.ndarray = LavaPyType(np.ndarray, float)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self):
        self.last[:] = self.inp.recv()


# ------------------------------------------------- the exact-reference dense

class DenseExact(AbstractProcess):
    """The reference network's dense product, as a Lava process.

    Computes a_out = W . s_in and sends it in the SAME time step, which is
    what the reference recurrence does.  (Lava's own `Dense` buffers its
    output by one step; see the module docstring.)
    """
    def __init__(self, weights, **kw):
        super().__init__(weights=weights, **kw)
        shape = np.shape(weights)
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=np.asarray(weights,
                                                        dtype=np.float32))


@implements(proc=DenseExact, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDenseExactModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        s_in = self.s_in.recv()
        self.a_out.send(self.weights.dot(s_in.astype(np.float32)))


# ------------------------------------------------- the exact-reference LIF

class LIFExact(AbstractProcess):
    """The reference network's LIF recurrence, as a Lava process.

        v <- beta*v + cur ;  spike if v >= thr ;  v <- v - thr

    (Lava's own `LIF` resets v to 0 on spike instead of subtracting the
    threshold; see the module docstring.)
    """
    def __init__(self, shape, beta, thr, **kw):
        super().__init__(shape=shape, beta=beta, thr=thr, **kw)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0.0)


@implements(proc=LIFExact, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyLIFExactModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        beta = float(self.proc_params.get("beta", 0.9))
        thr = float(self.proc_params.get("thr", 0.5))
        cur = self.a_in.recv()
        self.v[:] = beta * self.v + cur.astype(np.float32)
        s = (self.v >= thr).astype(np.float32)
        self.v[:] = self.v - thr * s
        self.s_out.send(s)


# ------------------------------------------------------------- the LIF params

# The LIF parameters of the reference network, translated to Lava's own
# parameterisation for the "lava" variant:
#   the reference does  v <- beta*v + cur ; spike if v >= thr ; v -= thr
#   Lava's LIF does     u <- u*(1-du) + a_in
#                       v <- v*(1-dv) + u + bias
#   with du = 0 (the reference has no current state) and
#   dv = 1 - beta = 0.1 (the reference's membrane decay).
LAVA_DU = 0.0
LAVA_DV = 0.1
LAVA_VTH = 0.5


def build_network(weights, cfg, name_prefix="c6", variant="exact"):
    """Build the Lava process graph for the fusion network.

    variant="exact" : the reference dense product and the reference LIF
                       recurrence, as Lava processes.  The network computes
                       the SAME function as C1..C5, and runs through Lava's
                       compiler, Lava's runtime and Lava's Loihi2SimCfg.

    variant="lava"  : Lava's OWN `Dense` and `LIF` processes.  This is the
                       configuration that measures Lava's Loihi 2 software
                       stack as shipped, including its one-step dense
                       buffering and its zero-reset LIF.

    Returns (source_process, recorder_process, dense_processes, lif_processes).
    """
    sizes = [(cfg["input_size"], cfg["hidden_size"]),
             (cfg["hidden_size"], cfg["hidden_size"]),
             (cfg["hidden_size"], cfg["output_size"])]

    if variant == "lava":
        from lava.proc.dense.process import Dense
        from lava.proc.lif.process import LIF
    elif variant == "exact":
        Dense = None
        LIF = None
    else:
        raise ValueError("unknown variant %r" % (variant,))

    src = FrameSource(name=f"{name_prefix}_source")
    rec = SpikeRecorder(name=f"{name_prefix}_recorder")

    denses, lifs = [], []
    prev = src.out
    for li, ((n_in, n_out), wk) in enumerate(zip(sizes, ("w1", "w2", "w3"))):
        W = weights[wk]                     # (n_in, n_out)
        # Lava's Dense takes a (n_out, n_in) matrix and computes W . s_in.
        # The first layer is GRADED (num_message_bits > 0) because the frame
        # input is a continuous range profile, not spikes; the later layers
        # are binary (num_message_bits = 0), which is Lava's spike-driven
        # dense.
        if variant == "lava":
            graded = (li == 0)
            d = Dense(weights=np.ascontiguousarray(W.T.astype(np.float32)),
                      num_message_bits=8 if graded else 0,
                      name=f"{name_prefix}_dense{li+1}")
            lf = LIF(shape=(n_out,), u=0.0, v=0.0, du=LAVA_DU, dv=LAVA_DV,
                     bias_mant=0.0, bias_exp=0.0, vth=LAVA_VTH,
                     name=f"{name_prefix}_lif{li+1}")
        else:
            d = DenseExact(weights=np.ascontiguousarray(W.T.astype(np.float32)),
                           name=f"{name_prefix}_dense{li+1}")
            lf = LIFExact(shape=(n_out,), beta=cfg["beta"],
                          thr=cfg["spike_threshold"],
                          name=f"{name_prefix}_lif{li+1}")
        denses.append(d)
        lifs.append(lf)
        prev.connect(d.s_in)
        d.a_out.connect(lf.a_in)
        prev = lf.s_out
    prev.connect(rec.inp)
    return src, rec, denses, lifs
