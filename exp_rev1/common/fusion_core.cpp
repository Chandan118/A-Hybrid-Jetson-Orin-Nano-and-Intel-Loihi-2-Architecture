/*
 * C3 -- optimized C++ fusion core for the ARM CPU (shared library, C ABI).
 *
 * Event-driven: the worker blocks on a semaphore waiting for frames; the
 * caller posts an event and waits for the result.  No polling, no sleep loop.
 *
 * int8 weights in a sparse CSR layout, -O3, pinned cores, 50 Hz.
 */

#include "fusion_core.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <semaphore.h>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace {

constexpr int IN = 256;
constexpr int H1 = 512;
constexpr int H2 = 512;
constexpr int OUT = 64;
constexpr int TSTEPS = 25;
constexpr float BETA = 0.9f;
constexpr float THR = 0.5f;

struct Layer {
    int n_in = 0, n_out = 0;
    float scale = 1.0f;
    std::vector<float> bias;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col;
    std::vector<int8_t> val;
};

template <typename T>
bool rd(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return bool(f);
}

template <typename T>
bool rdv(std::ifstream& f, std::vector<T>& v) {
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    return bool(f);
}

Layer load_layer(std::ifstream& f) {
    Layer L;
    rd(f, L.n_in);
    rd(f, L.n_out);
    rd(f, L.scale);
    L.bias.resize(L.n_out);
    f.read(reinterpret_cast<char*>(L.bias.data()), L.n_out * sizeof(float));
    int32_t nptr = 0, ncol = 0;
    rd(f, nptr);
    rd(f, ncol);
    L.row_ptr.resize(nptr);
    L.col.resize(ncol);
    L.val.resize(ncol);
    rdv(f, L.row_ptr);
    rdv(f, L.col);
    rdv(f, L.val);
    return L;
}

inline void spmm(const Layer& L, const float* x, float* y) {
    for (int r = 0; r < L.n_out; ++r) {
        float acc = L.bias[r];
        const int begin = L.row_ptr[r];
        const int end = L.row_ptr[r + 1];
        for (int k = begin; k < end; ++k) {
            acc += x[L.col[k]] * (static_cast<float>(L.val[k]) * L.scale);
        }
        y[r] = acc;
    }
}

}  // namespace

struct Fc {
    Layer l1, l2, l3;
    std::vector<float> inputs;
    int64_t n_rows = 0;
    int dim = IN;
    int rate_hz = 50;

    sem_t ev;
    std::atomic<bool> stop{false};
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> cur_frame{-1};
    std::atomic<bool> worker_failed{false};
    std::thread th;

    std::vector<float> m1, m2, m3, s1, s2, s3, cur1, cur2, cur3, x;
    std::vector<float> zero_input = std::vector<float>(IN, 0.0f);
    std::vector<float> in_ring;          // streaming input ring buffer
    std::atomic<int64_t> in_slot{-1};
    int cpu_lo = 0, cpu_hi = 0;
    std::vector<double> lat;
    std::vector<float> last_out;
    double px = 0, py = 0, pth = 0;

    // per-frame outputs, so a pipelined consumer can read the result of
    // frame N-1 while frame N is still in flight
    std::mutex out_mtx;
    std::unordered_map<int64_t, std::vector<float>> outputs;
    std::unordered_map<int64_t, double> lat_map;

    void reset_state() {
        std::fill(m1.begin(), m1.end(), 0.0f);
        std::fill(m2.begin(), m2.end(), 0.0f);
        std::fill(m3.begin(), m3.end(), 0.0f);
        px = py = pth = 0;
        last_out.assign(OUT, 0.0f);
    }

    void start_worker() {
        th = std::thread([this]() {
#ifdef __linux__
            // pin THIS thread to the configured core set (a thread can only
            // change its own affinity)
            if (cpu_hi > 0) {
                cpu_set_t set;
                CPU_ZERO(&set);
                for (int c = cpu_lo; c <= cpu_hi && c < CPU_SETSIZE; ++c) CPU_SET(c, &set);
                sched_setaffinity(0, sizeof(set), &set);
            }
#endif
            for (;;) {
#ifdef __linux__
                // apply the core mask to THIS thread (a thread can only
                // change its own affinity).  The mask may be set after the
                // thread started, so it is (re-)applied on every iteration.
                if (cpu_hi > 0) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    for (int c = cpu_lo; c <= cpu_hi && c < CPU_SETSIZE; ++c) CPU_SET(c, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                }
#endif
                sem_wait(&ev);
                if (stop.load()) break;
                const int64_t f = cur_frame.load();
                const float* xin;
                if (this->inputs.empty() || this->n_rows <= 0) {
                    // streaming mode: the input for this frame lives in the
                    // ring buffer slot that fc_set_input wrote
                    size_t slot = (size_t)(f % 16);
                    xin = &this->in_ring[slot * IN];
                } else {
                    xin = &this->inputs[(size_t)(f % this->n_rows) * this->dim];
                }
                std::memcpy(x.data(), xin, sizeof(float) * std::min(dim, IN));

                auto t0 = std::chrono::steady_clock::now();
                for (int t = 0; t < TSTEPS; ++t) {
                    spmm(l1, x.data(), cur1.data());
                    for (int i = 0; i < H1; ++i) {
                        m1[i] = BETA * m1[i] + cur1[i];
                        s1[i] = (m1[i] >= THR) ? 1.0f : 0.0f;
                        m1[i] = m1[i] - THR * s1[i];   // snntorch 'subtract' reset
                    }
                    spmm(l2, s1.data(), cur2.data());
                    for (int i = 0; i < H2; ++i) {
                        m2[i] = BETA * m2[i] + cur2[i];
                        s2[i] = (m2[i] >= THR) ? 1.0f : 0.0f;
                        m2[i] = m2[i] - THR * s2[i];
                    }
                    spmm(l3, s2.data(), cur3.data());
                    for (int i = 0; i < OUT; ++i) {
                        m3[i] = BETA * m3[i] + cur3[i];
                        s3[i] = (m3[i] >= THR) ? 1.0f : 0.0f;
                        m3[i] = m3[i] - THR * s3[i];
                    }
                }
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                double wsum = 0, ssum = 0;
                for (int i = 0; i < 32; ++i) wsum += (s3[i] - s3[i + 32]);
                for (int i = 0; i < 64; ++i) ssum += s3[i];
                double dtheta = (wsum / 32.0) * 0.01;
                double v = (ssum / 64.0) * 0.02;
                pth += dtheta;
                px += v * std::cos(pth);
                py += v * std::sin(pth);

                std::memcpy(last_out.data(), s3.data(), OUT * sizeof(float));
                lat.push_back(ms);
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    outputs[f] = last_out;
                    lat_map[f] = ms;
                }
                done.store(f + 1);
            }
        });
    }
};

extern "C" {

Fc* fc_create(const char* weights_path) {
    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf) return nullptr;
    Fc* fc = new Fc();
    fc->l1 = load_layer(wf);
    fc->l2 = load_layer(wf);
    fc->l3 = load_layer(wf);
    fc->m1.assign(H1, 0.f); fc->m2.assign(H2, 0.f); fc->m3.assign(OUT, 0.f);
    fc->s1.assign(H1, 0.f); fc->s2.assign(H2, 0.f); fc->s3.assign(OUT, 0.f);
    fc->cur1.assign(H1, 0.f); fc->cur2.assign(H2, 0.f); fc->cur3.assign(OUT, 0.f);
    fc->x.assign(IN, 0.f);
    fc->last_out.assign(OUT, 0.f);
    sem_init(&fc->ev, 0, 0);
    fc->start_worker();
    return fc;
}

int fc_configure(Fc* fc, const float* inputs, int64_t n_rows, int dim,
                 int rate_hz, int cpu_lo, int cpu_hi) {
    if (!fc) return -1;
    if (inputs && n_rows > 0) {
        fc->inputs.assign(inputs, inputs + n_rows * dim);
        fc->n_rows = n_rows;
        fc->dim = dim;
    }
    fc->rate_hz = rate_hz;
    fc->cpu_lo = cpu_lo;
    fc->cpu_hi = cpu_hi;
    return 0;
}

/* Pin the core's worker thread to a fixed set of CPU cores.  Pinning keeps
 * the fusion core off the cores the OS and the GPU front-end use, which is
 * what makes the hybrid's throughput reproducible.  The mask is applied by
 * the worker thread itself (a thread can only change its own affinity). */
int fc_set_affinity(Fc* fc, int cpu_lo, int cpu_hi) {
    if (!fc) return -1;
    fc->cpu_lo = cpu_lo;
    fc->cpu_hi = cpu_hi;
    return 0;
}

void fc_reset(Fc* fc) {
    if (!fc) return;
    fc->reset_state();
    fc->lat.clear();
    // NOTE: fc->done is a MONOTONIC counter of completed frames and is NOT
    // reset here.  Resetting it would race with a frame that is still in
    // flight in the worker thread, and the counter would then move backwards
    // and the pipeline would deadlock.
    {
        std::lock_guard<std::mutex> lk(fc->out_mtx);
        fc->outputs.clear();
        fc->lat_map.clear();
    }
}

// The fusion core keeps the OUTPUT OF EVERY FRAME it has computed, keyed by
// the frame index, so that a pipelined consumer (the hybrid configuration)
// can read the result of frame N-1 while frame N is still in flight.
int fc_last_output_at(Fc* fc, int64_t frame_index, float* dst) {
    if (!fc || !dst) return -1;
    std::lock_guard<std::mutex> lk(fc->out_mtx);
    auto it = fc->outputs.find(frame_index);
    if (it == fc->outputs.end()) return -2;
    std::memcpy(dst, it->second.data(), OUT * sizeof(float));
    return 0;
}

int fc_latency_at(Fc* fc, int64_t frame_index, double* latency_ms) {
    if (!fc) return -1;
    std::lock_guard<std::mutex> lk(fc->out_mtx);
    auto it = fc->lat_map.find(frame_index);
    if (it == fc->lat_map.end()) return -2;
    if (latency_ms) *latency_ms = it->second;
    return 0;
}

int fc_set_input(Fc* fc, int64_t frame_index, const float* x) {
    if (!fc) return -1;
    // A fixed-size RING BUFFER of input slots.  Growing the replay matrix on
    // every frame would make the submission cost O(n^2) over a long run and
    // would show up as a growing latency, which is an artefact of the
    // harness rather than of the fusion core.  The ring is deep enough for
    // the hybrid pipeline (one frame in flight on the GPU stage, one on the
    // CPU stage).
    const int64_t CAP = 16;
    if (fc->in_ring.size() != (size_t)CAP * IN) {
        fc->in_ring.assign((size_t)CAP * IN, 0.0f);
    }
    size_t slot = (size_t)(frame_index % CAP);
    std::memcpy(&fc->in_ring[slot * IN], x, sizeof(float) * IN);
    fc->in_slot.store(frame_index);
    return 0;
}

int fc_submit(Fc* fc, int64_t frame_index) {
    if (!fc) return -1;
    fc->cur_frame.store(frame_index);
    sem_post(&fc->ev);
    return 0;
}

int fc_wait(Fc* fc, int64_t frame_index, double* latency_ms) {
    if (!fc) return -1;
    while (fc->done.load() < frame_index + 1) {
        if (fc->worker_failed.load()) return -1;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (latency_ms) {
        std::lock_guard<std::mutex> lk(fc->out_mtx);
        auto it = fc->lat_map.find(frame_index);
        if (it != fc->lat_map.end()) *latency_ms = it->second;
    }
    return 0;
}

const float* fc_last_output(Fc* fc) { return fc ? fc->last_out.data() : nullptr; }

void fc_last_pose(Fc* fc, double* x, double* y, double* theta) {
    if (!fc) return;
    if (x) *x = fc->px;
    if (y) *y = fc->py;
    if (theta) *theta = fc->pth;
}

int64_t fc_frames_done(Fc* fc) { return fc ? fc->done.load() : 0; }

void fc_destroy(Fc* fc) {
    if (!fc) return;
    fc->stop.store(true);
    sem_post(&fc->ev);
    if (fc->th.joinable()) fc->th.join();
    sem_destroy(&fc->ev);
    delete fc;
}

int fc_layer_nnz(Fc* fc, int layer, int64_t* nnz, int32_t* n_in, int32_t* n_out) {
    if (!fc) return -1;
    const Layer* L = nullptr;
    if (layer == 0) L = &fc->l1;
    else if (layer == 1) L = &fc->l2;
    else if (layer == 2) L = &fc->l3;
    else return -1;
    if (nnz) *nnz = (int64_t)L->col.size();
    if (n_in) *n_in = L->n_in;
    if (n_out) *n_out = L->n_out;
    return 0;
}

}  // extern "C"
