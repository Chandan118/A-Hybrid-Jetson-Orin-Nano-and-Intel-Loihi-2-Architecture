/*
 * C3 -- optimized C++ fusion core for the ARM CPU (shared library, C ABI).
 *
 * Requirements from the reviewer:
 *   - event-driven (the worker blocks on a semaphore; it does not poll)
 *   - same network structure as the original: 256 -> 512 -> 512 -> 64 LIF,
 *     25 time steps, beta = 0.9, threshold = 0.5
 *   - int8 weights
 *   - sparse arrays (CSR: row pointer + column index + int8 value)
 *   - -O3
 *   - pinned cores
 *   - same 50 Hz rate
 *
 * The same source is used by the ROS 2 node (fusion_core_node.cpp), so there
 * is exactly one implementation of the fusion core in the project.
 *
 * The weights are loaded from the same binary that the Python backends use, so
 * every configuration evaluates exactly the same function.
 */

#ifndef FUSION_CORE_H
#define FUSION_CORE_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fc Fc;

/* Create a core and load the int8/CSR weights from `weights_path`.
 * Returns NULL on failure. */
Fc* fc_create(const char* weights_path);

/* Supply the replay input matrix (float32, n_rows x dim) and the rate.
 * The library copies the matrix. */
int fc_configure(Fc* fc, const float* inputs, int64_t n_rows, int dim,
                 int rate_hz, int cpu_lo, int cpu_hi);

/* Reset the LIF membrane state and the odometry integration. */
void fc_reset(Fc* fc);

/* Write one frame's 256-dim input into the replay matrix slot `frame_index`.
 * The event-driven worker then reads it when the frame is submitted. */
int fc_set_input(Fc* fc, int64_t frame_index, const float* x);

/* Event-driven submit: hands one frame to the worker and returns at once. */
int fc_submit(Fc* fc, int64_t frame_index);

/* Block until the frame submitted as `frame_index` has been computed.
 * On return, *latency_ms holds the pure compute latency of that frame.
 * Returns 0 on success, -1 if the worker failed. */
int fc_wait(Fc* fc, int64_t frame_index, double* latency_ms);

/* The 64-dim spike output of the most recently computed frame. */
const float* fc_last_output(Fc* fc);

/* The 64-dim spike output of a SPECIFIC frame, for pipelined consumers that
 * read frame N-1 while frame N is still in flight.  Returns 0 on success,
 * -2 if that frame has not been computed yet. */
int fc_last_output_at(Fc* fc, int64_t frame_index, float* dst);

/* The pure compute latency of a specific frame. */
int fc_latency_at(Fc* fc, int64_t frame_index, double* latency_ms);

/* Current pose (x, y, theta) of the most recently computed frame. */
void fc_last_pose(Fc* fc, double* x, double* y, double* theta);

/* Number of frames completed so far. */
int64_t fc_frames_done(Fc* fc);

void fc_destroy(Fc* fc);

/* Introspection for the report: number of stored int8 weights per layer. */
int fc_layer_nnz(Fc* fc, int layer, int64_t* nnz, int32_t* n_in, int32_t* n_out);

#ifdef __cplusplus
}
#endif
#endif /* FUSION_CORE_H */
