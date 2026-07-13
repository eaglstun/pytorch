#pragma once
#include <c10/metal/common.h>

#define MAX_THREADGROUP_SIZE static_cast<uint32_t>(1024)
C10_METAL_CONSTEXPR uint32_t SUM_NCHAINS = 8;
// Reduced extents below this threshold dispatch the thread-per-output-element
// `_small` kernels instead of the simdgroup/threadgroup-per-output ones,
// whose vectorized loops need at least 32 * SUM_NCHAINS elements per output
// to execute at all (below that, most lanes idle and the simd shuffle
// dominates the cost of adding a handful of elements).
C10_METAL_CONSTEXPR uint32_t SUM_SMALL_REDUCTION_THRESHOLD = 32 * SUM_NCHAINS;
// For middle-dim reductions of a contiguous tensor (the innermost dim is
// kept), adjacent threads of the thread-per-output kernel read adjacent
// addresses, so it stays fully coalesced at any reduced extent and beats
// the generic threadgroup-per-output kernel as long as there are enough
// output elements to occupy the GPU. This is that occupancy floor.
C10_METAL_CONSTEXPR uint32_t SUM_THREAD_PER_OUTPUT_MIN_OUTPUTS = 2048;
// The opposite pathology: fewer outputs than this with a large reduced
// extent cannot fill the GPU with one threadgroup per output (e.g.
// per-channel parameter gradients reduce 33M elements to 32 outputs). Such
// reductions are split across chunks with a partials buffer and second pass.
C10_METAL_CONSTEXPR uint32_t SUM_SPLIT_MAX_OUTPUTS = 1024;
// Rough total number of pass-1 threadgroups the split path aims for.
C10_METAL_CONSTEXPR uint32_t SUM_SPLIT_TARGET_TGS = 4096;

template <unsigned N = c10::metal::max_ndim>
struct NormParams {
  float p;
  uint32_t reduction_size;
  uint32_t ndim;

  ::c10::metal::array<uint32_t, N> input_sizes;
  ::c10::metal::array<uint32_t, N> input_strides;

  ::c10::metal::array<uint32_t, N> output_sizes;
  ::c10::metal::array<uint32_t, N> output_strides;
};
