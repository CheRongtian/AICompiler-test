// Fixed V0b baseline. Keep its 256-thread algorithm independent of candidate edits.
#include "backend/metal/MetalEmitter.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {

GeneratedKernel emitRMSNormBaseline(const RMSNormOp &op) {
  op.validate();
  const std::size_t elementCount = op.input.elementCount();
  const std::size_t width = op.input.shape.back();
  constexpr std::size_t kThreads = 256;
  if (elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal RMSNorm requires 32-bit tensor indices.");
  }

  GeneratedKernel kernel;
  kernel.functionName = "rms_norm";
  kernel.threadgroupCount = elementCount / width;
  kernel.threadsPerThreadgroup = kThreads;

  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "constant uint kWidth = " << width << "u;\n"
         << "constant uint kThreads = " << kThreads << "u;\n"
         << "constant float kEpsilon = " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << op.epsilon << "f;\n";
  source << R"metal(

kernel void rms_norm(device const float *input [[buffer(0)]],
                     device const float *weight [[buffer(1)]],
                     device float *output [[buffer(2)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {
  const uint rowOffset = group.x * kWidth;
  threadgroup float partial[kThreads];

  float sumSquares = 0.0f;
  for (ulong column = tid; column < kWidth; column += kThreads) {
    const float x = input[rowOffset + column];
    sumSquares += x * x;
  }
  partial[tid] = sumSquares;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // kThreads is a fixed power of two. All threads reach every barrier.
  for (uint stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inverseRms = rsqrt(partial[0] / float(kWidth) + kEpsilon);
  for (ulong column = tid; column < kWidth; column += kThreads) {
    output[rowOffset + column] =
        input[rowOffset + column] * inverseRms * weight[column];
  }
}
)metal";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
