#include "backend/metal/MetalEmitter.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {

GeneratedKernel emitRMSNorm(const RMSNormOp &op, std::size_t threadsPerThreadgroup) {
  op.validate();
  const std::size_t elementCount = op.input.elementCount();
  const std::size_t width = op.input.shape.back();
  if (threadsPerThreadgroup != 64 && threadsPerThreadgroup != 128 &&
      threadsPerThreadgroup != 256) {
    throw std::invalid_argument("V1 RMSNorm supports 64, 128, or 256 threads.");
  }
  const std::size_t kThreads = threadsPerThreadgroup;
  const char *elementType = op.input.dtype == DType::Float16 ? "half" : "float";
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
  source << "\nkernel void rms_norm(device const " << elementType << " *input [[buffer(0)]],\n"
         << "                     device const " << elementType << " *weight [[buffer(1)]],\n"
         << "                     device " << elementType << " *output [[buffer(2)]],\n"
         << R"metal(                     uint tid [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {
  const uint rowOffset = group.x * kWidth;
  threadgroup float partial[kThreads];

  float sumSquares = 0.0f;
  for (ulong column = tid; column < kWidth; column += kThreads) {
    const float x = float(input[rowOffset + column]);
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
        float(input[rowOffset + column]) * inverseRms * float(weight[column]);
  }
}
)metal";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
