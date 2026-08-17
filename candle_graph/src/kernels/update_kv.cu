#include <stdint.h>

template<typename T>
__device__ void copy2d(
  const uint64_t fingerprint,
  const T *src, T *dst,
  uint32_t d1, uint32_t d2,
  uint32_t src_o, uint32_t dst_o,
  uint32_t src_s, uint32_t dst_s
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  (dst + dst_o)[idx1 * dst_s + idx2] = (src + src_o)[idx1 * src_s + idx2];
}

// Same as `copy2d`, except the destination offset is read from `dst_o_ptr` at
// launch time instead of being passed as a plain scalar. A plain `dst_o` is a
// kernel-launch argument, so once a call is captured into a CUDA graph its
// value is frozen for every future replay: `Cache::append`'s current-seq-len
// offset would stay pinned at whatever it was during the one-time capture,
// and every replay would overwrite the same slot instead of advancing. Since
// `dst_o_ptr` is a device pointer, the *pointer* is what gets baked into the
// graph, not the value at that address -- writing a new value to `*dst_o_ptr`
// between replays (see `copy_inplace`) changes what the next replay writes to
// without re-capturing anything.
template<typename T>
__device__ void copy2d_dynoffset(
  const uint64_t fingerprint,
  const T *src, T *dst,
  uint32_t d1, uint32_t d2,
  uint32_t src_o, uint32_t dst_o_base, uint32_t dst_o_stride,
  uint32_t src_s, uint32_t dst_s,
  const uint32_t *dst_o_ptr
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  uint32_t dst_o = dst_o_base + (*dst_o_ptr) * dst_o_stride;
  (dst + dst_o)[idx1 * dst_s + idx2] = (src + src_o)[idx1 * src_s + idx2];
}

#define COPY2D_OP(TYPENAME, FNNAME) \
extern "C" __global__ \
void FNNAME( \
  const uint64_t fingerprint, \
  const TYPENAME *src, TYPENAME *dst, \
  uint32_t d1, uint32_t d2, \
  uint32_t src_o, uint32_t dst_o, \
  uint32_t src_s, uint32_t dst_s \
) { \
  copy2d(fingerprint, src, dst, d1, d2, src_o, dst_o, src_s, dst_s); \
} \

#define COPY2D_DYNOFFSET_OP(TYPENAME, FNNAME) \
extern "C" __global__ \
void FNNAME( \
  const uint64_t fingerprint, \
  const TYPENAME *src, TYPENAME *dst, \
  uint32_t d1, uint32_t d2, \
  uint32_t src_o, uint32_t dst_o_base, uint32_t dst_o_stride, \
  uint32_t src_s, uint32_t dst_s, \
  const uint32_t *dst_o_ptr \
) { \
  copy2d_dynoffset(fingerprint, src, dst, d1, d2, src_o, dst_o_base, dst_o_stride, src_s, dst_s, dst_o_ptr); \
} \

COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(uint8_t, copy2d_u8)
COPY2D_OP(uint32_t, copy2d_u32)
COPY2D_OP(int16_t, copy2d_i16)
COPY2D_OP(int32_t, copy2d_i32)
COPY2D_OP(int64_t, copy2d_i64)

COPY2D_DYNOFFSET_OP(float, copy2d_dynoffset_f32)
COPY2D_DYNOFFSET_OP(double, copy2d_dynoffset_f64)
COPY2D_DYNOFFSET_OP(uint8_t, copy2d_dynoffset_u8)
COPY2D_DYNOFFSET_OP(uint32_t, copy2d_dynoffset_u32)
COPY2D_DYNOFFSET_OP(int16_t, copy2d_dynoffset_i16)
COPY2D_DYNOFFSET_OP(int32_t, copy2d_dynoffset_i32)
COPY2D_DYNOFFSET_OP(int64_t, copy2d_dynoffset_i64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
COPY2D_OP(__half, copy2d_f16)
COPY2D_DYNOFFSET_OP(__half, copy2d_dynoffset_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
COPY2D_OP(__nv_bfloat16, copy2d_bf16)
COPY2D_DYNOFFSET_OP(__nv_bfloat16, copy2d_dynoffset_bf16)
#endif
