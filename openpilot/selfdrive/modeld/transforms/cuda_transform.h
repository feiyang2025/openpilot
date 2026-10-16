#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// All state for the CUDA transform pipeline
typedef struct {
  // Device buffers for Y/U/V transform outputs
  uint8_t *d_y, *d_u, *d_v;
  // Staging copy of the incoming NV12 frame. The caller passes a HOST pointer;
  // the kernels must never dereference it directly, because that only works if the
  // memory happens to be registered as device-mapped, which is not true for an
  // ordinary allocation and cannot be forced from Python on this platform (the
  // driver API reports no usable device there while the runtime API works fine).
  uint8_t *d_frame;
  int frame_bytes;             // capacity of d_frame
  // Temporal ring buffer (holds previous frames)
  uint8_t *d_img_buffer;
  // Output buffer (2 frames concatenated, same layout as OpenCL input_frames_cl)
  uint8_t *d_output;
  // Device copy of projection matrix
  float *d_proj_y, *d_proj_uv;

  int width, height;           // Model width/height (e.g. 512x256)
  int temporal_skip;           // Number of temporal skip frames
  int frame_size;              // MODEL_FRAME_SIZE
  int buf_size;                // frame_size * 2
  // Element type of d_output: 0 = uint8, 1 = fp16. The temporal ring buffer stays
  // uint8 either way; only the tensor handed to the engine is widened. A model
  // whose engine binds img as fp16 (the supercombo/TGC graphs, once their UINT8
  // inputs are rewritten) needs this, and a uint8-sized buffer reinterpreted as
  // fp16 would be read twice past its end.
  int out_dtype;               // 0 = uint8, 1 = fp16

  // Whether the state has been initialized
  int initialized;
} CUDATransformState;

// Initialize CUDA transform state. out_dtype: 0 = uint8 output, 1 = fp16 output.
void cuda_transform_init(CUDATransformState *s, int width, int height, int temporal_skip,
                         int out_dtype);

// Destroy CUDA transform state and free all GPU memory
void cuda_transform_destroy(CUDATransformState *s);

// Execute the full transform pipeline:
//   input_nv12_host - HOST pointer to the NV12 frame (copied into the device
//                     staging buffer first; it is not dereferenced by the kernels)
//   frame_width, frame_height, frame_stride, frame_uv_offset - source frame params
//   projection - 3x3 perspective transform matrix
//
// After this call:
//   s->d_output contains 2 temporal frames in model input format
//   Get the device pointer via cuda_transform_get_output(s)
uint8_t* cuda_transform_execute(CUDATransformState *s,
                                 const uint8_t *input_nv12_host,
                                 int frame_width, int frame_height,
                                 int frame_stride, int frame_uv_offset,
                                 const float *projection);

// Get the output buffer device pointer (for from_blob)
uint8_t* cuda_transform_get_output(CUDATransformState *s);

#ifdef __cplusplus
}
#endif
