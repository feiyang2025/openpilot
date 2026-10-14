// C ABI for trt_c_api.cc. Kept in a header so the Python side's expectations
// and the implementation cannot drift apart silently.

#ifndef TRT_C_API_H_
#define TRT_C_API_H_

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// nvinfer1::DataType values, so callers can map them without including NvInfer.h.
enum TrtDataType {
  TRT_DATA_TYPE_FLOAT = 0,
  TRT_DATA_TYPE_HALF = 1,
  TRT_DATA_TYPE_INT8 = 2,
  TRT_DATA_TYPE_INT32 = 3,
  TRT_DATA_TYPE_BOOL = 4,
  TRT_DATA_TYPE_UINT8 = 5,
  TRT_DATA_TYPE_FP8 = 6,
  TRT_DATA_TYPE_BF16 = 7,
};

// Opaque handle owning the runtime, engine and execution context. Null on
// failure; the reason is written to stderr.
void* trt_engine_create(const void* blob, size_t size);
void trt_engine_destroy(void* handle);

int32_t trt_engine_num_inputs(void* handle);
int32_t trt_engine_num_outputs(void* handle);

// Pointers stay valid until trt_engine_destroy.
const char* trt_engine_input_name(void* handle, int32_t index);
const char* trt_engine_output_name(void* handle, int32_t index);

// -1 on a bad index, otherwise a TrtDataType value.
int32_t trt_engine_input_dtype(void* handle, int32_t index);
int32_t trt_engine_output_dtype(void* handle, int32_t index);

int32_t trt_engine_input_rank(void* handle, int32_t index);
int32_t trt_engine_input_dim(void* handle, int32_t index, int32_t axis);
int32_t trt_engine_output_rank(void* handle, int32_t index);

// Returns the true rank even when `capacity` is too small, so the caller can
// detect the truncation instead of reading uninitialised memory.
int32_t trt_engine_output_shape(void* handle, int32_t index, int64_t* dims, int32_t capacity);

// 0 on success. `input_addrs` is indexed by input order and every entry must be
// a live device pointer. `output_dev` receives the first output.
int32_t trt_engine_execute(void* handle, const uint64_t* input_addrs, uint64_t output_dev, uint64_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRT_C_API_H_
