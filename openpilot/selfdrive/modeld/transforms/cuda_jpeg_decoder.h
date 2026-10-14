#ifndef CUDA_JPEG_DECODER_H
#define CUDA_JPEG_DECODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  int frame_size;       // width * height * 3 / 2  (NV12)
  uint8_t *d_rgb;       // device interleaved RGB (nvJPEG output)
  uint8_t *d_nv12;      // device NV12 output (after RGB->NV12 conversion)
  void *nvjpeg_handle;  // nvjpegHandle_t
  void *nvjpeg_state;   // nvjpegJpegState_t
  int initialized;
  int nvjpeg_error;     // last error code
  int owns_nv12;        // 1 = decoder allocated d_nv12, 0 = external
} JPEGDecoderState;

// Create decoder for given frame dimensions
void jpeg_decoder_init(JPEGDecoderState *s, int width, int height);

// Set an external NV12 output buffer (e.g., exported via CUDA IPC).
// If not called, decoder allocates its own internal buffer.
void jpeg_decoder_set_output(JPEGDecoderState *s, uint8_t *d_nv12, int frame_size);

// Decode JPEG to NV12. Returns device pointer to NV12 buffer.
// jpeg_data must be host (CPU) memory. Returns NULL on failure.
uint8_t* jpeg_decoder_decode(JPEGDecoderState *s, const uint8_t *jpeg_data, size_t jpeg_size);

// Get the device pointer to the latest decoded frame
uint8_t* jpeg_decoder_get_output(JPEGDecoderState *s);

// Debug: returns CUDA device index, -1 on failure
int cuda_test_device();

// Cleanup
void jpeg_decoder_destroy(JPEGDecoderState *s);

#ifdef __cplusplus
}
#endif

#endif
