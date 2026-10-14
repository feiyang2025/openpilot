#include "cuda_jpeg_decoder.h"
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <cuda.h>

// CUDA kernel: convert RGB interleaved to NV12 (YUV 4:2:0)
// NV12: Y plane (w*h) + UV interleaved (w*h/2)
// U and V are subsampled 2x horizontally and 2x vertically
// For each 2x2 RGB block, average the U and V values
__global__ void rgb_to_nv12_kernel(const uint8_t *rgb, uint8_t *y, uint8_t *uv,
    int width, int height, int rgb_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y_idx >= height) return;

    // Read pixel
    int rgb_idx = y_idx * rgb_pitch + x * 3;
    uint8_t r = rgb[rgb_idx];
    uint8_t g = rgb[rgb_idx + 1];
    uint8_t b = rgb[rgb_idx + 2];

    // RGB to Y (BT.601)
    int y_val = (77 * r + 150 * g + 29 * b + 128) >> 8;
    if (y_val > 255) y_val = 255;
    y[y_idx * width + x] = (uint8_t)y_val;

    // UV at half resolution - only for even x and even y
    if ((x & 1) == 0 && (y_idx & 1) == 0) {
        // For each 2x2 block, compute average of the 4 pixels
        int r_sum = r, g_sum = g, b_sum = b;

        if (x + 1 < width) {
            int idx2 = y_idx * rgb_pitch + (x + 1) * 3;
            r_sum += rgb[idx2];
            g_sum += rgb[idx2 + 1];
            b_sum += rgb[idx2 + 2];
        }
        if (y_idx + 1 < height) {
            int idx3 = (y_idx + 1) * rgb_pitch + x * 3;
            r_sum += rgb[idx3];
            g_sum += rgb[idx3 + 1];
            b_sum += rgb[idx3 + 2];
        }
        if (x + 1 < width && y_idx + 1 < height) {
            int idx4 = (y_idx + 1) * rgb_pitch + (x + 1) * 3;
            r_sum += rgb[idx4];
            g_sum += rgb[idx4 + 1];
            b_sum += rgb[idx4 + 2];
        }

        // Count valid pixels in 2x2 block
        int count = 4;
        if (x + 1 >= width) count -= 2;
        if (y_idx + 1 >= height) count -= 2;
        if (x + 1 >= width && y_idx + 1 >= height) count = 1;

        // Average
        r = r_sum / count;
        g = g_sum / count;
        b = b_sum / count;

        // RGB to U and V (BT.601)
        int u_val = (-38 * r - 74 * g + 112 * b + 128) / 256 + 128;
        int v_val = (112 * r - 94 * g - 18 * b + 128) / 256 + 128;
        if (u_val < 0) u_val = 0;
        if (u_val > 255) u_val = 255;
        if (v_val < 0) v_val = 0;
        if (v_val > 255) v_val = 255;

        int uv_idx = (y_idx / 2) * width + x;
        uv[uv_idx * 2] = (uint8_t)u_val;
        uv[uv_idx * 2 + 1] = (uint8_t)v_val;
    }
}

extern "C" {

int cuda_test_device()
{
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) return -1;
    return dev;
}

void jpeg_decoder_init(JPEGDecoderState *s, int width, int height)
{
    s->width = width;
    s->height = height;
    s->frame_size = width * height * 3 / 2;
    s->d_rgb = NULL;
    s->d_nv12 = NULL;
    s->nvjpeg_handle = NULL;
    s->nvjpeg_state = NULL;
    s->initialized = 0;
    s->nvjpeg_error = 0;
    s->owns_nv12 = 1;

    // Allocate RGB buffer (interleaved RGB)
    CUdeviceptr p;
    if (cuMemAlloc_v2(&p, width * height * 3) != CUDA_SUCCESS) { s->nvjpeg_error = 10; return; }
    s->d_rgb = (uint8_t*)p;

    // Allocate NV12 output buffer
    if (cuMemAlloc_v2(&p, s->frame_size) != CUDA_SUCCESS) { s->nvjpeg_error = 11; return; }
    s->d_nv12 = (uint8_t*)p;

    // Create nvJPEG handle
    nvjpegHandle_t handle;
    nvjpegStatus_t jstatus = nvjpegCreateSimple(&handle);
    if (jstatus != NVJPEG_STATUS_SUCCESS) {
        s->nvjpeg_error = (int)jstatus;
        return;
    }
    s->nvjpeg_handle = (void*)handle;

    // Create decoder state
    nvjpegJpegState_t state;
    jstatus = nvjpegJpegStateCreate(handle, &state);
    if (jstatus != NVJPEG_STATUS_SUCCESS) {
        s->nvjpeg_error = (int)jstatus;
        return;
    }
    s->nvjpeg_state = (void*)state;

    s->initialized = 1;
}

void jpeg_decoder_set_output(JPEGDecoderState *s, uint8_t *d_nv12, int frame_size)
{
    // Free internal NV12 buffer if owned by us
    if (s->d_nv12 && s->owns_nv12) {
        cuMemFree_v2((CUdeviceptr)s->d_nv12);
    }
    s->d_nv12 = d_nv12;
    s->frame_size = frame_size;
    s->owns_nv12 = 0;
}

uint8_t* jpeg_decoder_decode(JPEGDecoderState *s, const uint8_t *jpeg_data, size_t jpeg_size)
{
    if (!s->initialized) return NULL;

    nvjpegHandle_t handle = (nvjpegHandle_t)s->nvjpeg_handle;
    nvjpegJpegState_t state = (nvjpegJpegState_t)s->nvjpeg_state;

    // Set up output image descriptor for RGB interleaved
    nvjpegImage_t img;
    img.pitch[0] = s->width * 3;  // RGB stride
    img.pitch[1] = 0;
    img.pitch[2] = 0;
    img.pitch[3] = 0;
    img.channel[0] = s->d_rgb;
    img.channel[1] = NULL;
    img.channel[2] = NULL;
    img.channel[3] = NULL;

    // Decode JPEG to RGB interleaved on GPU
    nvjpegStatus_t status = nvjpegDecode(handle, state,
        jpeg_data, jpeg_size, NVJPEG_OUTPUT_RGBI, &img, NULL);
    if (status != NVJPEG_STATUS_SUCCESS) {
        s->nvjpeg_error = (int)status;
        return NULL;
    }

    // Convert RGB to NV12 on GPU
    dim3 block(32, 8);
    dim3 grid((s->width + block.x - 1) / block.x, (s->height + block.y - 1) / block.y);
    rgb_to_nv12_kernel<<<grid, block>>>(s->d_rgb, s->d_nv12,
        s->d_nv12 + s->width * s->height,
        s->width, s->height, s->width * 3);
    {
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            s->nvjpeg_error = (int)cerr + 100;
            return NULL;
        }
    }

    // Sync
    CUresult err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        s->nvjpeg_error = (int)err + 200;
        return NULL;
    }

    return s->d_nv12;
}

uint8_t* jpeg_decoder_get_output(JPEGDecoderState *s)
{
    return s->d_nv12;
}

void jpeg_decoder_destroy(JPEGDecoderState *s)
{
    if (s->d_rgb) cuMemFree_v2((CUdeviceptr)s->d_rgb);
    if (s->d_nv12 && s->owns_nv12) cuMemFree_v2((CUdeviceptr)s->d_nv12);
    if (s->nvjpeg_state) {
        nvjpegJpegStateDestroy((nvjpegJpegState_t)s->nvjpeg_state);
    }
    if (s->nvjpeg_handle) {
        nvjpegDestroy((nvjpegHandle_t)s->nvjpeg_handle);
    }
    s->initialized = 0;
}

} // extern "C"
