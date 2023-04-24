#ifndef CU_CAMERA_POST_LENS_FLARE_H
#define CU_CAMERA_POST_LENS_FLARE_H

#include "utils.h"

#define LENS_FLARE_NUM_BUFFERS 4

/*
 * I based this implementation on a nice blog post by Lena Piquet:
 * https://www.froyok.fr/blog/2021-09-ue4-custom-lens-flare/
 */

extern "C" void device_lens_flare_init(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  instance->lens_flare_buffers_gpu = (RGBAhalf**) malloc(sizeof(RGBAhalf*) * LENS_FLARE_NUM_BUFFERS);

  device_malloc((void**) &(instance->lens_flare_buffers_gpu[0]), sizeof(RGBAhalf) * width * height);
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[1]), sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[2]), sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_malloc((void**) &(instance->lens_flare_buffers_gpu[3]), sizeof(RGBAhalf) * (width >> 2) * (height >> 2));
}

__global__ void _lens_flare_ghosts(const RGBAhalf* source, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int ghost_count         = 8;
  const float ghost_scales[]    = {-1.0f, -0.5f, -0.25f, -2.0f, -3.0f, -4.0f, 2.0f, 0.25f};
  const RGBAhalf ghost_colors[] = {get_RGBAhalf(1.0f, 0.5f, 1.0f, 1.0f),   get_RGBAhalf(0.1f, 0.5f, 1.0f, 0.4f),
                                   get_RGBAhalf(1.0f, 1.0f, 0.5f, 0.2f),   get_RGBAhalf(1.0f, 0.75f, 0.1f, 0.7f),
                                   get_RGBAhalf(1.0f, 0.1f, 1.0f, 0.5f),   get_RGBAhalf(1.0f, 0.5f, 0.1f, 0.3f),
                                   get_RGBAhalf(0.25f, 0.1f, 0.75f, 0.5f), get_RGBAhalf(0.75f, 0.1f, 0.25f, 0.2f)};

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float sx = scale_x * x + 0.5f * scale_x;
    const float sy = scale_y * y + 0.5f * scale_y;

    RGBAhalf pixel = get_RGBAhalf(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < ghost_count; i++) {
      const float sxi = ((sx - 0.5f) * ghost_scales[i]) + 0.5f;
      const float syi = ((sy - 0.5f) * ghost_scales[i]) + 0.5f;

      RGBAhalf ghost = sample_pixel_border(source, sxi, syi, sw, sh);

      ghost = scale_RGBAhalf(ghost, 0.0005f);
      ghost = mul_RGBAhalf(ghost, ghost_colors[i]);

      pixel = add_RGBAhalf(pixel, ghost);
    }

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

static void _lens_flare_apply_ghosts(RaytraceInstance* instance, const RGBAhalf* src, RGBAhalf* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  _lens_flare_ghosts<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
}

__device__ UV _lens_flare_fisheye(UV uv, const float compression, const float zoom) {
  uv.u = 2.0f * uv.u - 1.0f;
  uv.v = 2.0f * uv.v - 1.0f;

  const float scale           = compression * atanf(1.0f / compression);
  const float radius_distance = sqrtf(uv.u * uv.u + uv.v * uv.v) * scale;
  const float radius_dir      = compression * tanf(radius_distance / compression) * zoom;
  const float phi             = atan2f(uv.v, uv.u);

  UV result;

  result.u = __saturatef(0.5f * (radius_dir * cosf(phi) + 1.0f));
  result.v = __saturatef(0.5f * (radius_dir * sinf(phi) + 1.0f));

  return result;
}

__global__ void _lens_flare_halo(const RGBAhalf* src, const int sw, const int sh, RGBAhalf* target, const int tw, const int th) {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  const float scale_x = 1.0f / (tw - 1);
  const float scale_y = 1.0f / (th - 1);

  while (id < tw * th) {
    const int x = id % tw;
    const int y = id / tw;

    const float width        = 0.45f;
    const float chroma_shift = 0.005f;

    UV src_uv;
    src_uv.u = scale_x * x + 0.5f * scale_x;
    src_uv.v = scale_y * y + 0.5f * scale_y;

    const UV fish_uv          = _lens_flare_fisheye(src_uv, 1.0f, 1.0f);
    const float2 neg_offset   = make_float2(0.5f - src_uv.u, 0.5f - src_uv.v);
    const float offset_length = sqrtf(neg_offset.x * neg_offset.x + neg_offset.y * neg_offset.y);
    const float factor        = width / fmaxf(eps, offset_length);
    const float2 v_halo       = make_float2(factor * neg_offset.x, factor * neg_offset.y);

    UV src_uv_r;
    src_uv_r.u = (fish_uv.u - 0.5f) * (1.0f + chroma_shift) + 0.5f + v_halo.x;
    src_uv_r.v = (fish_uv.v - 0.5f) * (1.0f + chroma_shift) + 0.5f + v_halo.y;

    UV src_uv_g;
    src_uv_g.u = fish_uv.u + v_halo.x;
    src_uv_g.v = fish_uv.v + v_halo.y;

    UV src_uv_b;
    src_uv_b.u = (fish_uv.u - 0.5f) * (1.0f - chroma_shift) + 0.5f + v_halo.x;
    src_uv_b.v = (fish_uv.v - 0.5f) * (1.0f - chroma_shift) + 0.5f + v_halo.y;

    const RGBAhalf pr = sample_pixel_border(src, src_uv_r.u, src_uv_r.v, sw, sh);
    const RGBAhalf pg = sample_pixel_border(src, src_uv_g.u, src_uv_g.v, sw, sh);
    const RGBAhalf pb = sample_pixel_border(src, src_uv_b.u, src_uv_b.v, sw, sh);

    RGBAhalf pixel = get_RGBAhalf(pr.rg.x, pg.rg.y, pb.ba.x, 0.0f);

    store_RGBAhalf(target + x + y * tw, bound_RGBAhalf(pixel));

    id += blockDim.x * gridDim.x;
  }
}

static void _lens_flare_apply_halo(RaytraceInstance* instance, const RGBAhalf* src, RGBAhalf* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  _lens_flare_halo<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(src, width >> 1, height >> 1, dst, width >> 1, height >> 1);
}

extern "C" void device_lens_flare_apply(RaytraceInstance* instance, const RGBAhalf* src, RGBAhalf* dst) {
  int width  = instance->output_width;
  int height = instance->output_height;

  if (!instance->lens_flare_buffers_gpu) {
    device_lens_flare_init(instance);

    if (!instance->lens_flare_buffers_gpu) {
      crash_message("Failed to initialize lens flare buffers.");
    }
  }

  image_downsample_threshold<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    src, width, height, instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, instance->scene.camera.lens_flare_threshold);

  image_downsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, instance->lens_flare_buffers_gpu[3], width >> 2, height >> 2);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[3], width >> 2, height >> 2, instance->lens_flare_buffers_gpu[2], instance->lens_flare_buffers_gpu[2],
    width >> 1, height >> 1, 0.5f, 0.5f);

  _lens_flare_apply_ghosts(instance, instance->lens_flare_buffers_gpu[2], instance->lens_flare_buffers_gpu[1]);
  _lens_flare_apply_halo(instance, instance->lens_flare_buffers_gpu[1], instance->lens_flare_buffers_gpu[2]);

  image_upsample<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    instance->lens_flare_buffers_gpu[2], width >> 1, height >> 1, dst, dst, width, height, 1.0f, 1.0f);
}

extern "C" void device_lens_flare_clear(RaytraceInstance* instance) {
  int width  = instance->output_width;
  int height = instance->output_height;

  device_free(instance->lens_flare_buffers_gpu[0], sizeof(RGBAhalf) * width * height);
  device_free(instance->lens_flare_buffers_gpu[1], sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_free(instance->lens_flare_buffers_gpu[2], sizeof(RGBAhalf) * (width >> 1) * (height >> 1));
  device_free(instance->lens_flare_buffers_gpu[3], sizeof(RGBAhalf) * (width >> 2) * (height >> 2));

  free(instance->lens_flare_buffers_gpu);
}

#endif /* CU_CAMERA_POST_LENS_FLARE_H */
