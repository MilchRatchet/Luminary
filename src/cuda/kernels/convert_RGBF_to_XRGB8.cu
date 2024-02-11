#include "camera_post_common.cuh"
#include "math.cuh"
#include "purkinje.cuh"
#include "tonemap.cuh"
#include "utils.cuh"

__global__ void convert_RGBF_to_XRGB8(
  const RGBF* source, XRGB8* dest, const int width, const int height, const int ld, const OutputVariable output_variable) {
  unsigned int id = THREAD_ID;

  const int amount    = width * height;
  const float scale_x = 1.0f / (width - 1);
  const float scale_y = 1.0f / (height - 1);

  const int src_width  = (output_variable == OUTPUT_VARIABLE_BEAUTY) ? device.output_width : device.width;
  const int src_height = (output_variable == OUTPUT_VARIABLE_BEAUTY) ? device.output_height : device.height;

  while (id < amount) {
    const int x = id % width;
    const int y = id / width;

    const float sx = x * scale_x;
    const float sy = y * scale_y;

    RGBF pixel = sample_pixel_clamp(source, sx, sy, src_width, src_height);

    if (output_variable != OUTPUT_VARIABLE_ALBEDO_GUIDANCE && output_variable != OUTPUT_VARIABLE_NORMAL_GUIDANCE) {
      if (device.scene.camera.purkinje) {
        pixel = purkinje_shift(pixel);
      }

      if (device.scene.camera.use_color_correction) {
        RGBF hsv = rgb_to_hsv(pixel);

        hsv = add_color(hsv, device.scene.camera.color_correction);

        if (hsv.r < 0.0f)
          hsv.r += 1.0f;
        if (hsv.r > 1.0f)
          hsv.r -= 1.0f;
        hsv.g = __saturatef(hsv.g);
        if (hsv.b < 0.0f)
          hsv.b = 0.0f;

        pixel = hsv_to_rgb(hsv);
      }

      pixel.r *= device.scene.camera.exposure;
      pixel.g *= device.scene.camera.exposure;
      pixel.b *= device.scene.camera.exposure;

      switch (device.scene.camera.tonemap) {
        case TONEMAP_NONE:
          break;
        case TONEMAP_ACES:
          pixel = tonemap_aces(pixel);
          break;
        case TONEMAP_REINHARD:
          pixel = tonemap_reinhard(pixel);
          break;
        case TONEMAP_UNCHARTED2:
          pixel = tonemap_uncharted2(pixel);
          break;
        case TONEMAP_AGX:
          pixel = tonemap_agx(pixel);
          break;
        case TONEMAP_AGX_PUNCHY:
          pixel = tonemap_agx_punchy(pixel);
          break;
        case TONEMAP_AGX_CUSTOM:
          pixel = tonemap_agx_custom(pixel);
          break;
      }

      switch (device.scene.camera.filter) {
        case FILTER_NONE:
          break;
        case FILTER_GRAY:
          pixel = filter_gray(pixel);
          break;
        case FILTER_SEPIA:
          pixel = filter_sepia(pixel);
          break;
        case FILTER_GAMEBOY:
          pixel = filter_gameboy(pixel, x, y);
          break;
        case FILTER_2BITGRAY:
          pixel = filter_2bitgray(pixel, x, y);
          break;
        case FILTER_CRT:
          pixel = filter_crt(pixel, x, y);
          break;
        case FILTER_BLACKWHITE:
          pixel = filter_blackwhite(pixel, x, y);
          break;
      }
    }

    const float dither = (device.scene.camera.dithering) ? random_dither_mask(x, y) : 0.0f;

    pixel.r = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.r));
    pixel.g = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.g));
    pixel.b = fminf(255.9f, dither + 255.9f * linearRGB_to_SRGB(pixel.b));

    XRGB8 converted_pixel;
    converted_pixel.ignore = 0;
    converted_pixel.r      = (uint8_t) pixel.r;
    converted_pixel.g      = (uint8_t) pixel.g;
    converted_pixel.b      = (uint8_t) pixel.b;

    dest[x + y * ld] = converted_pixel;

    id += blockDim.x * gridDim.x;
  }
}
