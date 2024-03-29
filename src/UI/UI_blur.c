#include "UI_blur.h"

#include <math.h>
#include <stdlib.h>

#include "UI.h"

#define SCALE 5
#define BLUR_WIDTH UI_WIDTH / SCALE
#define BLUR_HEIGHT (UI_HEIGHT + UI_BORDER_SIZE) / SCALE

#define SIGMA 3.0f
#define STRENGTH 8.0f
#define KERNEL_SIZE ((int) (SIGMA * 3.0f) + 1)

#define SHIFT 11
#define FACTOR 2048

void blur_background(UI* ui, uint8_t* target, int width, int height, int ld) {
  const int off = 4 * ld * ui->y + 4 * ui->x;
  float* temp   = (float*) ui->scratch;

  const int vis_width  = min(width - ui->x, UI_WIDTH);
  const int vis_height = min(height - ui->y, UI_HEIGHT + UI_BORDER_SIZE);

  const int b_width  = vis_width / SCALE;
  const int b_height = vis_height / SCALE;

  float* gauss_kernel = temp + (b_height * b_width * 3);
  uint8_t* internal   = (uint8_t*) (gauss_kernel + KERNEL_SIZE);

  for (int i = 0; i < b_height; i++) {
    for (int j = 0; j < b_width; j++) {
      internal[i * b_width * 3 + j * 3] =
        (((int) target[off + 4 * (SCALE * i * width + SCALE * j) + 2]
          + (int) target[off + 4 * ((SCALE * i + SCALE / 2) * width + (SCALE * j + SCALE / 2)) + 2])
         >> 1);
      internal[i * b_width * 3 + j * 3 + 1] =
        (((int) target[off + 4 * (SCALE * i * width + SCALE * j) + 1]
          + (int) target[off + 4 * ((SCALE * i + SCALE / 2) * width + (SCALE * j + SCALE / 2)) + 1])
         >> 1);
      internal[i * b_width * 3 + j * 3 + 2] =
        (((int) target[off + 4 * (SCALE * i * width + SCALE * j)]
          + (int) target[off + 4 * ((SCALE * i + SCALE / 2) * width + (SCALE * j + SCALE / 2))])
         >> 1);
    }
  }

  const float gauss_factor1 = 1.0f / (2.0f * SIGMA * SIGMA);
  const float gauss_factor2 = gauss_factor1 / 3.14159265358979323846f;

  for (int i = 0; i < KERNEL_SIZE; i++) {
    gauss_kernel[i] = STRENGTH * gauss_factor2 * expf(-i * i * gauss_factor1);
  }

  for (int i = 0; i < b_height; i++) {
    for (int j = 0; j < b_width * 3; j++) {
      float collector = 0;

      const int start = max(-KERNEL_SIZE + 1, -i);
      const int end   = min(KERNEL_SIZE, b_height - i);

      for (int k = -KERNEL_SIZE + 1; k < start; k++) {
        collector += gauss_kernel[abs(k)] * internal[(i + start) * b_width * 3 + j];
      }

      for (int k = start; k < end; k++) {
        collector += gauss_kernel[abs(k)] * internal[(i + k) * b_width * 3 + j];
      }

      for (int k = end; k < KERNEL_SIZE; k++) {
        collector += gauss_kernel[abs(k)] * internal[(i + end - 1) * b_width * 3 + j];
      }

      temp[i * b_width * 3 + j] = fminf(255.0f, collector);
    }
  }

  for (int i = 0; i < b_height; i++) {
    for (int j = 0; j < b_width * 3; j++) {
      float collector = 0;

      const int start = max(-KERNEL_SIZE + 1, -j / 3);
      const int end   = min(KERNEL_SIZE, b_width - j / 3);

      for (int k = -KERNEL_SIZE + 1; k < start; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * b_width * 3 + (j + start * 3)];
      }

      for (int k = start; k < end; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * b_width * 3 + (j + k * 3)];
      }

      for (int k = end; k < KERNEL_SIZE; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * b_width * 3 + (j + (end - 1) * 3)];
      }

      internal[i * b_width * 3 + j] = fminf(255.0f, collector);
    }
  }

  const int scale_x = (int) ((double) FACTOR * (b_width - 1) / vis_width + 0.5);
  const int scale_y = (int) ((double) FACTOR * (b_height - 1) / (vis_height) + 0.5);

  for (int i = 0; i < vis_height; i++) {
    const int source_y = i * scale_y;
    int y0             = source_y >> SHIFT;
    const int fy       = source_y - (y0 << SHIFT);
    const int ify      = FACTOR - fy;

    for (int j = 0; j < vis_width; j++) {
      const int source_x = j * scale_x;
      int x0             = source_x >> SHIFT;
      const int fx       = source_x - (x0 << SHIFT);
      const int ifx      = FACTOR - fx;

      const int index0 = 3 * (y0 * b_width + x0);
      const int index1 = 3 * (y0 * b_width + x0 + 1);
      const int index2 = 3 * ((y0 + 1) * b_width + x0);
      const int index3 = 3 * ((y0 + 1) * b_width + x0 + 1);

      target[off + i * ld * 4 + 4 * j + 2] =
        (uint8_t) ((internal[index0] * ifx * ify + internal[index1] * fx * ify + internal[index2] * ifx * fy + internal[index3] * fx * fy + (FACTOR * FACTOR / 2)) >> (2 * SHIFT));

      target[off + i * ld * 4 + 4 * j + 1] =
        (uint8_t) ((internal[index0 + 1] * ifx * ify + internal[index1 + 1] * fx * ify + internal[index2 + 1] * ifx * fy + internal[index3 + 1] * fx * fy + (FACTOR * FACTOR / 2)) >> (2 * SHIFT));

      target[off + i * ld * 4 + 4 * j] =
        (uint8_t) ((internal[index0 + 2] * ifx * ify + internal[index1 + 2] * fx * ify + internal[index2 + 2] * ifx * fy + internal[index3 + 2] * fx * fy + (FACTOR * FACTOR / 2)) >> (2 * SHIFT));
    }
  }
}

size_t blur_scratch_needed() {
  size_t val = 0;
  // temp
  val += sizeof(float) * BLUR_WIDTH * BLUR_HEIGHT * 3;
  // kernel
  val += sizeof(float) * KERNEL_SIZE;
  // convolution
  val += sizeof(uint8_t) * BLUR_WIDTH * BLUR_HEIGHT * 3;

  return val;
}
