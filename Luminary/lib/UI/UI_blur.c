#include <stdlib.h>
#include <math.h>
#include "UI_blur.h"

#define SCALE 5
#define BLUR_WIDTH UI_WIDTH / SCALE
#define BLUR_HEIGHT UI_HEIGHT / SCALE

#define SIGMA 3.0f
#define STRENGTH (8.0f)
#define KERNEL_SIZE ((int) (SIGMA * 3.0f) + 1)

void blur_background(UI* ui, uint8_t* target, int width, int height) {
  const int off = 3 * width * ui->y + 3 * ui->x;
  float* temp   = (float*) ui->scratch;

  float* gauss_kernel         = temp + (BLUR_HEIGHT * BLUR_WIDTH * 3);
  uint8_t* target_convolution = (uint8_t*) (gauss_kernel + KERNEL_SIZE);

  for (int i = 0; i < BLUR_HEIGHT; i++) {
    for (int j = 0; j < BLUR_WIDTH; j++) {
      target_convolution[i * BLUR_WIDTH * 3 + j * 3] =
        target[off + SCALE * i * width * 3 + 3 * SCALE * j];
      target_convolution[i * BLUR_WIDTH * 3 + j * 3 + 1] =
        target[off + SCALE * i * width * 3 + 3 * SCALE * j + 1];
      target_convolution[i * BLUR_WIDTH * 3 + j * 3 + 2] =
        target[off + SCALE * i * width * 3 + 3 * SCALE * j + 2];
    }
  }

  const float gauss_factor1 = 1.0f / (2.0f * SIGMA * SIGMA);
  const float gauss_factor2 = gauss_factor1 / 3.14159265358979323846f;

  for (int i = 0; i < KERNEL_SIZE; i++) {
    gauss_kernel[i] = STRENGTH * gauss_factor2 * expf(-i * i * gauss_factor1);
  }

  for (int i = 0; i < BLUR_HEIGHT; i++) {
    for (int j = 0; j < BLUR_WIDTH * 3; j++) {
      float collector = 0;

      const int start = max(-KERNEL_SIZE + 1, -i);
      const int end   = min(KERNEL_SIZE, BLUR_HEIGHT - i);

      for (int k = -KERNEL_SIZE + 1; k < start; k++) {
        collector += gauss_kernel[abs(k)] * target_convolution[(i + start) * BLUR_WIDTH * 3 + j];
      }

      for (int k = start; k < end; k++) {
        collector += gauss_kernel[abs(k)] * target_convolution[(i + k) * BLUR_WIDTH * 3 + j];
      }

      for (int k = end; k < KERNEL_SIZE; k++) {
        collector += gauss_kernel[abs(k)] * target_convolution[(i + end - 1) * BLUR_WIDTH * 3 + j];
      }

      temp[i * BLUR_WIDTH * 3 + j] = fminf(255.0f, collector);
    }
  }

  for (int i = 0; i < BLUR_HEIGHT; i++) {
    for (int j = 0; j < BLUR_WIDTH * 3; j++) {
      float collector = 0;

      const int start = max(-KERNEL_SIZE + 1, -j / 3);
      const int end   = min(KERNEL_SIZE, BLUR_WIDTH - j / 3);

      for (int k = -KERNEL_SIZE + 1; k < start; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * BLUR_WIDTH * 3 + (j + start * 3)];
      }

      for (int k = start; k < end; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * BLUR_WIDTH * 3 + (j + k * 3)];
      }

      for (int k = end; k < KERNEL_SIZE; k++) {
        collector += gauss_kernel[abs(k)] * temp[i * BLUR_WIDTH * 3 + (j + (end - 1) * 3)];
      }

      target_convolution[i * BLUR_WIDTH * 3 + j] = fminf(255.0f, collector);
    }
  }

  for (int i = 0; i < UI_HEIGHT; i++) {
    for (int j = 0; j < UI_WIDTH; j++) {
      target[off + i * width * 3 + 3 * j] =
        target_convolution[(i / SCALE) * BLUR_WIDTH * 3 + 3 * (j / SCALE)];
      target[off + i * width * 3 + 3 * j + 1] =
        target_convolution[(i / SCALE) * BLUR_WIDTH * 3 + 3 * (j / SCALE) + 1];
      target[off + i * width * 3 + 3 * j + 2] =
        target_convolution[(i / SCALE) * BLUR_WIDTH * 3 + 3 * (j / SCALE) + 2];
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
