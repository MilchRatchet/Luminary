#include "ui_renderer_blur.h"

#include <immintrin.h>
#include <string.h>

#include "display.h"

#define BLUR_REDUCTION (4)
#define BLUR_BOX_LENGTH (10)
#define BLUR_PIXEL_SIZE (sizeof(LuminaryARGB8))

static uint32_t _ui_renderer_blur_reduced_dim(uint32_t length) {
  return length / BLUR_REDUCTION + (length % BLUR_REDUCTION != 0);
}

size_t ui_renderer_blur_buffer_size(uint32_t width, uint32_t height) {
  return (size_t) _ui_renderer_blur_reduced_dim(width) * _ui_renderer_blur_reduced_dim(height) * BLUR_PIXEL_SIZE;
}

static inline __m128i _ui_renderer_blur_load_channels(const uint8_t* src) {
  int32_t pixel;
  memcpy(&pixel, src, sizeof(pixel));
  return _mm_cvtepu8_epi16(_mm_cvtsi32_si128(pixel));
}

static void _ui_renderer_box_blur_line(
  const uint8_t* restrict src, uint8_t* restrict dst, uint32_t length, size_t src_step, size_t dst_step, uint32_t left) {
  const uint32_t right     = BLUR_BOX_LENGTH - 1 - left;
  const uint32_t last      = length - 1;
  const __m128i zero       = _mm_setzero_si128();
  const __m128i round      = _mm_set1_epi16(BLUR_BOX_LENGTH / 2);
  const __m128i reciprocal = _mm_set1_epi16(((1u << 16) + BLUR_BOX_LENGTH - 1) / BLUR_BOX_LENGTH);
  __m128i sums             = _mm_mullo_epi16(_ui_renderer_blur_load_channels(src), _mm_set1_epi16((short) (left + 1)));

  for (uint32_t offset = 1; offset <= right; ++offset) {
    const uint8_t* pixel = src + (offset < last ? offset : last) * src_step;
    sums                 = _mm_add_epi16(sums, _ui_renderer_blur_load_channels(pixel));
  }

  for (uint32_t index = 0; index < length; ++index) {
    const uint32_t remove_idx   = index > left ? index - left : 0;
    const uint32_t add_idx      = index + right + 1 < last ? index + right + 1 : last;
    const uint8_t* remove_pixel = src + remove_idx * src_step;
    const uint8_t* add_pixel    = src + add_idx * src_step;
    const __m128i average       = _mm_mulhi_epu16(_mm_add_epi16(sums, round), reciprocal);
    const int32_t pixel         = _mm_cvtsi128_si32(_mm_packus_epi16(average, zero));
    memcpy(dst + index * dst_step, &pixel, sizeof(pixel));

    const __m128i delta = _mm_sub_epi16(_ui_renderer_blur_load_channels(add_pixel), _ui_renderer_blur_load_channels(remove_pixel));
    sums                = _mm_add_epi16(sums, delta);
  }
}

static void _ui_renderer_downsample_blur_source(
  const uint8_t* restrict src, size_t src_ld, uint32_t width, uint32_t height, uint8_t* restrict dst, size_t dst_ld) {
  const __m256i zero  = _mm256_setzero_si256();
  const __m256i round = _mm256_set1_epi16(8);

  for (uint32_t row = 0; row < height; row += BLUR_REDUCTION, dst += dst_ld) {
    const uint8_t* lines[BLUR_REDUCTION];
    for (uint32_t offset = 0; offset < BLUR_REDUCTION; ++offset)
      lines[offset] = src + (row + offset < height ? row + offset : height - 1) * src_ld;

    uint32_t col    = 0;
    uint8_t* output = dst;
    for (; col + 2 * BLUR_REDUCTION <= width; col += 2 * BLUR_REDUCTION, output += 2 * BLUR_PIXEL_SIZE) {
      __m256i sum_low  = zero;
      __m256i sum_high = zero;
      for (uint32_t offset = 0; offset < BLUR_REDUCTION; ++offset) {
        const __m256i pixels = _mm256_loadu_si256((const __m256i*) (lines[offset] + col * BLUR_PIXEL_SIZE));
        sum_low              = _mm256_add_epi16(sum_low, _mm256_unpacklo_epi8(pixels, zero));
        sum_high             = _mm256_add_epi16(sum_high, _mm256_unpackhi_epi8(pixels, zero));
      }
      const __m256i pairs  = _mm256_add_epi16(sum_low, sum_high);
      const __m256i sums   = _mm256_add_epi16(pairs, _mm256_srli_si256(pairs, 8));
      const __m256i avg    = _mm256_srli_epi16(_mm256_add_epi16(sums, round), 4);
      const __m256i packed = _mm256_packus_epi16(avg, zero);
      const __m128i pixels = _mm_unpacklo_epi32(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
      _mm_storel_epi64((__m128i*) output, pixels);
    }

    for (; col < width; col += BLUR_REDUCTION, output += BLUR_PIXEL_SIZE) {
      uint32_t sums[BLUR_PIXEL_SIZE] = {0};
      for (uint32_t sample_row = 0; sample_row < BLUR_REDUCTION; ++sample_row) {
        for (uint32_t offset = 0; offset < BLUR_REDUCTION; ++offset) {
          const uint32_t sample_col = col + offset < width ? col + offset : width - 1;
          const uint8_t* pixel      = lines[sample_row] + sample_col * BLUR_PIXEL_SIZE;
          for (size_t channel = 0; channel < BLUR_PIXEL_SIZE; ++channel)
            sums[channel] += pixel[channel];
        }
      }
      for (size_t channel = 0; channel < BLUR_PIXEL_SIZE; ++channel)
        output[channel] = (uint8_t) ((sums[channel] + 8) >> 4);
    }
  }
}

static void _ui_renderer_expand_blur_row(const uint8_t* restrict src, uint32_t cols, uint8_t* restrict dst) {
  const __m128i zero  = _mm_setzero_si128();
  const __m128i round = _mm_set1_epi16(4);
  memcpy(dst, src, BLUR_PIXEL_SIZE);
  memcpy(dst + BLUR_PIXEL_SIZE, src, BLUR_PIXEL_SIZE);
  dst += 2 * BLUR_PIXEL_SIZE;

  for (uint32_t col = 1; col < cols; ++col) {
    const __m128i pixels = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) src), zero);
    const __m128i delta  = _mm_sub_epi16(_mm_srli_si128(pixels, 8), pixels);
    const __m128i step   = _mm_slli_epi16(delta, 1);
    const __m128i first  = _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(pixels, 3), delta), round);
    const __m128i second = _mm_add_epi16(first, step);
    const __m128i third  = _mm_add_epi16(second, step);
    const __m128i fourth = _mm_add_epi16(third, step);
    const __m128i low    = _mm_srli_epi16(_mm_unpacklo_epi64(first, second), 3);
    const __m128i high   = _mm_srli_epi16(_mm_unpacklo_epi64(third, fourth), 3);
    _mm_storeu_si128((__m128i*) dst, _mm_packus_epi16(low, high));
    src += BLUR_PIXEL_SIZE;
    dst += BLUR_REDUCTION * BLUR_PIXEL_SIZE;
  }

  memcpy(dst, src, BLUR_PIXEL_SIZE);
  memcpy(dst + BLUR_PIXEL_SIZE, src, BLUR_PIXEL_SIZE);
}

static void _ui_renderer_blend_blur_rows(
  const uint8_t* restrict upper, const uint8_t* restrict lower, uint8_t* restrict dst, size_t row_bytes, uint32_t weight) {
  const __m256i round   = _mm256_set1_epi16(4);
  const __m256i weights = _mm256_set1_epi16((short) ((weight << 8) | (8 - weight)));
  size_t offset         = 0;

  for (; offset + 32 <= row_bytes; offset += 32) {
    const __m256i top    = _mm256_loadu_si256((const __m256i*) (upper + offset));
    const __m256i bottom = _mm256_loadu_si256((const __m256i*) (lower + offset));
    const __m256i low    = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(top, bottom), weights);
    const __m256i high   = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(top, bottom), weights);
    const __m256i pixels =
      _mm256_packus_epi16(_mm256_srli_epi16(_mm256_add_epi16(low, round), 3), _mm256_srli_epi16(_mm256_add_epi16(high, round), 3));
    _mm256_storeu_si256((__m256i*) (dst + offset), pixels);
  }

  for (; offset < row_bytes; ++offset)
    dst[offset] = (uint8_t) (((8 - weight) * upper[offset] + weight * lower[offset] + 4) >> 3);
}

const uint8_t* ui_renderer_blur_rows_get(UIRendererBlurRows* blur, uint32_t row) {
  if (row < 2)
    return blur->expanded[0];

  const uint32_t upper_idx = (row - 2) / BLUR_REDUCTION;
  const uint32_t phase     = (row - 2) % BLUR_REDUCTION;
  const uint8_t* upper     = blur->expanded[upper_idx & 1];
  if (upper_idx + 1 == blur->rows)
    return upper;

  uint8_t* lower = blur->expanded[(upper_idx + 1) & 1];
  if (phase == 0) {
    const size_t reduced_ld = blur->cols * BLUR_PIXEL_SIZE;
    _ui_renderer_expand_blur_row(blur->reduced + (upper_idx + 1) * reduced_ld, blur->cols, lower);
  }

  _ui_renderer_blend_blur_rows(upper, lower, blur->blended, blur->row_bytes, 2 * phase + 1);
  return blur->blended;
}

static void _ui_renderer_blur_reduced(
  const uint8_t* restrict src, uint32_t width, uint32_t height, size_t src_ld, uint8_t* restrict dst, uint32_t cols, uint32_t rows) {
  const size_t reduced_ld = cols * BLUR_PIXEL_SIZE;
  uint8_t scratch_buffer[BLUR_MAX_SUPPORTED_DIM / BLUR_REDUCTION * BLUR_PIXEL_SIZE];
  _ui_renderer_downsample_blur_source(src, src_ld, width, height, dst, reduced_ld);

  for (uint32_t row = 0; row < rows; ++row) {
    uint8_t* line = dst + row * reduced_ld;
    _ui_renderer_box_blur_line(line, scratch_buffer, cols, BLUR_PIXEL_SIZE, BLUR_PIXEL_SIZE, 4);
    _ui_renderer_box_blur_line(scratch_buffer, line, cols, BLUR_PIXEL_SIZE, BLUR_PIXEL_SIZE, 5);
  }

  for (uint32_t col = 0; col < cols; ++col) {
    uint8_t* line = dst + col * BLUR_PIXEL_SIZE;
    _ui_renderer_box_blur_line(line, scratch_buffer, rows, reduced_ld, BLUR_PIXEL_SIZE, 4);
    _ui_renderer_box_blur_line(scratch_buffer, line, rows, BLUR_PIXEL_SIZE, reduced_ld, 5);
  }
}

bool ui_renderer_blur_rows_init(UIRendererBlurRows* blur, Display* display, Window* window, uint32_t height) {
  if (window->width == 0 || height == 0)
    return false;

  if (window->width > BLUR_MAX_SUPPORTED_DIM || height > BLUR_MAX_SUPPORTED_DIM) {
    error_message("Blur not supported for given dimensions. Skipping blur...");
    return false;
  }

  const uint8_t* src = display->buffer + window->x * BLUR_PIXEL_SIZE + (size_t) window->y * display->pitch;
  blur->cols         = _ui_renderer_blur_reduced_dim(window->width);
  blur->rows         = _ui_renderer_blur_reduced_dim(height);
  blur->row_bytes    = window->width * BLUR_PIXEL_SIZE;
  blur->reduced      = window->background_blur_buffer;
  _ui_renderer_blur_reduced(src, window->width, height, display->pitch, window->background_blur_buffer, blur->cols, blur->rows);
  _ui_renderer_expand_blur_row(blur->reduced, blur->cols, blur->expanded[0]);
  return true;
}
