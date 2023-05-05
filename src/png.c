#include "png.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "log.h"
#include "structs.h"
#include "texture.h"
#include "utils.h"
#include "zlib/zlib.h"

enum PNGChunkTypes { CHUNK_IHDR = 1229472850u, CHUNK_IDAT = 1229209940u, CHUNK_gAMA = 1732332865u } typedef PNGChunkTypes;

static inline void _png_write_uint32_big_endian(uint8_t* buffer, const uint32_t value) {
  buffer[0] = value >> 24;
  buffer[1] = value >> 16;
  buffer[2] = value >> 8;
  buffer[3] = value;
}

static void _png_write_header_to_file(FILE* file) {
  uint8_t* header = (uint8_t*) malloc(8);

  header[0] = 0x89u;

  header[1] = 0x50u;
  header[2] = 0x4Eu;
  header[3] = 0x47u;

  header[4] = 0x0Du;
  header[5] = 0x0Au;

  header[6] = 0x1Au;

  header[7] = 0x0Au;

  fwrite(header, 1, 8, file);

  free(header);
}

static void _png_write_IHDR_chunk_to_file(
  FILE* file, const uint32_t width, const uint32_t height, const uint8_t bit_depth, const uint8_t color_type,
  const uint8_t interlace_method) {
  uint8_t* chunk = (uint8_t*) malloc(25);

  _png_write_uint32_big_endian(chunk, 13u);

  chunk[4] = 'I';
  chunk[5] = 'H';
  chunk[6] = 'D';
  chunk[7] = 'R';

  _png_write_uint32_big_endian(chunk + 8, width);
  _png_write_uint32_big_endian(chunk + 12, height);

  chunk[16] = bit_depth;
  chunk[17] = color_type;
  chunk[18] = 0u;
  chunk[19] = 0u;
  chunk[20] = interlace_method;

  _png_write_uint32_big_endian(chunk + 21, (uint32_t) crc32(0, chunk + 4, 17));

  fwrite(chunk, 1, 25, file);

  free(chunk);
}

static void _png_write_sRGB_chunk_to_file(FILE* file) {
  uint8_t* chunk = (uint8_t*) malloc(13);

  _png_write_uint32_big_endian(chunk, 1u);

  chunk[4] = 's';
  chunk[5] = 'R';
  chunk[6] = 'G';
  chunk[7] = 'B';

  // Perceptual sRGB
  chunk[8] = 0;

  _png_write_uint32_big_endian(chunk + 9, (uint32_t) crc32(0, chunk + 4, 5));

  fwrite(chunk, 1, 13, file);

  free(chunk);
}

static void _png_write_gAMA_chunk_to_file(FILE* file) {
  uint8_t* chunk = (uint8_t*) malloc(16);

  _png_write_uint32_big_endian(chunk, 4u);

  chunk[4] = 'g';
  chunk[5] = 'A';
  chunk[6] = 'M';
  chunk[7] = 'A';

  // PNG standard defines this value to be stored in the gAMA chunk if sRGB chunk is present
  _png_write_uint32_big_endian(chunk + 8, 45455u);

  _png_write_uint32_big_endian(chunk + 12, (uint32_t) crc32(0, chunk + 4, 8));

  fwrite(chunk, 1, 16, file);

  free(chunk);
}

static void _png_write_cHRM_chunk_to_file(FILE* file) {
  uint8_t* chunk = (uint8_t*) malloc(44);

  _png_write_uint32_big_endian(chunk, 32u);

  chunk[4] = 'c';
  chunk[5] = 'H';
  chunk[6] = 'R';
  chunk[7] = 'M';

  // PNG standard defines these values to be stored in the cHRM chunk if sRGB chunk is present
  _png_write_uint32_big_endian(chunk + 8, 31270u);
  _png_write_uint32_big_endian(chunk + 12, 32900u);
  _png_write_uint32_big_endian(chunk + 16, 64000u);
  _png_write_uint32_big_endian(chunk + 20, 33000u);
  _png_write_uint32_big_endian(chunk + 24, 30000u);
  _png_write_uint32_big_endian(chunk + 28, 60000u);
  _png_write_uint32_big_endian(chunk + 32, 15000u);
  _png_write_uint32_big_endian(chunk + 36, 6000u);

  _png_write_uint32_big_endian(chunk + 40, (uint32_t) crc32(0, chunk + 4, 36));

  fwrite(chunk, 1, 44, file);

  free(chunk);
}

static void _png_write_IDAT_chunk_to_file(FILE* file, const uint8_t* compressed_image, const uint32_t compressed_length) {
  uint8_t* chunk = (uint8_t*) malloc(12 + compressed_length);

  _png_write_uint32_big_endian(chunk, compressed_length);

  chunk[4] = 'I';
  chunk[5] = 'D';
  chunk[6] = 'A';
  chunk[7] = 'T';

  memcpy(chunk + 8, compressed_image, compressed_length);

  _png_write_uint32_big_endian(chunk + 8 + compressed_length, (uint32_t) crc32(0, chunk + 4, 4 + compressed_length));

  fwrite(chunk, 1, 12 + compressed_length, file);

  free(chunk);
}

static void _png_write_IEND_chunk_to_file(FILE* file) {
  uint8_t* chunk = (uint8_t*) malloc(12);

  _png_write_uint32_big_endian(chunk, 0u);

  chunk[4] = 'I';
  chunk[5] = 'E';
  chunk[6] = 'N';
  chunk[7] = 'D';

  _png_write_uint32_big_endian(chunk + 8, (uint32_t) crc32(0, chunk + 4, 4));

  fwrite(chunk, 1ul, 12ul, file);

  free(chunk);
}

/*
 * Does not support filter yet.
 */
void png_store(
  const char* filename, const uint8_t* image, const uint32_t image_length, const uint32_t width, const uint32_t height,
  const PNGColortype color_type, const PNGBitdepth bit_depth) {
  log_message("Storing png file (%s) Size: %dx%d Depth: %d Colortype: %d", filename, width, height, bit_depth, color_type);
  bench_tic();

  const uint8_t bytes_per_channel = (bit_depth == PNG_BITDEPTH_8) ? 1 : 2;

  uint8_t bytes_per_pixel;
  switch (color_type) {
    case PNG_COLORTYPE_GRAYSCALE:
      bytes_per_pixel = 1 * bytes_per_channel;
      break;
    case PNG_COLORTYPE_TRUECOLOR:
      bytes_per_pixel = 3 * bytes_per_channel;
      break;
    case PNG_COLORTYPE_INDEXED:
      bytes_per_pixel = 1 * bytes_per_channel;
      break;
    case PNG_COLORTYPE_GRAYSCALE_ALPHA:
      bytes_per_pixel = 2 * bytes_per_channel;
      break;
    case PNG_COLORTYPE_TRUECOLOR_ALPHA:
      bytes_per_pixel = 4 * bytes_per_channel;
      break;
    default:
      error_message("Invalid color type: %d", color_type);
      return;
  }

  /* Adding filter byte at the beginning of each scanline */
  uint8_t* filtered_image = (uint8_t*) malloc(image_length + height);
  for (uint32_t i = 0; i < height; i++) {
    filtered_image[i * width * bytes_per_pixel + i] = 0;
    memcpy(filtered_image + i * width * bytes_per_pixel + i + 1, image + i * width * bytes_per_pixel, width * bytes_per_pixel);
  }

  FILE* file = fopen(filename, "wb");

  if (!file) {
    error_message("Failed to create file: %s", filename);
    return;
  }

  _png_write_header_to_file(file);

  _png_write_IHDR_chunk_to_file(file, width, height, bit_depth, color_type, PNG_INTERLACE_OFF);

  _png_write_sRGB_chunk_to_file(file);
  _png_write_gAMA_chunk_to_file(file);
  _png_write_cHRM_chunk_to_file(file);

  uint8_t* compressed_image = (uint8_t*) malloc(image_length + height);

  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree  = Z_NULL;
  defstream.opaque = Z_NULL;

  defstream.avail_in  = (uInt) image_length + height;
  defstream.next_in   = (Bytef*) filtered_image;
  defstream.avail_out = (uInt) image_length + height;
  defstream.next_out  = (Bytef*) compressed_image;

  deflateInit(&defstream, Z_BEST_COMPRESSION);
  deflate(&defstream, Z_FINISH);

  const uLong compressed_length = defstream.total_out;

  deflateEnd(&defstream);

  _png_write_IDAT_chunk_to_file(file, compressed_image, compressed_length);

  _png_write_IEND_chunk_to_file(file);

  fclose(file);

  free(compressed_image);
  free(filtered_image);

  bench_toc("Storing PNG");
}

static inline uint32_t _png_read_uint32_big_endian(uint8_t* buffer) {
  uint32_t result = 0;

  result |= ((uint32_t) buffer[0]) << 24;
  result |= ((uint32_t) buffer[1]) << 16;
  result |= ((uint32_t) buffer[2]) << 8;
  result |= ((uint32_t) buffer[3]);

  return result;
}

static inline uint16_t _png_read_uint16_big_endian(uint8_t* buffer) {
  uint16_t result = 0;

  result |= ((uint32_t) buffer[0]) << 8;
  result |= ((uint32_t) buffer[1]);

  return result;
}

static int _png_verify_header(FILE* file) {
  uint8_t* header = (uint8_t*) malloc(8);

  if (header == (uint8_t*) 0) {
    puts("png.c: Failed to allocate memory!");
    return 1;
  }

  fread(header, 1, 8, file);

  int result = 0;

  result += header[0] ^ 0x89u;
  result += header[1] ^ 0x50u;
  result += header[2] ^ 0x4Eu;
  result += header[3] ^ 0x47u;
  result += header[4] ^ 0x0Du;
  result += header[5] ^ 0x0Au;
  result += header[6] ^ 0x1Au;
  result += header[7] ^ 0x0Au;

  free(header);

  return result;
}

static inline TextureRGBA _png_default_texture() {
  RGBA8* data         = malloc(sizeof(RGBA8) * 4);
  RGBA8 default_pixel = {.r = 255, .g = 0, .b = 0, .a = 255};
  data[0]             = default_pixel;
  data[1]             = default_pixel;
  data[2]             = default_pixel;
  data[3]             = default_pixel;
  TextureRGBA fallback;

  texture_create(&fallback, 2, 2, 1, 2, data, TexDataUINT8, TexStorageCPU);

  return fallback;
}

static inline TextureRGBA _png_default_failure() {
  error_message("File content is corrupted!");
  return _png_default_texture();
}

//////////////////////////////////////////////////////////////////////
// Reconstruction filters
//
// Filters act on bytes and not pixels [PNG Docs 9.2]
//////////////////////////////////////////////////////////////////////

static void _png_reconstruction_1(uint8_t* line, const uint32_t line_length, const int size) {
  if (size > 8) {
    return;
  }

  uint8_t a[8];
  memset(a, 0, 8);

  for (uint32_t j = 0; j < line_length; j++) {
    for (int i = 0; i < size; i++) {
      a[i] += line[size * j + i];
    }

    for (int i = 0; i < size; i++) {
      line[size * j + i] = a[i];
    }
  }
}

static void _png_reconstruction_2(uint8_t* line, const uint32_t line_length, const uint8_t* prior_line, const int size) {
  if (size > 8) {
    return;
  }

  uint8_t a[8];

  for (uint32_t j = 0; j < line_length; j++) {
    for (int i = 0; i < size; i++) {
      a[i] = prior_line[size * j + i];
    }

    for (int i = 0; i < size; i++) {
      line[size * j + i] += a[i];
    }
  }
}

static void _png_reconstruction_3(uint8_t* line, const uint32_t line_length, const uint8_t* prior_line, const int size) {
  if (size > 8) {
    return;
  }

  uint8_t a[8];
  uint8_t b[8];
  memset(a, 0, 8);

  for (uint32_t j = 0; j < line_length; j++) {
    for (int i = 0; i < size; i++) {
      b[i] = prior_line[size * j + i];
    }

    for (int i = 0; i < size; i++) {
      line[size * j + i] += ((uint16_t) a[i] + (uint16_t) b[i]) / 2;
    }

    for (int i = 0; i < size; i++) {
      a[i] = line[size * j + i];
    }
  }
}

static uint8_t paeth(uint8_t a, uint8_t b, uint8_t c) {
  uint8_t pr;
  const int16_t p  = (int16_t) a + (int16_t) b - (int16_t) c;
  const int16_t pa = abs(p - (int16_t) a);
  const int16_t pb = abs(p - (int16_t) b);
  const int16_t pc = abs(p - (int16_t) c);
  if (pa <= pb && pa <= pc) {
    pr = a;
  }
  else if (pb <= pc) {
    pr = b;
  }
  else {
    pr = c;
  }
  return pr;
}

static void _png_reconstruction_4(uint8_t* line, const uint32_t line_length, const uint8_t* prior_line, const int size) {
  if (size > 8) {
    return;
  }

  uint8_t a[8];
  uint8_t b[8];
  uint8_t c[8];
  memset(a, 0, 8);
  memset(c, 0, 8);

  for (uint32_t j = 0; j < line_length; j++) {
    for (int i = 0; i < size; i++) {
      b[i] = prior_line[size * j + i];
    }

    for (int i = 0; i < size; i++) {
      line[size * j + i] += paeth(a[i], b[i], c[i]);
    }

    for (int i = 0; i < size; i++) {
      a[i] = line[size * j + i];
    }

    for (int i = 0; i < size; i++) {
      c[i] = prior_line[size * j + i];
    }
  }
}

TextureRGBA png_load(const char* filename) {
  log_message("Loading png file (%s)", filename);

  FILE* file = fopen(filename, "rb");

  if (!file) {
    error_message("File %s could not be opened!", filename);
    return _png_default_texture();
  }

  if (_png_verify_header(file)) {
    error_message("File header does not correspond to png!");
    fclose(file);
    return _png_default_texture();
  }

  uint8_t* IHDR = (uint8_t*) malloc(25);

  if (IHDR == (uint8_t*) 0) {
    error_message("Failed to allocate memory!");
    fclose(file);
    return _png_default_texture();
  }

  fread(IHDR, 1, 25, file);

  if (_png_read_uint32_big_endian(IHDR) != 13u) {
    free(IHDR);
    fclose(file);
    return _png_default_failure();
  }

  if (_png_read_uint32_big_endian(IHDR + 4) != CHUNK_IHDR) {
    free(IHDR);
    fclose(file);
    return _png_default_failure();
  }

  const uint32_t width         = _png_read_uint32_big_endian(IHDR + 8);
  const uint32_t height        = _png_read_uint32_big_endian(IHDR + 12);
  const uint8_t bit_depth      = IHDR[16];
  const uint8_t color_type     = IHDR[17];
  const uint8_t interlace_type = IHDR[20];

  if ((uint32_t) crc32(0, IHDR + 4, 17) != _png_read_uint32_big_endian(IHDR + 21)) {
    free(IHDR);
    error_message("Texture %s is corrupted!", filename);
    fclose(file);
    return _png_default_failure();
  }

  free(IHDR);

  if (color_type != PNG_COLORTYPE_TRUECOLOR_ALPHA && color_type != PNG_COLORTYPE_TRUECOLOR) {
    error_message("Texture %s is not in RGB/RGBA format!", filename);
    fclose(file);
    return _png_default_texture();
  }

  if (bit_depth != PNG_BITDEPTH_8 && bit_depth != PNG_BITDEPTH_16) {
    error_message("Texture %s does not have 8 or 16 bit depth!", filename);
    fclose(file);
    return _png_default_texture();
  }

  if (interlace_type == PNG_INTERLACE_ADAM7) {
    error_message("Texture %s is interlaced, which is not supported.", filename);
    fclose(file);
    return _png_default_texture();
  }

  const uint32_t byte_per_channel = (bit_depth == PNG_BITDEPTH_8) ? 1u : 2u;
  const uint32_t byte_per_pixel   = byte_per_channel * ((color_type == PNG_COLORTYPE_TRUECOLOR) ? 3u : 4u);

  TextureRGBA result;
  const TextureDataType tex_data_type = (bit_depth == PNG_BITDEPTH_8) ? TexDataUINT8 : TexDataUINT16;
  texture_create(&result, width, height, 1, width, (void*) 0, tex_data_type, TexStorageCPU);

  uint8_t* chunk = (uint8_t*) malloc(8);

  if (chunk == (uint8_t*) 0) {
    error_message("Failed to allocate memory!");
    return _png_default_texture();
  }

  uint8_t* filtered_data     = (uint8_t*) malloc(width * height * byte_per_pixel + height);
  uint8_t* compressed_buffer = (uint8_t*) malloc(2 * (width * height * byte_per_pixel + height));

  uint32_t combined_compressed_length = 0;

  int data_left = (fread(chunk, 1, 8, file) == 8);

  while (data_left) {
    const uint32_t length = _png_read_uint32_big_endian(chunk);

    const uint32_t chunk_type = _png_read_uint32_big_endian(chunk + 4);

    if (chunk_type == CHUNK_IDAT) {
      uint8_t* compressed_data = (uint8_t*) malloc(length + 4);

      _png_write_uint32_big_endian(compressed_data, CHUNK_IDAT);

      fread(compressed_data + 4, 1, length, file);

      fread(chunk, 1, 4, file);

      if ((uint32_t) crc32(0, compressed_data, length + 4) != _png_read_uint32_big_endian(chunk)) {
        return _png_default_failure();
      }

      memcpy(compressed_buffer + combined_compressed_length, compressed_data + 4, length);

      combined_compressed_length += length;

      free(compressed_data);
    }
    else if (chunk_type == CHUNK_gAMA) {
      if (length != 4) {
        error_message("Texture %s has a broken gAMA chunk. Ignoring it.", filename);
        fseek(file, length + 4u, SEEK_CUR);
      }
      else {
        uint8_t data[8];
        data[0] = 'g';
        data[1] = 'A';
        data[2] = 'M';
        data[3] = 'A';
        fread(data + 4, 1, 4, file);
        fread(chunk, 1, 4, file);

        if ((uint32_t) crc32(0, data, length + 4) != _png_read_uint32_big_endian(chunk)) {
          error_message("Texture %s has a broken gAMA chunk. Ignoring it.", filename);
        }
        else {
          const uint32_t gAMA_val = _png_read_uint32_big_endian(data + 4);
          result.gamma            = 100000.0f / ((float) gAMA_val);
        }
      }
    }
    else {
      fseek(file, length + 4u, SEEK_CUR);
    }

    data_left = (fread(chunk, 1, 8, file) == 8);
  }

  fclose(file);
  free(chunk);

  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree  = Z_NULL;
  defstream.opaque = Z_NULL;

  defstream.avail_in  = (uInt) combined_compressed_length;
  defstream.next_in   = (Bytef*) compressed_buffer;
  defstream.avail_out = (uInt) width * height * byte_per_pixel + height;
  defstream.next_out  = (Bytef*) filtered_data;

  inflateInit(&defstream);
  inflate(&defstream, Z_FINISH);

  inflateEnd(&defstream);

  free(compressed_buffer);

  uint8_t* data = malloc(width * height * byte_per_pixel);

  void* line_buffer    = malloc(width * byte_per_pixel);
  void* buffer_address = line_buffer;

  if (line_buffer == (void*) 0) {
    error_message("Failed to allocate memory!");
    free(data);
    free(filtered_data);
    return _png_default_texture();
  }

  memset(line_buffer, 0, width * byte_per_pixel);

  for (uint32_t i = 0; i < height; i++) {
    const uint8_t filter = filtered_data[i * (width * byte_per_pixel + 1)];

    void* data_ptr = (void*) (data + i * width * byte_per_pixel);

    memcpy(data_ptr, filtered_data + 1 + i * (width * byte_per_pixel + 1), width * byte_per_pixel);

    switch (filter) {
      case 0:
        break;
      case 1:
        _png_reconstruction_1(data_ptr, width, byte_per_pixel);
        break;
      case 2:
        _png_reconstruction_2(data_ptr, width, line_buffer, byte_per_pixel);
        break;
      case 3:
        _png_reconstruction_3(data_ptr, width, line_buffer, byte_per_pixel);
        break;
      case 4:
        _png_reconstruction_4(data_ptr, width, line_buffer, byte_per_pixel);
        break;
      default:
        error_message("Invalid filter encountered: %u", filter);
        break;
    }

    line_buffer = data_ptr;
  }

  free(buffer_address);

  void* output_data;
  if (bit_depth == PNG_BITDEPTH_8) {
    output_data = malloc(width * height * sizeof(RGBA8));

    if (color_type == PNG_COLORTYPE_TRUECOLOR) {
      for (uint32_t i = 0; i < width * height; i++) {
        const size_t offset = i * byte_per_pixel;
        const RGBA8 pixel   = {.r = data[offset], .g = data[offset + 1], .b = data[offset + 2], .a = 255};

        ((RGBA8*) output_data)[i] = pixel;
      }
    }
    else {
      memcpy(output_data, data, width * height * sizeof(RGBA8));
    }
  }
  else {
    output_data = malloc(width * height * sizeof(RGBA16));

    if (color_type == PNG_COLORTYPE_TRUECOLOR) {
      for (uint32_t i = 0; i < width * height; i++) {
        const size_t or    = i * byte_per_pixel + 0 * byte_per_channel;
        const size_t og    = i * byte_per_pixel + 1 * byte_per_channel;
        const size_t ob    = i * byte_per_pixel + 2 * byte_per_channel;
        const RGBA16 pixel = {
          .r = _png_read_uint16_big_endian(data + or),
          .g = _png_read_uint16_big_endian(data + og),
          .b = _png_read_uint16_big_endian(data + ob),
          .a = 65535};

        ((RGBA16*) output_data)[i] = pixel;
      }
    }
    else {
      for (uint32_t i = 0; i < width * height; i++) {
        const size_t or    = i * byte_per_pixel + 0 * byte_per_channel;
        const size_t og    = i * byte_per_pixel + 1 * byte_per_channel;
        const size_t ob    = i * byte_per_pixel + 2 * byte_per_channel;
        const size_t oa    = i * byte_per_pixel + 3 * byte_per_channel;
        const RGBA16 pixel = {
          .r = _png_read_uint16_big_endian(data + or),
          .g = _png_read_uint16_big_endian(data + og),
          .b = _png_read_uint16_big_endian(data + ob),
          .a = _png_read_uint16_big_endian(data + oa)};

        ((RGBA16*) output_data)[i] = pixel;
      }
    }
  }

  free(data);

  result.data = output_data;

  free(filtered_data);

  log_message("PNG (%s) Size: %dx%d Depth: %d Colortype: %d", filename, width, height, bit_depth, color_type);

  return result;
}

void png_store_XRGB8(const char* filename, const XRGB8* image, const int width, const int height) {
  uint8_t* buffer = (uint8_t*) malloc(width * height * 3);

  RGB8* buffer_rgb8 = (RGB8*) buffer;
  for (int i = 0; i < height * width; i++) {
    XRGB8 a        = image[i];
    RGB8 result    = {.r = a.r, .g = a.g, .b = a.b};
    buffer_rgb8[i] = result;
  }

  png_store(filename, buffer, width * height * 3, width, height, PNG_COLORTYPE_TRUECOLOR, PNG_BITDEPTH_8);

  free(buffer);
}
