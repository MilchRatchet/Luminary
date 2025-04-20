#include "device_embedded.h"

#include <stdio.h>

#include "ceb.h"
#include "internal_error.h"

// #define DEVICE_GENERATE_EMBEDDED_DATA

static const char* embedded_file_names[DEVICE_EMBEDDED_FILE_COUNT] = {
  [DEVICE_EMBEDDED_FILE_MOON_ALBEDO] = "moon_albedo.png",
  [DEVICE_EMBEDDED_FILE_MOON_NORMAL] = "moon_normal.png",
  [DEVICE_EMBEDDED_FILE_BLUENOISE1D] = "bluenoise_1D.bin",
  [DEVICE_EMBEDDED_FILE_BLUENOISE2D] = "bluenoise_2D.bin",
  [DEVICE_EMBEDDED_FILE_BRIDGE_LUT]  = "bridge_lut.bin",
  [DEVICE_EMBEDDED_FILE_LTC0]        = "ltc_tex0.bin",
  [DEVICE_EMBEDDED_FILE_LTC1]        = "ltc_tex1.bin",
  [DEVICE_EMBEDDED_FILE_LTC2]        = "ltc_tex2.bin"};

////////////////////////////////////////////////////////////////////
// Data generation
////////////////////////////////////////////////////////////////////

#ifdef DEVICE_GENERATE_EMBEDDED_DATA

#include "ltc_amplitude.h"
#include "ltc_m_reparam.h"

static LuminaryResult _device_embedded_generate_ltc0() {
  float* data0;
  __FAILURE_HANDLE(host_malloc(&data0, sizeof(float) * 8 * 8 * 8 * 8 * 4));

  for (int i4 = 0; i4 < 8; ++i4) {
    for (int i3 = 0; i3 < 8; ++i3) {
      for (int i2 = 0; i2 < 8; ++i2) {
        for (int i1 = 0; i1 < 8; ++i1) {
          data0[0 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[0 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data0[1 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[1 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data0[2 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[2 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data0[3 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[3 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
        }
      }
    }
  }

  FILE* file0 = fopen(embedded_file_names[DEVICE_EMBEDDED_FILE_LTC0], "wb");

  if (file0 == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file: %s.", embedded_file_names[DEVICE_EMBEDDED_FILE_LTC0]);
  }

  fwrite(data0, sizeof(float), 8 * 8 * 8 * 8 * 4, file0);

  fclose(file0);

  __FAILURE_HANDLE(host_free(&data0));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_embedded_generate_ltc1() {
  float* data1;
  __FAILURE_HANDLE(host_malloc(&data1, sizeof(float) * 8 * 8 * 8 * 8 * 4));

  for (int i4 = 0; i4 < 8; ++i4) {
    for (int i3 = 0; i3 < 8; ++i3) {
      for (int i2 = 0; i2 < 8; ++i2) {
        for (int i1 = 0; i1 < 8; ++i1) {
          data1[0 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[4 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data1[1 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[5 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data1[2 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[6 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data1[3 + 4 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[7 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
        }
      }
    }
  }

  FILE* file1 = fopen(embedded_file_names[DEVICE_EMBEDDED_FILE_LTC1], "wb");

  if (file1 == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file: %s.", embedded_file_names[DEVICE_EMBEDDED_FILE_LTC1]);
  }

  fwrite(data1, sizeof(float), 8 * 8 * 8 * 8 * 4, file1);

  fclose(file1);

  __FAILURE_HANDLE(host_free(&data1));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_embedded_generate_ltc2() {
  float* data2;
  __FAILURE_HANDLE(host_malloc(&data2, sizeof(float) * 8 * 8 * 8 * 8 * 2));

  for (int i4 = 0; i4 < 8; ++i4) {
    for (int i3 = 0; i3 < 8; ++i3) {
      for (int i2 = 0; i2 < 8; ++i2) {
        for (int i1 = 0; i1 < 8; ++i1) {
          data2[0 + 2 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (anisomats))[8 + 9 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))];
          data2[1 + 2 * (i1 + 8 * (i2 + 8 * (i3 + 8 * i4)))] = ((float*) (ltcamp))[i1 + 8 * (i2 + 8 * (i3 + 8 * i4))];
        }
      }
    }
  }

  FILE* file2 = fopen(embedded_file_names[DEVICE_EMBEDDED_FILE_LTC2], "wb");

  if (file2 == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file: %s.", embedded_file_names[DEVICE_EMBEDDED_FILE_LTC2]);
  }

  fwrite(data2, sizeof(float), 8 * 8 * 8 * 8 * 2, file2);

  fclose(file2);

  __FAILURE_HANDLE(host_free(&data2));

  return LUMINARY_SUCCESS;
}

#endif /* DEVICE_GENERATE_EMBEDDED_DATA */

////////////////////////////////////////////////////////////////////
// Common loader
////////////////////////////////////////////////////////////////////

LuminaryResult device_embedded_load(const DeviceEmbeddedFile file, void** data, int64_t* size) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(size);

#ifdef DEVICE_GENERATE_EMBEDDED_DATA
  switch (file) {
    case DEVICE_EMBEDDED_FILE_LTC0:
      __FAILURE_HANDLE(_device_embedded_generate_ltc0());
      *data = (void*) 0;
      *size = 0;
      return LUMINARY_SUCCESS;
    case DEVICE_EMBEDDED_FILE_LTC1:
      __FAILURE_HANDLE(_device_embedded_generate_ltc1());
      *data = (void*) 0;
      *size = 0;
      return LUMINARY_SUCCESS;
    case DEVICE_EMBEDDED_FILE_LTC2:
      __FAILURE_HANDLE(_device_embedded_generate_ltc2());
      *data = (void*) 0;
      *size = 0;
      return LUMINARY_SUCCESS;
    default:
      break;
  }
#endif /* DEVICE_GENERATE_EMBEDDED_DATA */

  uint64_t info = 0;
  ceb_access(embedded_file_names[file], data, size, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_MISSING_DATA, "Failed to embedded file: %s.", embedded_file_names[file]);
  }

  return LUMINARY_SUCCESS;
}
