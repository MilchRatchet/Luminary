#include "device_embedded.h"

#include <stdio.h>

#include "ceb.h"
#include "internal_error.h"

static const char* embedded_file_names[DEVICE_EMBEDDED_FILE_COUNT] = {
  [DEVICE_EMBEDDED_FILE_MOON_ALBEDO] = "moon_albedo.png",
  [DEVICE_EMBEDDED_FILE_MOON_NORMAL] = "moon_normal.png",
  [DEVICE_EMBEDDED_FILE_BLUENOISE1D] = "bluenoise_1D.bin",
  [DEVICE_EMBEDDED_FILE_BLUENOISE2D] = "bluenoise_2D.bin",
  [DEVICE_EMBEDDED_FILE_BRIDGE_LUT]  = "bridge_lut.bin"};

////////////////////////////////////////////////////////////////////
// Common loader
////////////////////////////////////////////////////////////////////

LuminaryResult device_embedded_load(const DeviceEmbeddedFile file, void** data, int64_t* size) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(size);

  uint64_t info = 0;
  ceb_access(embedded_file_names[file], data, size, &info);

  if (info) {
    __RETURN_ERROR(LUMINARY_ERROR_MISSING_DATA, "Failed to embedded file: %s.", embedded_file_names[file]);
  }

  return LUMINARY_SUCCESS;
}
