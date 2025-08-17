#ifndef LUMINARY_DEVICE_LIBRARY_H
#define LUMINARY_DEVICE_LIBRARY_H

#include "device_utils.h"

struct DeviceLibraryCubin {
  CUlibrary cuda_library;
  uint32_t major;
  uint32_t minor;
} typedef DeviceLibraryCubin;

struct DeviceLibrary {
  ARRAY DeviceLibraryCubin* cubins;
} typedef DeviceLibrary;

LuminaryResult device_library_create(DeviceLibrary** library);
LuminaryResult device_library_add(DeviceLibrary* library, uint32_t major, uint32_t minor);
LuminaryResult device_library_get(DeviceLibrary* library, uint32_t major, uint32_t minor, CUlibrary* cuda_library);  // May return NULL
LuminaryResult device_library_destroy(DeviceLibrary** library);

#endif /* LUMINARY_DEVICE_LIBRARY_H */