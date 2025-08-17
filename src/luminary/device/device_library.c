#include "device_library.h"

#include <stdio.h>

#include "ceb.h"
#include "internal_error.h"

LuminaryResult device_library_create(DeviceLibrary** library) {
  __CHECK_NULL_ARGUMENT(library);

  __FAILURE_HANDLE(host_malloc(library, sizeof(DeviceLibrary)));

  __FAILURE_HANDLE(array_create(&(*library)->cubins, sizeof(DeviceLibraryCubin), 4));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_library_add(DeviceLibrary* library, uint32_t major, uint32_t minor) {
  __CHECK_NULL_ARGUMENT(library);

  uint64_t info = 0;

  char cubin_name[4096];
  sprintf(cubin_name, "cuda_kernels_sm_%u%u.cubin", major, minor);

  void* cuda_kernels_data;
  int64_t cuda_kernels_data_length;
  ceb_access(cubin_name, &cuda_kernels_data, &cuda_kernels_data_length, &info);

  if (info) {
    warn_message(
      "Failed to load cubin file for CUDA architecture sm_%u%u. All devices of this architecture will be unavailable. Recompile Luminary "
      "with this architecture enabled.",
      major, minor);
    return LUMINARY_SUCCESS;
  }

  DeviceLibraryCubin cubin;
  cubin.major = major;
  cubin.minor = minor;

  // Tells CUDA that we keep the cubin data unchanged which allows CUDA to not create a copy.
  CUlibraryOption library_option = CU_LIBRARY_BINARY_IS_PRESERVED;

  CUDA_FAILURE_HANDLE(cuLibraryLoadData(&cubin.cuda_library, cuda_kernels_data, 0, 0, 0, &library_option, 0, 1));

  __FAILURE_HANDLE(array_push(&library->cubins, &cubin));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_library_get(DeviceLibrary* library, uint32_t major, uint32_t minor, CUlibrary* cuda_library) {
  __CHECK_NULL_ARGUMENT(library);
  __CHECK_NULL_ARGUMENT(cuda_library);

  uint32_t num_cubins;
  __FAILURE_HANDLE(array_get_num_elements(library->cubins, &num_cubins));

  for (uint32_t cubin_id = 0; cubin_id < num_cubins; cubin_id++) {
    DeviceLibraryCubin cubin = library->cubins[cubin_id];

    if (cubin.major == major && cubin.minor == minor) {
      *cuda_library = cubin.cuda_library;
      return LUMINARY_SUCCESS;
    }
  }

  // If the requested library is not present, return NULL.
  *cuda_library = (CUlibrary) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_library_destroy(DeviceLibrary** library) {
  __CHECK_NULL_ARGUMENT(library);
  __CHECK_NULL_ARGUMENT(*library);

  uint32_t num_cubins;
  __FAILURE_HANDLE(array_get_num_elements((*library)->cubins, &num_cubins));

  for (uint32_t cubin_id = 0; cubin_id < num_cubins; cubin_id++) {
    __FAILURE_HANDLE(cuLibraryUnload((*library)->cubins[cubin_id].cuda_library));
  }

  __FAILURE_HANDLE(array_destroy(&(*library)->cubins));
  __FAILURE_HANDLE(host_free(library));

  return LUMINARY_SUCCESS;
}