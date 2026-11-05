/*
  Copyright (C) 2021-2026 Max Jenke

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef LUMINARY_API_DEBUG_H
#define LUMINARY_API_DEBUG_H

#include <luminary/api_utils.h>
#include <luminary/error.h>
#include <stddef.h>
#include <stdint.h>

struct LuminaryHost;

typedef struct LuminaryHostMemoryAllocation {
  const void* ptr;
  size_t size;
  const char* name;
} LuminaryHostMemoryAllocation;

typedef struct LuminaryHostMemoryAllocationList {
  LuminaryHostMemoryAllocation* allocations;
  uint64_t count;
} LuminaryHostMemoryAllocationList;

typedef struct LuminaryDeviceMemoryAllocation {
  uint64_t device_address;
  size_t size;
  const char* name;
} LuminaryDeviceMemoryAllocation;

typedef struct LuminaryDeviceMemoryAllocationList {
  LuminaryDeviceMemoryAllocation* allocations;
  uint64_t count;
} LuminaryDeviceMemoryAllocationList;

/*
 * Returns a point-in-time copy of live host allocations. Allocation addresses
 * may become stale after this function returns; the copied entries and names
 * remain valid until the list is destroyed.
 */
/* The destination must be zero-initialized and must not own an existing snapshot. */
LUMINARY_API LuminaryResult luminary_host_memory_get_allocations(LuminaryHostMemoryAllocationList* list);

/* Frees the copied entries and names, then clears the list. */
LUMINARY_API LuminaryResult luminary_host_memory_allocation_list_destroy(LuminaryHostMemoryAllocationList* list);

/*
 * Returns a point-in-time copy of allocations made through Luminary's device memory wrappers for the host device-list index.
 * Device addresses are CUDA virtual addresses, not host-dereferenceable pointers. Addresses may become stale after this function returns;
 * copied entries and names remain valid until the list is destroyed.
 */
/* The destination must be zero-initialized and must not own an existing snapshot. */
LUMINARY_API LuminaryResult
  luminary_host_device_memory_get_allocations(struct LuminaryHost* host, uint32_t device_id, LuminaryDeviceMemoryAllocationList* list);

/* Frees the copied entries and names, then clears the list. */
LUMINARY_API LuminaryResult luminary_host_device_memory_allocation_list_destroy(LuminaryDeviceMemoryAllocationList* list);

#endif /* LUMINARY_API_DEBUG_H */
