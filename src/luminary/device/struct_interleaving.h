#ifndef LUMINARY_STRUCT_INTERLEAVING_H
#define LUMINARY_STRUCT_INTERLEAVING_H

#include "device_utils.h"

LuminaryResult struct_triangles_interleave(DeviceTriangle* dst, const DeviceTriangle* src, uint32_t count);
LuminaryResult struct_triangles_deinterleave(DeviceTriangle* dst, const DeviceTriangle* src, uint32_t count);

#endif /* LUMINARY_STRUCT_INTERLEAVING_H */
