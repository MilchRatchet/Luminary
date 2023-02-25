#ifndef DEVICE_H
#define DEVICE_H

#include "stddef.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////
// device.cu
////////////////////////////////////////////////////////////////////
#define device_update_symbol(symbol, data) _device_update_symbol(offsetof(DeviceConstantMemory, symbol), &(data), sizeof(data))
#define device_gather_symbol(symbol, data) _device_gather_symbol(&(data), offsetof(DeviceConstantMemory, symbol), sizeof(data))
void _device_update_symbol(const size_t offset, const void* src, const size_t size);
void _device_gather_symbol(void* dst, const size_t offset, const size_t size);
void device_initialize_random_generators();

////////////////////////////////////////////////////////////////////
// bloom.cuh
////////////////////////////////////////////////////////////////////
void device_bloom_allocate_mips(RaytraceInstance* instance);
void device_bloom_free_mips(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* DEVICE_H */
