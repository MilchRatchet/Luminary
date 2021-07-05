#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#include "utils.cuh"

//===========================================================================================
// Memory Prefetch Functions
//===========================================================================================

__device__
void __prefetch_global_l1(const void* const ptr)
{
  asm("prefetch.global.L1 [%0];" : : "l"(ptr));
}

__device__
void __prefetch_global_l2(const void* const ptr)
{
  asm("prefetch.global.L2 [%0];" : : "l"(ptr));
}

//===========================================================================================
// Minimal Cache Pollution Loads/Stores
//===========================================================================================


#endif /* CU_MEMORY_H */
