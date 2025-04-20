#ifndef CU_LUMINARY_LIGHT_LINKED_LIST_H
#define CU_LUMINARY_LIGHT_LINKED_LIST_H

#include "light_ltc.cuh"
#include "ris.cuh"
#include "utils.cuh"

#define LIGHT_LINKED_LIST_MAX_REFERENCES 32

struct LightLinkedListReference {
  uint32_t id;
  float probability;
} typedef LightLinkedListReference;
LUM_STATIC_SIZE_ASSERT(LightLinkedListReference, 0x08);

__device__ uint32_t light_linked_list_resample(
  const GBufferData data, const LightLinkedListReference stack[LIGHT_LINKED_LIST_MAX_REFERENCES], const uint32_t num_references,
  RISReservoir& reservoir) {
  return 0;
}

#endif /* CU_LUMINARY_LIGHT_LINKED_LIST_H */
