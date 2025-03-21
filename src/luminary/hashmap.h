#ifndef LUMINARY_HASHMAP_H
#define LUMINARY_HASHMAP_H

#include "utils.h"

struct HashMap {
  uint8_t* occupancy;
  uint32_t* data;
  uint32_t size;
} typedef HashMap;

LuminaryResult hash_map_create(HashMap** hash_map);
LuminaryResult hash_map_reset(HashMap* hash_map, uint32_t size);
LuminaryResult hash_map_add(HashMap* hash_map, uint32_t key, const void* data, size_t size_of_data);
LuminaryResult hash_map_destroy(HashMap** hash_map);

#endif /* LUMINARY_HASHMAP_H */
