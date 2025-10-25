#ifndef CU_LUMINARY_HASHMAP_H
#define CU_LUMINARY_HASHMAP_H

#include "../device_utils.h"

// The top bit of the key marks if this entry had a collision during construction
// This is an optimization to fast forward collision free entries (which will be the case most of the time)
#define HASH_MAP_COLLISION_BIT (0x80000000)
#define HASH_MAP_KEY_MASK (~HASH_MAP_COLLISION_BIT)

#define FNV_PRIME (0x01000193u)
#define FNV_OFFSET_BASIS (0x811c9dc5u)

LUMINARY_FUNCTION uint32_t hash_map_compute_hash_uint2(const uint2 value) {
  uint32_t hash = FNV_OFFSET_BASIS;

  hash ^= (value.x >> 0) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.x >> 8) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.x >> 16) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.x >> 24) & 0xFF;
  hash *= FNV_PRIME;

  hash ^= (value.y >> 0) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.y >> 8) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.y >> 16) & 0xFF;
  hash *= FNV_PRIME;
  hash ^= (value.y >> 24) & 0xFF;
  hash *= FNV_PRIME;

  return hash;
}

LUMINARY_FUNCTION uint32_t
  hash_map_get_uint2_to_uint32(const uint32_t* hashmap_ptr, const uint2* key_value_map, const uint32_t mask, const uint2 value) {
  uint32_t hash = hash_map_compute_hash_uint2(value) & mask;

  // Note: It is assumed that the value is present in the hashmap, else we will end up in an endless loop.
  uint32_t key = __ldg(hashmap_ptr + hash);

  if (key & HASH_MAP_COLLISION_BIT) {
    uint2 value_at_key = __ldg(key_value_map + (key & HASH_MAP_KEY_MASK));

    while ((value_at_key.x != value.x) || (value_at_key.y != value.y)) {
      // Actually this is terrible for the cache because the key_value_map is not indexed linearly
      hash         = (hash + 1) & mask;
      key          = __ldg(hashmap_ptr + hash);
      value_at_key = __ldg(key_value_map + (key & HASH_MAP_KEY_MASK));
    }
  }

  return key & HASH_MAP_KEY_MASK;
}

#endif /* CU_LUMINARY_HASHMAP_H */
