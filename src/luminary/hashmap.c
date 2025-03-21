#include "hashmap.h"

#include <string.h>

#include "internal_error.h"

/*
 * Must be synced with hashmap.cuh if the hash map is also supposed to be used on the GPU.
 */

#define HASH_MAP_COLLISION_BIT (0x80000000)
#define HASH_MAP_KEY_MASK (~HASH_MAP_COLLISION_BIT)

#define FNV_PRIME (0x01000193u)
#define FNV_OFFSET_BASIS (0x811c9dc5u)

LuminaryResult hash_map_create(HashMap** hash_map) {
  __CHECK_NULL_ARGUMENT(hash_map);

  __FAILURE_HANDLE(host_malloc(hash_map, sizeof(HashMap)));

  (*hash_map)->size = 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult hash_map_reset(HashMap* hash_map, uint32_t size) {
  __CHECK_NULL_ARGUMENT(hash_map);

  if (hash_map->size) {
    __FAILURE_HANDLE(host_free(hash_map->data));
    __FAILURE_HANDLE(host_free(hash_map->occupancy));
  }

  hash_map->size = 0;

  if (size == 0) {
    return LUMINARY_SUCCESS;
  }

  uint32_t map_size = size;

  // Compute next power of 2
  map_size--;
  map_size |= map_size >> 1;
  map_size |= map_size >> 2;
  map_size |= map_size >> 4;
  map_size |= map_size >> 8;
  map_size |= map_size >> 16;
  map_size++;

  hash_map->size = map_size;

  __FAILURE_HANDLE(host_malloc(&hash_map->data, hash_map->size * sizeof(uint32_t)));
  memset(hash_map->data, 0, hash_map->size * sizeof(uint32_t));

  __FAILURE_HANDLE(host_malloc(&hash_map->occupancy, hash_map->size));
  memset(hash_map->occupancy, 0, hash_map->size);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _hash_map_hash(const uint8_t* data, size_t size_of_data, uint32_t* hash_out) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(hash_out);

  uint32_t hash = FNV_OFFSET_BASIS;

  for (size_t offset = 0; offset < size_of_data; offset++) {
    hash ^= (size_t) data[offset];
    hash *= FNV_PRIME;
  }

  *hash_out = hash;

  return LUMINARY_SUCCESS;
}

LuminaryResult hash_map_add(HashMap* hash_map, uint32_t key, const void* data, size_t size_of_data) {
  __CHECK_NULL_ARGUMENT(hash_map);
  __CHECK_NULL_ARGUMENT(data);

  uint32_t hash;
  __FAILURE_HANDLE(_hash_map_hash(data, size_of_data, &hash));

  uint32_t offset = 0;

  while (hash_map->occupancy[(hash + offset) & (hash_map->size - 1)] && (offset != hash_map->size)) {
    offset++;
  }

  if (offset == hash_map->size) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Hashmap ran out of entries.");
  }

  // Mark the original slot as having had a collision.
  if (offset) {
    hash_map->data[hash & (hash_map->size - 1)] |= HASH_MAP_COLLISION_BIT;
  }

  hash = (hash + offset) & (hash_map->size - 1);

  hash_map->data[hash]      = key & HASH_MAP_KEY_MASK;
  hash_map->occupancy[hash] = 1;

  return LUMINARY_SUCCESS;
}

LuminaryResult hash_map_destroy(HashMap** hash_map) {
  __CHECK_NULL_ARGUMENT(hash_map);
  __CHECK_NULL_ARGUMENT(*hash_map);

  if ((*hash_map)->size) {
    __FAILURE_HANDLE(host_free((*hash_map)->data));
    __FAILURE_HANDLE(host_free((*hash_map)->occupancy));
  }

  __FAILURE_HANDLE(host_free(hash_map));

  return LUMINARY_SUCCESS;
}
