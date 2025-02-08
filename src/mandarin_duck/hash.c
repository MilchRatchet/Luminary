#include "hash.h"

#include <string.h>

#define FNV_PRIME (0x01000193u)
#define FNV_OFFSET_BASIS (0x811c9dc5u)

void hash_init(Hash* hash) {
  MD_CHECK_NULL_ARGUMENT(hash);

  memset(hash, 0, sizeof(Hash));

  hash->hash = FNV_OFFSET_BASIS;
}

void hash_string(Hash* hash, const char* string) {
  MD_CHECK_NULL_ARGUMENT(hash);

  if (string == (const char*) 0)
    return;

  const size_t string_length = strlen(string);

  for (size_t offset = 0; offset < string_length; offset++) {
    hash->hash ^= (size_t) string[offset];
    hash->hash *= FNV_PRIME;
  }
}
