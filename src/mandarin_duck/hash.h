#ifndef MANDARIN_DUCK_HASH_H
#define MANDARIN_DUCK_HASH_H

#include "utils.h"

struct Hash {
  uint64_t hash;
} typedef Hash;

void hash_init(Hash* hash);
void hash_string(Hash* hash, const char* string);

#endif /* MANDARIN_DUCK_HASH_H */
