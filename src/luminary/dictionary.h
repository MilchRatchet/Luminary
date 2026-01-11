#ifndef LUMINARY_DICTIONARY_H
#define LUMINARY_DICTIONARY_H

#include "utils.h"

struct Dictionary typedef Dictionary;

LuminaryResult dictionary_create(Dictionary** dict);
LuminaryResult dictionary_add_entry(Dictionary* dict, uint32_t id, const char* string);
LuminaryResult dictionary_find_by_name(Dictionary* dict, const char* string, uint32_t* id, bool* found);
LuminaryResult dictionary_find_by_id(Dictionary* dict, uint32_t id, const char** string, bool* found);
LuminaryResult dictionary_remove_entry(Dictionary* dict, uint32_t id);
LuminaryResult dictionary_destroy(Dictionary** dict);

#endif /* LUMINARY_DICTIONARY_H */
