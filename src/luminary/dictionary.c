#include "dictionary.h"

#include <string.h>

#include "internal_error.h"

struct DictionaryEntry {
  uint32_t id;
  const char* string;
} typedef DictionaryEntry;

struct Dictionary {
  ARRAY DictionaryEntry* entries;
};

LuminaryResult dictionary_create(Dictionary** dict) {
  __CHECK_NULL_ARGUMENT(dict);

  __FAILURE_HANDLE(host_malloc(dict, sizeof(Dictionary)));
  memset(*dict, 0, sizeof(Dictionary));

  __FAILURE_HANDLE(array_create(&(*dict)->entries, sizeof(DictionaryEntry), 16));

  return LUMINARY_SUCCESS;
}

LuminaryResult dictionary_add_entry(Dictionary* dict, uint32_t id, const char* string) {
  __CHECK_NULL_ARGUMENT(dict);
  __CHECK_NULL_ARGUMENT(string);

  const size_t string_length = strlen(string);

  char* new_string;
  __FAILURE_HANDLE(host_malloc(&new_string, string_length + 1));

  memcpy(new_string, string, string_length);
  new_string[string_length] = '\0';

  DictionaryEntry new_entry;
  new_entry.id     = id;
  new_entry.string = new_string;

  uint32_t num_entries;
  __FAILURE_HANDLE(array_get_num_elements(dict->entries, &num_entries));

  uint32_t entry_id;
  for (entry_id = 0; entry_id < num_entries; entry_id++) {
    if (dict->entries[entry_id].string == (const char*) 0)
      break;
  }

  if (entry_id < num_entries) {
    dict->entries[entry_id] = new_entry;
  }
  else {
    __FAILURE_HANDLE(array_push(&dict->entries, &new_entry));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult dictionary_find_by_name(Dictionary* dict, const char* string, uint32_t* id, bool* found) {
  __CHECK_NULL_ARGUMENT(dict);
  __CHECK_NULL_ARGUMENT(id);

  uint32_t num_entries;
  __FAILURE_HANDLE(array_get_num_elements(dict->entries, &num_entries));

  uint32_t entry_id;
  for (entry_id = 0; entry_id < num_entries; entry_id++) {
    DictionaryEntry* entry = dict->entries + entry_id;

    if (entry->string == (const char*) 0)
      continue;

    if (strcmp(string, entry->string) == 0)
      break;
  }

  const bool found_entry = entry_id < num_entries;

  if (found != (bool*) 0)
    *found = found_entry;

  if (found_entry)
    *id = dict->entries[entry_id].id;

  return LUMINARY_SUCCESS;
}

LuminaryResult dictionary_find_by_id(Dictionary* dict, uint32_t id, const char** string, bool* found) {
  __CHECK_NULL_ARGUMENT(dict);
  __CHECK_NULL_ARGUMENT(string);

  uint32_t num_entries;
  __FAILURE_HANDLE(array_get_num_elements(dict->entries, &num_entries));

  uint32_t entry_id;
  for (entry_id = 0; entry_id < num_entries; entry_id++) {
    DictionaryEntry* entry = dict->entries + entry_id;

    if (entry->id == id && entry->string != (const char*) 0)
      break;
  }

  const bool found_entry = entry_id < num_entries;

  if (found != (bool*) 0)
    *found = found_entry;

  if (found_entry)
    *string = dict->entries[entry_id].string;

  return LUMINARY_SUCCESS;
}

LuminaryResult dictionary_remove_entry(Dictionary* dict, uint32_t id) {
  __CHECK_NULL_ARGUMENT(dict);

  uint32_t num_entries;
  __FAILURE_HANDLE(array_get_num_elements(dict->entries, &num_entries));

  uint32_t entry_id;
  for (entry_id = 0; entry_id < num_entries; entry_id++) {
    DictionaryEntry* entry = dict->entries + entry_id;

    if (entry->id == id && entry->string != (const char*) 0)
      break;
  }

  if (entry_id == num_entries)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to remove dictionary entry of id %u. Entry does not exist.", id);

  __FAILURE_HANDLE(host_free(&(dict->entries[entry_id].string)));

  return LUMINARY_SUCCESS;
}

LuminaryResult dictionary_destroy(Dictionary** dict) {
  __CHECK_NULL_ARGUMENT(dict);
  __CHECK_NULL_ARGUMENT(*dict);

  uint32_t num_entries;
  __FAILURE_HANDLE(array_get_num_elements((*dict)->entries, &num_entries));

  for (uint32_t entry_id = 0; entry_id < num_entries; entry_id++) {
    DictionaryEntry* entry = (*dict)->entries + entry_id;

    if (entry->string)
      __FAILURE_HANDLE(host_free(&entry->string));
  }

  __FAILURE_HANDLE(array_destroy(&(*dict)->entries));

  __FAILURE_HANDLE(host_free(dict));

  return LUMINARY_SUCCESS;
}
