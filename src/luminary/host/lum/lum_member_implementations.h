#ifndef LUMINARY_LUM_MEMBER_IMPLEMENTATIONS_H
#define LUMINARY_LUM_MEMBER_IMPLEMENTATIONS_H

#include "lum_builtins.h"
#include "utils.h"

enum LumMemberType {
  LUM_MEMBER_TYPE_VEC3,
  LUM_MEMBER_TYPE_RGBF,
  LUM_MEMBER_TYPE_UINT32,
  LUM_MEMBER_TYPE_UINT16,
  LUM_MEMBER_TYPE_FLOAT,
  LUM_MEMBER_TYPE_BOOL,
  LUM_MEMBER_TYPE_ENUM
} typedef LumMemberType;

struct LumMemberEntry {
  const char* name;
  size_t offset;
  LumMemberType type;
} typedef LumMemberEntry;

extern const LumMemberEntry* lum_member_entries[LUM_BUILTIN_TYPE_COUNT];

#endif /* LUMINARY_LUM_MEMBER_IMPLEMENTATIONS_H */
