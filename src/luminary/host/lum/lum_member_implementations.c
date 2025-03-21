#include "lum_member_implementations.h"

#include <stddef.h>

static const LumMemberEntry lum_member_entries_settings[] = {
  {.name = "width", .offset = offsetof(LuminaryRendererSettings, width), .type = LUM_MEMBER_TYPE_UINT32},
  {.name = "height", .offset = offsetof(LuminaryRendererSettings, height), .type = LUM_MEMBER_TYPE_UINT32}};

const LumMemberEntry* lum_member_entries[LUM_BUILTIN_TYPE_COUNT] = {[LUM_BUILTIN_TYPE_SETTINGS] = lum_member_entries_settings};
