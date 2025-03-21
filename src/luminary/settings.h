#ifndef LUMINARY_SETTINGS_H
#define LUMINARY_SETTINGS_H

#include "utils.h"

LuminaryResult settings_get_default(RendererSettings* settings);
LuminaryResult settings_check_for_dirty(
  const RendererSettings* input, const RendererSettings* old, bool* integration_dirty, bool* buffers_dirty);

#endif /* LUMINARY_SETTINGS_H */
